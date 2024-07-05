import gc
import jax
import time
import random
import functools
import optax as tx
from tqdm import tqdm
import jax.random as jr
import jax.numpy as jnp
from typing import Any, Union
from mergedeep import merge as dmerge
from flax.training import train_state
from fpilot.opt_utils.freezer import freeze
from fpilot.checkpoints.checkpoint import *


def get_prngs(num):
    k = jr.PRNGKey(random.randint(1, 1000))
    if num > 1:
        return jr.split(k, num)
    return k


class FPState(train_state.TrainState):
    lm_trackers: {
        'lt': Any,
        'mt': Any
    }
    global_key: jax.Array
    val_step: Union[int, jax.Array] = 0


class Trainer:
    def __init__(self, flax_model, input_shape, optimizer, loss_metric_fn, trackers, opt_mask=True):
        """

        :param flax_model: An instance of flax model.
        :param input_shape: A dict with model's forward pass param's name as key and their shapes as value.
                            The input_shape parameter is used for linen.module.init method, which initializes
                            with float inputs by default. If integer inputs are desired, they must be manually
                            converted to integers within the model call, because the model will always be init with float.

        :param optimizer:
            - In single-objective training: It must be a single Optax optimizer. This optimizer is applied to
              all the parameters to minimize the loss function.
            - In multi-objective training (e.g., GANs): It must be a dictionary of optimizers. The keys of this
              dictionary correspond to the parameter's key in the Flax `TrainState.params` that need to be optimized,
              and the values are the respective Optax optimizers for those parameters. This allows different sets of
              parameters to be optimized with different strategies.
            - Nested dict can also be passed where the nested keys trace the path to the parameter's key whose children
              are to be optimized and leaf of this nested dict is the optimizer.

        :param loss_metric_fn: A loss function that takes specific inputs and returns:
            (i) Scalar loss to minimized and
            (ii) Nested dict to update loss and metric trackers, with 'lt'(loss tracker) and 'mt'(metric tracker) as parent keys.
                 Value to parent key is a child dict with key as string used to refer tracker and value as a scalar value to update tracker.

        :param trackers: A dict of with parent keys 'lt'(loss tracker) and 'mt'(metric tracker).
                         Value to parent key is a child dict with key as string to refer the tracker and value as instance of tracker.

        parent keys 'lt' and 'mt' shouldn't be changed, child key names can be set to anything, but must match
        between 'trackers' param and 'loss_metric_fn' 2nd return element.

        :param opt_mask: A Pytree with similar structure of TrainState.params or its prefix with boolean leaves to
                         indicate which params must be frozen and which must be trainable. True -> trainable
                         and False -> frozen.
                         Or Simply a boolean value to represent all the params.


        """
        self.model = flax_model
        self.input_shape = input_shape
        self.loss_metric_fn = loss_metric_fn
        self.optimizer = optimizer
        self.trackers = trackers
        self.opt_mask = opt_mask
        self.opt_trace = None
        self.state = None
        self.objs = None
        self.build()

    @staticmethod
    def trace_to_obj_params(optimizer, opt_mask):
        if isinstance(opt_mask, dict):
            keys = list(opt_mask['params'].keys())
            if isinstance(keys, bool):
                opt_mask = keys

        if isinstance(opt_mask, bool):
            if opt_mask is False:
                raise Exception("opt_mask is set to false, so whole model is frozen.")

        if isinstance(optimizer, tx.GradientTransformation):
            optimizer = {'params': optimizer}

        if len(optimizer) == 1 and list(optimizer.keys())[0] == 'params':
            return [['params', ], ], {'params': 'params'}, {'params': list(optimizer.values())[0]}, opt_mask

        trace = {}
        objs = []
        flatten_optimizer = {}

        def recursive_trace(path, result, obj_path):
            for key, value in path.items():
                if isinstance(value, dict):
                    result[key] = {}
                    obj_path.append(key)
                    recursive_trace(value, result[key], obj_path)
                    obj_path = ['params', ]

                else:
                    result[key] = key

                    if isinstance(value, tx.GradientTransformation):
                        obj_path.append(key)
                        objs.append(obj_path)
                        obj_path = ['params', ]
                        flatten_optimizer[key] = path[key]

                    else:
                        raise Exception(
                            "Leaves of optimizer dict must be optax.GradientTransformation instance or False(weights freeze)")

        recursive_trace(optimizer, trace, ['params', ])
        trace = {'params': trace}
        return objs, trace, flatten_optimizer, opt_mask

    def build(self, ):
        for ip_shp in self.input_shape:
            self.input_shape[ip_shp] = jnp.ones(self.input_shape[ip_shp])

        self.objs, self.opt_trace, self.optimizer, self.opt_mask = self.trace_to_obj_params(self.optimizer,
                                                                                            self.opt_mask)
        param_key, global_key = get_prngs(2)

        optimizer = tx.multi_transform(self.optimizer, self.opt_trace)

        # for backward compatibility
        optimizer = freeze(optimizer, self.opt_mask) if isinstance(self.opt_mask, dict) else optimizer

        state = FPState.create(
            tx=optimizer,
            apply_fn=self.model.apply,
            params=self.model.init(param_key, deterministic=True, **self.input_shape),
            lm_trackers={'lt': self.trackers['lt'], 'mt': self.trackers['mt']},
            global_key=global_key
        )
        self.state = replicate(state)
        if jax.device_count() > 1:
            self.state = self.state.replace(global_key=get_prngs(jax.device_count()))

    def transfer_params(self, path, params):
        """
        Transfers the params from `params` to `path`.

        :param path: Nested list where each inner list represents the path(dict keys) to the param in TrainState to be replaced.
        :param params: List of replacement params where each param is replaced at the corresponding inner list in `path` .
        """
        new_params = self.state.params
        params = iter(params)

        for pth in path:
            temp_cur_params = new_params
            for p_i in pth[:-1]:
                temp_cur_params = temp_cur_params[p_i]
            temp_cur_params[pth[-1]] = replicate(next(params))

        self.state = self.state.replace(params=new_params)

    def __call__(self, rngs, tensor_inputs, method='__call__', **kwargs):
        """
        Applies calls to model's methods.

        :param rngs: Dict of prng values where keys are rng names that's used in the model.
        :param method: String, Name of the model's method to call. At default, this method is '__call__'.
        :param tensor_inputs: Input to model's 'method' as dict with key as param names and value as tensors.
        :param kwargs: Any constant args to model's 'method'. Like passing 'deterministic' value for Dropout.

        :return: output from model's 'method'.
        """

        @functools.partial(jax.pmap)
        def call(input_data, state):
            return state.apply_fn(state.params, **input_data, method=method, rngs=rngs, **kwargs)

        output = call(tensor_inputs, self.state)
        return output

    @functools.partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(0,))
    def train_step(self, state, sample):
        prng_key = jr.fold_in(state.global_key, state.step)
        sub_grad_fn = jax.value_and_grad(self.loss_metric_fn, has_aux=True)

        grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        lmd = dict()

        for obj in self.objs:
            if len(self.objs) > 1:
                (_, sub_lmd), sub_grads = sub_grad_fn(state.params, state.apply_fn, sample, False, prng_key, state.step,
                                                      "_".join(obj[1:]))
            else:
                (_, sub_lmd), sub_grads = sub_grad_fn(state.params, state.apply_fn, sample, False, prng_key, state.step)

            path_grads = grads
            for gp in obj[:-1]:
                path_grads = path_grads[gp]
                sub_grads = sub_grads[gp]

            path_grads[obj[-1]] = sub_grads[obj[-1]]
            lmd = dmerge(lmd, sub_lmd)

        grads = jax.lax.pmean(grads, axis_name='devices')
        state = state.apply_gradients(grads=grads)
        state = state.replace(lm_trackers=self.update_met(state.lm_trackers, lmd))
        return state

    @functools.partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(0,))
    def val_step(self, state, sample):
        prng_key = jr.fold_in(state.global_key, -state.val_step)

        lmd = dict()
        for obj in self.objs:
            if len(self.objs) > 1:
                _, sub_lmd, = self.loss_metric_fn(state.params, state.apply_fn, sample, True, prng_key, state.step,
                                                  "_".join(obj[1:]))
            else:
                _, sub_lmd, = self.loss_metric_fn(state.params, state.apply_fn, sample, True, prng_key, state.step)
            lmd = dmerge(lmd, sub_lmd)

        state = state.replace(val_step=state.val_step + 1)
        state = state.replace(lm_trackers=self.update_met(state.lm_trackers, lmd))
        return state

    @functools.partial(jax.pmap, static_broadcasted_argnums=(0,))
    def tracker_reset(self, state):
        lf, mf = state.lm_trackers['lt'], state.lm_trackers['mt']
        for L in lf:
            lf[L].reset()
        for M in mf:
            mf[M].reset()
        return state.replace(lm_trackers={'lt': lf, 'mt': mf})

    @staticmethod
    def update_met(lmf, lmv):
        lf, mf = lmf['lt'], lmf['mt']
        lv, mv = lmv['lt'], lmv['mt']
        for L in lf:
            lf[L].update(lv[L])
        for M in mf:
            if isinstance(mv[M], tuple):
                mf[M].update(*mv[M])
            elif isinstance(mv[M], dict):
                mf[M].update(**mv[M])

        return {'lt': lf, 'mt': mf}

    def compute_trackers(self):
        """

        :return: results of loss and metric trackers.
        """

        @functools.partial(jax.pmap)
        def compute(state):
            lf, mf = state.lm_trackers['lt'], state.lm_trackers['mt']
            for L in lf:
                lf[L] = lf[L].compute()
            for M in mf:
                mf[M] = mf[M].compute()
            return {'lt': lf, 'mt': mf}

        return compute(self.state)

    def print_trackers(self, time_taken=None):
        """
        A function to display results during training after each epoch.
        :param time_taken: time taken to execute single train loop or val loop.

        :return: A string, that contains results of trackers along with time taken to execute loop.
        """
        trackers = self.state.lm_trackers['lt'] | self.state.lm_trackers['mt']
        output = ""
        for k in trackers:
            output += (k + ": " + str(trackers[k].compute().mean()) + ", ")
        if time_taken:
            output += ("time: " + str(time_taken))
        return output

    def evaluate(self, val_ds, val_steps):
        """
        An evaluation loop and prints a string that contains results of validation loop.


        :param val_ds: validation tf.data.Dataset as numpy iterator.
        :param val_steps: steps per epoch for validation dataset.
        """
        self.state = self.tracker_reset(self.state)
        ST = time.time()

        for _ in range(val_steps):
            sample = next(val_ds)
            self.state = self.val_step(self.state, sample)

        print("Val: ", self.print_trackers(time.time() - ST))

    def train(self, epochs, train_ds, val_ds, t_steps, v_steps, ckpt_dir, max2keep=3):
        """
        Executes training loop and performs validation after train loop.

        :param epochs: Number of epochs.
        :param train_ds: Training tf.data.Dataset as numpy iterator.
        :param val_ds:  Validation tf.data.Dataset as numpy iterator.
        :param t_steps: steps per epoch for training dataset.
        :param v_steps: steps per epoch for validation dataset.
        :param ckpt_dir: folder to save checkpoint during training. Set to None to disable checkpoint saving during training.
        :param max2keep: number of checkpoints to preserve in directory throughout training.

        """
        train_ds, val_ds = iter(train_ds), iter(val_ds)
        for epoch in range(epochs):

            ################################################ TRAIN #####################################################
            gc.collect()
            self.state = self.tracker_reset(self.state)
            ST = time.time()

            for _ in tqdm(range(t_steps), leave=True, position=0, desc="Epoch {0}".format(epoch + 1)):
                sample = next(train_ds)
                self.state = self.train_step(self.state, sample)

            print("Train: ", self.print_trackers(time.time() - ST))

            ################################################  VAL  #####################################################
            self.evaluate(val_ds, v_steps)
            ############################################################################################################
            if ckpt_dir:
                self.save_state(ckpt_dir, max2keep)

    def save_state(self, save_dir, max2keep=3):
        """
        Saves current 'FPState' (TrainState).

        :param save_dir: folder to save checkpoint.
        :param max2keep: number of checkpoints to preserve throughout training.
        """
        save_result = save_ckpt(save_dir, self.state, max2keep)
        if save_result:
            print("State saved..")

    def load_state(self, save_dir, step, max2keep=3):
        """
        Loads saved checkpoint into 'FPState' (TrainState).

        :param save_dir: folder containing checkpoints.
        :param step: checkpoint step to restore.
        :param max2keep: number of checkpoints to preserve throughout training.

        """
        self.state = load_from_ckpt(step, save_dir, self.state, max2keep)
        if jax.device_count() > 1:
            self.state = self.state.replace(global_key=get_prngs(jax.device_count()))
