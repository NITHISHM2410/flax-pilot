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
    def __init__(self, flax_model, input_shape, optimizer, loss_metric_fn, trackers):
        """

        :param flax_model: An instance of flax model.
        :param input_shape: A dict with model's forward pass param's name as key and their shapes as value.
                            The input_shape parameter is used for linen.module.init method, which initializes
                            with float inputs by default. If integer inputs are desired, they must be manually
                            converted to integers within the model call, because the model will always be init with float.

        :param optimizer: An optax optimizer.
        :param loss_metric_fn: A loss function that takes specific inputs and returns:
            (i) Scalar loss to minimized and
            (ii) Nested dict to update loss and metric trackers, with 'lt'(loss tracker) and 'mt'(metric tracker) as parent keys.
                 Value to parent key is a child dict with key as string used to refer tracker and value as a scalar value to update tracker.

        :param trackers: A dict of with parent keys 'lt'(loss tracker) and 'mt'(metric tracker).
                         Value to parent key is a child dict with key as string to refer the tracker and value as instance of tracker.

        parent keys 'lt' and 'mt' shouldn't be changed, child key names can be set to anything, but must match
        between 'trackers' param and 'loss_metric_fn' 2nd return element.



        """
        self.model = flax_model
        self.input_shape = input_shape
        self.loss_metric_fn = loss_metric_fn

        self.optimizer = optimizer
        self.objs = ('main',)
        if isinstance(self.optimizer, dict):
            self.objs = list(self.optimizer.keys())

        self.trackers = trackers
        self.state = None
        self.build()

    def build(self, ):
        for ip_shp in self.input_shape:
            self.input_shape[ip_shp] = jnp.ones(self.input_shape[ip_shp])

        def trace_to_obj_params(params):
            obj_params = list(params['params'].keys())
            assert set(obj_params) == set(self.objs), "Optimizer target params doesn't match the params of the model."
            trace = {'params': dict([p, p] for p in obj_params)}
            return trace

        param_key, global_key = get_prngs(2)
        state = FPState.create(
            tx=self.optimizer if not len(self.objs) > 1 else tx.multi_transform(self.optimizer, trace_to_obj_params),
            apply_fn=self.model.apply,
            params=self.model.init(param_key, deterministic=True, **self.input_shape),
            lm_trackers={'lt': self.trackers['lt'], 'mt': self.trackers['mt']},
            global_key=global_key
        )
        self.state = replicate(state)
        if jax.device_count() > 1:
            self.state = self.state.replace(global_key=get_prngs(jax.device_count()))

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

        if len(self.objs) > 1:
            grads = dict()
            lmd = dict()
            for obj in self.objs:
                (_, sub_lmd), sub_grads = sub_grad_fn(state.params, state.apply_fn, sample, False, prng_key, state.step, obj)
                sub_grads['params'] = {obj: sub_grads['params'][obj]}
                grads = dmerge(grads, sub_grads)
                lmd = dmerge(lmd, sub_lmd)
        else:
            (_, lmd), grads = sub_grad_fn(state.params, state.apply_fn, sample, False, prng_key, state.step)

        grads = jax.lax.pmean(grads, axis_name='devices')
        state = state.apply_gradients(grads=grads)
        state = state.replace(lm_trackers=self.update_met(state.lm_trackers, lmd))
        return state

    @functools.partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(0,))
    def val_step(self, state, sample):
        prng_key = jr.fold_in(state.global_key, -state.val_step)

        if len(self.objs) > 1:
            lmd = dict()
            for obj in self.objs:
                _, sub_lmd = self.loss_metric_fn(state.params, state.apply_fn, sample, False, prng_key, state.step, obj)
                lmd = dmerge(lmd, sub_lmd)
        else:
            _, lmd = self.loss_metric_fn(state.params, state.apply_fn, sample, False, prng_key, state.step)

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
