from fpilot.utils.opt_utils import freeze
from fpilot.utils.trainer_utils import get_prngs, FPState
from fpilot.utils.checkpoint_utils import load_from_ckpt, save_ckpt
from fpilot.utils.tracker_utils import Mean, MeanMetric, ClassMetric
from fpilot.utils.common_utils import time, tqdm, functools, dmerge, gc, jnp, tx, jr, jax, jtu, nn, replicate, typing as tp


class Trainer:
    def __init__(
            self,
            flax_model: nn.Module,
            input_shape: dict[str, tuple|list],
            optimizer: tx.GradientTransformation | dict[str, tx.GradientTransformation],
            loss_metric_fn: tp.Callable,
            trackers: dict[str, dict[str, Mean|MeanMetric|ClassMetric]],
            opt_mask: dict[str, bool] | None = True,
            devices: list[jax.Device] | None = None
    ):
        """
                Args:
                    flax_model: An instance of a `flax.linen.Module`.

                    input_shape: A dict with model's forward pass param's name as key and their shapes as value.
                            The `input_shape` parameter is used for `linen.module.init` method, which initializes
                            with float inputs by default. If integer inputs are desired, they must be manually
                            converted to integers within the model `__call__`, because the model will always be init with float.

                    optimizer:
                        - Single-Objective Training:
                            A single Optax optimizer to optimize all parameters to minimize the loss function.
                        - Multi-Objective Training (e.g., GANs):
                            A dictionary of optimizers. The keys correspond to the sub-model's name within the encapsulated model.
                        - Example `optimizer`:

                            >>> import flax.linen as nn
                            >>> import optax as tx
                            >>>
                            >>> class GAN(nn.Module): # Encapsulated model
                            ...     def setup(self):
                            ...         self.discriminator = Discriminator() # sub-model
                            ...         self.generator = Generator()         # sub-model
                            ...
                            ...     def __call__(self, x, deterministic):
                            ...         pass # Logic
                            >>>
                            >>> # For Multi Objective like GANS, a structure like below must be passed to this param
                            >>> opt_multi = {
                            ...     'generator': tx.adam(1e-4)
                            ...     'discriminator': tx.sgd(1e-3)
                            ... }
                            ...
                            >>> #======================================================================================
                            >>> # For single objective, a structure like below must be passed to this param
                            >>> opt = tx.adam(1e-4)

                    loss_metric_fn:
                        - A function that computes the loss and returns loss, updated mutable variables and tracker updates. All arguments to this function will be
                          passed from the `Trainer` itself. This function definition must simply align with the following format:

                            - This function must accept the following parameters:
                                - `param_1`: `flax.linen.Module` parameters.
                                - `param_2`: Mutable variables like BatchNorm stats of `flax.linen.Module`. If no Batchnorm like layer, value received will be an empty dict.
                                - `param_3`: `flax.training.TrainState.apply_fn` fn to call model's methods.
                                - `param_4`: Input values as a dictionary, tuple, or list.
                                - `param_5`: Deterministic mode of the model. During training steps, value received here will be `False` and during eval the value received will be `True`.
                                - `param_6`: A `jax.random.PRNGKey` for stochastic layers like Dropout.
                                - `param_7`: Objective to be minimized (sub-model's name). Defaults to 'params` for single-objective cases.
                                             For multi-objective cases, the keys from the `optimizer` dictionary are passed iteratively such that this function will be called multiple times with each time receving different objective name.
                                - `param_8`: The current optimizer step, which can be used for dynamic loss weighting.

                            - The function must return:
                                - `return_1`: A scalar loss to be minimized. For multi-objective cases, the `obj` parameter can be used to condition the loss return.
                                - `return_2`: Updated `variables` obtained after model's forward pass- `apply(params, x...)`. If no Batchnorm like layer, just return empty dict.
                                - `return_3`: Tracker updates in the same structure as the `trackers` parameter, but with leaves containing the updates for the tracker.

                            - The param names and return names are free to be changed. The functionality depends on the order.

                            - Example function def for `loss_metric_fn` without variables:

                                >>> def loss_fn(params, variables, apply, sample, deterministic, deterministic_rng_key, step, objective):
                                ...     x, y = sample
                                ...     yp = apply(params, x, deterministic=deterministic, rngs={'dropout': deterministic_rng_key})
                                ...     loss = ((y-yp)**2).mean()
                                ...     return loss,  dict(), {'lt': {'mean': loss}, 'mt': {'mae':(y, yp)}}



                            - Example function def for `loss_metric_fn` with variables:

                                >>> def loss_fn(params, variables, apply, sample, deterministic, deterministic_rng_key, step, objective):
                                ...     x, y = sample
                                ...
                                ...     # For the below line - `yp`: model's actual output. `variables`: Updated vars. Automatically received when calling `apply`. No need to return anything from model manually.
                                ...     yp, variables = apply(params|variables, x, deterministic=deterministic, rngs={'dropout': deterministic_rng_key}, mutable=list(variables.keys()))
                                ...     loss = ((y-yp)**2).mean()
                                ...     return loss,  variables, {'lt': {'mean': loss}, 'mt': {'mae':(y, yp)}}


                    trackers:
                        A dictionary with the following structure:
                        - Keys: `lt` (loss tracker), `mt` (metric tracker). These keys must not be changed.
                        - Values: Child dictionaries, where:
                            - Each key is a string referring to a tracker. These are customizable but must match between
                              the `trackers` parameter and the second return value of the `loss_metric_fn`.
                            - Each value is an instance of a tracker.

                        - Example `trackers` dict:

                            >>> from fpilot import Trackers
                            >>>
                            >>> tracker_dict = {
                            ...     'lt': {'mean': Trackers.Mean()}
                            ...     'mt': {'mae': Trackers.MAE()}
                            ...}

                    devices:
                        List of `jax.Device`, indicating across which devices should training be distributed. Default value `None` indicates all devices.


                """

        self.model = flax_model
        self.loss_metric_fn = loss_metric_fn
        self.trackers = trackers
        self.opt_mask = opt_mask

        self.input_shape = {k: jnp.array(input_shape[k]) for k in input_shape}

        self.objectives = ['params', ]
        if isinstance(optimizer, dict):
            self.optimizer = optimizer
            self.objectives = list(k for k in self.optimizer)
            self.opt_trace = {'params': {k: k for k in self.optimizer}}

        else:
            self.optimizer = {'params': optimizer}
            self.opt_trace = {'params': 'params'}

        self.optimizer = tx.multi_transform(self.optimizer, self.opt_trace)
        self.optimizer = freeze(self.optimizer, self.opt_mask) if isinstance(self.opt_mask, dict) else self.optimizer
        self.multi_obj = len(self.objectives) > 1

        self.devices = jax.devices() if devices is None else devices

        @jax.jit
        def init(ip_shapes):
            return self.model.init(get_prngs(1), deterministic=True, **ip_shapes)

        params = init(jtu.tree_map(jnp.ones, self.input_shape))

        if len(params) == 1:
            variables = {'variables': {}}
        else:
            params, variables = {'params': params['params']}, {'variables': {
                k: params[k] for k in params if k!='params'
            }}

        @jax.jit
        def state_init(trainable_params):
            return FPState.create(
                tx=self.optimizer,
                apply_fn=self.model.apply,
                params=trainable_params,
                lm_trackers=self.trackers,
                deterministic_key=get_prngs(1),
                variables=variables
            )

        self.state = replicate(state_init(params), self.devices)

        if len(self.devices) > 1:
            deterministic_key = get_prngs(len(self.devices))
            self.state = self.state.replace(deterministic_key=deterministic_key)

        self.train_step = functools.partial(jax.pmap, axis_name='devices', devices=self.devices)(self._train_step_impl)
        self.val_step = functools.partial(jax.pmap, axis_name='devices', devices=self.devices)(self._val_step_impl)
        self.reset_trackers = functools.partial(jax.pmap, axis_name='devices', devices=self.devices)(self._reset_trackers_impl)


    def transfer_params(self, path, params):
        """
        Transfers the params from `params` to `path`.

        :param path: List of string where each string represents the path(dict keys) to the param in TrainState to be replaced. Keys must be joined with `/` .
        :param params: List of replacement params where each param is replaced at the corresponding inner list in `path` .

        Note: Both model weights (params) and variables (batch norm states) can be included.
        """
        new_params = self.state.params | self.state.variables
        params = iter(params)

        for pth in path:
            temp_cur_params = new_params
            pth_list = pth.split('/')
            for p_i in pth_list[:-1]:
                temp_cur_params = temp_cur_params[p_i]
            temp_cur_params[pth_list[-1]] = replicate(next(params), self.devices)

        self.state = self.state.replace(
            params={'params': new_params['params']},
            variables={'variables': new_params['variables']}
        )


    def __call__(self, rng, tensor_inputs, method='__call__', **kwargs):
        """
        Applies calls to model's methods.

        :param rng: Dict of prng values where keys are rng names that's used in the model.
        :param method: String, Name of the model's method to call. At default, this method is '__call__'.
        :param tensor_inputs: Input to model's 'method' as dict with key as param names and value as tensors.
        :param kwargs: Any constant args to model's 'method'. Like passing 'deterministic' value for Dropout.

        :return: output from model's 'method'.
        """

        @functools.partial(jax.pmap)
        def call(input_data, state):
            return state.apply_fn(
                state.params | state.variables['variables'],
                **input_data, method=method, rngs=rng, **kwargs
            )

        output = call(tensor_inputs, self.state)
        return output

    def compute_loss(self, *args):
        loss_fn_returns = self.loss_metric_fn(*args)
        loss, var, lmv = loss_fn_returns
        aux_data = {'lmv': lmv, 'variables': var}
        return loss, aux_data


    def _train_step_impl(self, state, sample):
        prng_key = jr.fold_in(state.deterministic_key, state.step)
        sub_grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        aux_data = dict()
        grads = jtu.tree_map(jnp.zeros_like, state.params)
        model_args = (state.params, state.variables['variables'])
        common_args = (state.apply_fn, sample, False, prng_key, state.step)

        for obj in self.objectives:
            common_args_mul_obj = common_args + (obj, )
            (_, sub_aux_data), sub_grads = sub_grad_fn(*model_args, *common_args_mul_obj)
            aux_data = dmerge(aux_data, sub_aux_data)

            if self.multi_obj:
                grads['params'][obj] = sub_grads['params'][obj]
            else:
                grads['params'] = sub_grads['params']

        grads = jax.lax.pmean(grads, axis_name='devices')
        state = state.apply_gradients(grads=grads)
        state = state.replace(variables={'variables': jax.lax.pmean(aux_data['variables'], axis_name='devices')})
        state = state.replace(lm_trackers=self.update_trackers(state.lm_trackers, aux_data['lmv']))
        return state


    def _val_step_impl(self, state, sample):
        prng_key = jr.fold_in(state.deterministic_key, -state.step)
        aux_data = dict()
        model_args = (state.params, state.variables['variables'])
        common_args = (state.apply_fn, sample, True, prng_key, state.step)

        for obj in self.objectives:
            common_args_mul_obj = common_args + (obj, )
            _, sub_aux_data = self.compute_loss(*model_args, *common_args_mul_obj)
            aux_data = dmerge(aux_data, sub_aux_data)

        state = state.replace(variables={'variables': jax.lax.pmean(aux_data['variables'], axis_name='devices')})
        state = state.replace(lm_trackers=self.update_trackers(state.lm_trackers, aux_data['lmv']))
        return state


    def _reset_trackers_impl(self, state):
        lf, mf = state.lm_trackers['lt'], state.lm_trackers['mt']
        for L in lf:
            lf[L].reset()
        for M in mf:
            mf[M].reset()
        return state.replace(lm_trackers={'lt': lf, 'mt': mf})


    @staticmethod
    def update_trackers(lmf, lmv):
        lf, mf = lmf['lt'], lmf['mt']
        lv, mv = lmv['lt'], lmv['mt']
        for L in lf:
            lf[L].update(lv[L])
        for M in mf:
            if isinstance(mv[M], tuple):
                mf[M].update(*mv[M])
            elif isinstance(mv[M], dict):
                mf[M].update(**mv[M])
            else:
                mf[M].update(mv[M])

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
        self.state = self.reset_trackers(self.state)
        ST = time.time()

        for _ in range(val_steps):
            sample = next(val_ds)
            self.state = self.val_step(self.state, sample)

        print("Val: ", self.print_trackers(time.time() - ST))

    def train(self, epochs, train_ds, val_ds, t_steps, v_steps, ckpt_dir, max2keep=3):
        """
        Executes training loop and performs validation after train loop.

        :param epochs: Number of epochs.
        :param train_ds: Training Repeated tf.data.Dataset as numpy iterator.
        :param val_ds:  Validation Repeated tf.data.Dataset as numpy iterator.
        :param t_steps: Steps per epoch for training dataset.
        :param v_steps: Steps per epoch for validation dataset.
        :param ckpt_dir: Folder to save checkpoint during training. Set to None to disable checkpoint saving during training.
        :param max2keep: Number of checkpoints to preserve in directory throughout training.

        """

        for epoch in range(epochs):

            ################################################ TRAIN #####################################################
            gc.collect()
            self.state = self.reset_trackers(self.state)
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
        if len(self.devices) > 1:
            self.state = self.state.replace(deterministic_key=get_prngs(len(self.devices)))
