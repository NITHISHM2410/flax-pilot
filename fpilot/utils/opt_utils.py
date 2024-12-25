from fpilot.utils.common_utils import typing as tp, jtu, jnp, tx
from optax._src import base



class MaskedState(tp.NamedTuple):
    inner_state: tp.Any


class MaskedNode(tp.NamedTuple):
    """

    """


def freeze(inner: base.GradientTransformation, mask: tp.Union[base.PyTree]) -> base.GradientTransformationExtraArgs:
    def mask_pytree(pytree, mask_tree):
        return jtu.tree_map(lambda m, p: p if m else MaskedNode(), mask_tree, pytree)

    def init_fn(params):
        masked_params = mask_pytree(params, mask)
        return MaskedState(inner_state=inner.init(masked_params))

    def update_fn(updates, state, params):
        masked_updates = mask_pytree(updates, mask)
        masked_params = None if params is None else mask_pytree(params, mask)
        new_masked_updates, new_inner_state = inner.update(masked_updates, state.inner_state, masked_params)
        new_updates = jtu.tree_map(
            lambda m, u, p: u if m else jtu.tree_map(jnp.zeros_like, p),
            mask, new_masked_updates, params)
        return new_updates, MaskedState(inner_state=new_inner_state)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)