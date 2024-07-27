from optax._src import base
from fpilot.common import NamedTuple, Any, Union, jtu, jnp


class MaskedState(NamedTuple):
    inner_state: Any


class MaskedNode(NamedTuple):
    """

    """
