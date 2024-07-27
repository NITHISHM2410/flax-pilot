from fpilot.common import jax, jr, jnp, replicate, unreplicate, Any, Union, random
from flax.training import train_state
import optax as tx


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
    variables: Any
    val_step: Union[int, jax.Array] = 0
