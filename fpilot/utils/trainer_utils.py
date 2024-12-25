from fpilot.utils.common_utils import jax, jr, typing as tp, random
from flax.training import train_state


def get_prngs(num):
    k = jr.PRNGKey(random.randint(1, 1000))
    if num > 1:
        return jr.split(k, num)
    return k


class FPState(train_state.TrainState):
    lm_trackers: {
        "lt": tp.Any,
        'mt': tp.Any
    }
    deterministic_key: jax.Array
    variables: tp.Any
    val_step: tp.Union[int, jax.Array] = 0
