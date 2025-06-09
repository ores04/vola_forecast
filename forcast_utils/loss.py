from forcast_utils.models import LSTM
from jax import numpy as jnp


def t_loss_fn(model: LSTM, x: jnp.ndarray, y: jnp.ndarray, v: int) -> jnp.ndarray:
    """ This function implements the T-loss function. Which is a loss function modeled after the Student's t-distribution.
    v is the degrees of freedom parameter. The lower the v, the heavier the tails of the distribution."""
    assert v > 2, "Degrees of freedom v must be greater than 2 for the T-loss function to be well-defined."
    y_pred = model(x) # TODO we should look if y pred is the variance or the std deviation
    loss = jnp.mean(jnp.log(jnp.square(y_pred))/2 + (v+1)/2 * jnp.log(1 + (y/((v -2) * jnp.square(y_pred)))))
    return loss