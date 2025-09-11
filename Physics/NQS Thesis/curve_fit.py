from flax import nnx
import jax.numpy as jnp
from optax import adamw

from models.ffn import FFN

def loss_fn(model: FFN, batch):
    return 1/2 * ((model(batch['x']) - batch['y'])**2).mean()

@nnx.jit
def train_step(model: FFN, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grad = grad_fn(model, batch)
    optimizer.update(grad)

def nn_fit(x, y, model: FFN, lr: float, n_iter: int):
    optimizer = nnx.Optimizer(model, adamw(lr))
    batch = {'x': jnp.array(x).reshape(-1, 1), 'y': jnp.array(y)}
    for iter in range(n_iter):
        train_step(model, optimizer, batch)

    return model
