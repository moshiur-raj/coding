from flax import nnx
import jax.numpy as jnp
from typing import Callable

def log_cosh(x):
    """
    Logarithm of the hyperbolic cosine, implemented in a more stable way.
    """
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class RBM(nnx.Module):
    def __init__(
            self,
            in_features: int,
            alpha: int = 1,
            activation: Callable = log_cosh,
            kernel_init: Callable = nnx.initializers.normal(),
            bias_init: Callable = nnx.initializers.zeros_init(),
            seed: int = 0,
            param_dtype: jnp.dtype = jnp.float64
            ):

        self.in_features = in_features
        self.alpha = alpha
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.seed = seed
        self.param_dtype = param_dtype

        gen_key = nnx.Rngs(self.seed)

        self.linear = nnx.Linear(
                in_features=self.in_features,
                out_features=self.alpha*self.in_features,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                rngs=nnx.Rngs(gen_key()),
                param_dtype=self.param_dtype,
                )

        self.visible_bias = nnx.Param(
                self.kernel_init(
                    shape=(self.in_features,),
                    key=gen_key(),
                    dtype=self.param_dtype,
                    )
                )

    def __call__(self, spins: jnp.ndarray) -> jnp.ndarray:
        x = self.linear(spins)
        x = self.activation(x)
        x = x.sum(axis=-1)

        x += spins @ self.visible_bias

        return x

class RBMModPhase(nnx.Module):
    def __init__(
            self,
            in_features: int,
            alpha: int = 1,
            activation: Callable = log_cosh,
            kernel_init: Callable = nnx.initializers.normal(),
            bias_init: Callable = nnx.initializers.zeros_init(),
            seed: int = 0,
            param_dtype: jnp.dtype = jnp.float64
            ):

        self.in_features = in_features
        self.alpha = alpha
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.seed = seed
        self.param_dtype = param_dtype

        self.real = RBM(
                in_features = self.in_features,
                alpha = self.alpha,
                activation = self.activation,
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                seed = self.seed,
                param_dtype = self.param_dtype,
                )

        self.imag = RBM(
                in_features = self.in_features,
                alpha = self.alpha,
                activation = self.activation,
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                seed = self.seed + 1,
                param_dtype = self.param_dtype,
                )

    def __call__(self, spins: jnp.ndarray) -> jnp.ndarray:
        return self.real(spins) + 1j * self.imag(spins)
