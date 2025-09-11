from flax import nnx
import netket.nn as nknn
from math import lcm
import jax
import jax.numpy as jnp
from typing import Callable

class FFN(nnx.Module):

    def __init__(
            self,
            features: list | tuple,
            activation: Callable = nnx.gelu,
            kernel_init: Callable = nnx.initializers.lecun_normal(),
            bias_init: Callable = nnx.initializers.zeros_init(),
            seed: int = 0,
            param_dtype: jnp.dtype = jnp.float64
            ):

        if features[-1] != 1:
            raise ValueError("features[-1] must be 1")

        self.features = features
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.seed = seed
        self.param_dtype = param_dtype

        self.layers = []

        seeds = jax.random.randint(jax.random.key(seed), (len(features),), minval=-2**30,
                                   maxval=2**30).tolist()

        for seed, fan_in, fan_out in zip(seeds, self.features[:-1], self.features[1:]):
            self.layers.append( nnx.Linear(fan_in, fan_out, kernel_init=self.kernel_init,
                                           bias_init=self.bias_init, rngs=nnx.Rngs(seed)) )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for layer in self.layers[:-1]:
            x = self.activation( layer(x) )

        x = self.layers[-1](x)

        return x.flatten()

class FFNComplex(nnx.Module):

    def __init__(
            self,
            features: list | tuple,
            activation: Callable = nnx.gelu,
            kernel_init: Callable = nnx.initializers.lecun_normal(),
            bias_init: Callable = nnx.initializers.zeros_init(),
            seed: int = 0,
            param_dtype: jnp.dtype = jnp.float64
            ):

        if features[-1] != 2:
            raise ValueError("features[-1] must be 2")

        self.features = features
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.seed = seed
        self.param_dtype = param_dtype

        self.layers = []

        seeds = jax.random.randint(jax.random.key(seed), (len(features),), minval=-2**30,
                                   maxval=2**30).tolist()

        for seed, fan_in, fan_out in zip(seeds, self.features[:-1], self.features[1:]):
            self.layers.append( nnx.Linear(fan_in, fan_out, kernel_init=self.kernel_init,
                                           bias_init=self.bias_init, rngs=nnx.Rngs(seed)) )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for layer in self.layers[:-1]:
            x = self.activation( layer(x) )

        x = self.layers[-1](x)
        x = x[..., 0] + 1j * x[..., 1]

        return x.flatten()

class FFNModPhase(nnx.Module):

    def __init__(
            self,
            features: list | tuple,
            activation: Callable = nnx.gelu,
            kernel_init: Callable = nnx.initializers.lecun_normal(),
            bias_init: Callable = nnx.initializers.zeros_init(),
            seed: int = 0,
            param_dtype: jnp.dtype = jnp.float64
            ):

        if features[-1] != 1:
            raise ValueError("features[-1] must be 1")

        self.features = features
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.seed = seed
        self.param_dtype = param_dtype

        self.real = FFN(
                features=self.features,
                activation=self.activation,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                seed=self.seed,
                param_dtype=self.param_dtype
                )

        self.imag = FFN(
                features=self.features,
                activation=self.activation,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                seed=self.seed + 1,
                param_dtype=self.param_dtype
                )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.real(x) + 1j * self.imag(x)
