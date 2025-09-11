from functools import partial
from typing import Callable
from flax import nnx
from flax.typing import Dtype
import jax
import jax.numpy as jnp
from math import prod

default_float_type = jnp.float64
default_kernel_init = nnx.initializers.lecun_normal(dtype=default_float_type)
default_bias_init = nnx.initializers.zeros_init()
default_ffn_activation = nnx.gelu

default_patch_shape = (2, 2)
default_ffn_multiplier = 4
default_hidden_features_multiplier = 2

default_seed = 0
default_key = jax.random.key(0)

def log_cosh(x):
    """
    Logarithm of the hyperbolic cosine, implemented in a more stable way.
    """
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

def extract_patches_2d(
        x: jnp.ndarray,
        patch_shape: tuple[int, int],
        n_patches: tuple[int, int],
        ) -> jnp.ndarray:

    x = x.reshape(*x.shape[:-1], n_patches[0], patch_shape[0], n_patches[1], patch_shape[1])
    x = x.swapaxes(-2, -3)
    x = x.reshape(*x.shape[:-4], prod(n_patches), prod(patch_shape))

    return x

class Embed(nnx.Module):
    def __init__(
            self,
            patch_shape: tuple[int, int],
            n_patches: tuple[int, int],
            embed_dim: int,
            kernel_init: nnx.Initializer = default_kernel_init,
            bias_init: nnx.Initializer = default_bias_init,
            key: jax.Array = default_key,
            param_dtype: Dtype = default_float_type,
            ):

        self.embed_dim = embed_dim
        self.patch_shape = patch_shape
        self.n_patches = n_patches
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.key = key
        self.param_dtype = param_dtype

        self.extract_patches = extract_patches_2d

        self.embed = nnx.Linear(
                in_features=prod(self.patch_shape),
                out_features=self.embed_dim,
                rngs=nnx.Rngs(self.key),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                param_dtype=self.param_dtype,
                )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.extract_patches(x, self.patch_shape, self.n_patches)
        x = self.embed(x)

        return x

@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def translations_2d(lattice, i, j):
    return jnp.roll(jnp.roll(lattice, i, axis=-2), j, axis=-1).reshape(lattice.shape[0], -1)

class Attention(nnx.Module):
    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            n_patches: tuple[int, int],
            trans_inv: bool,
            kernel_init: nnx.Initializer = default_kernel_init,
            bias_init: nnx.Initializer = default_bias_init,
            key: jax.Array = default_key,
            param_dtype: Dtype = default_float_type,
            ):

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.trans_inv = trans_inv
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.key = key
        self.param_dtype = param_dtype

        gen_key = nnx.Rngs(self.key)

        self.v = nnx.Linear(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                rngs=nnx.Rngs(gen_key()),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                param_dtype=self.param_dtype,
                )

        if self.trans_inv:
            self.alpha = nnx.Param(
                    self.kernel_init(
                        key=gen_key(),
                        shape=(self.n_heads, *self.n_patches),
                        dtype=self.param_dtype,
                        )
                    )
        else:
            self.alpha = nnx.Param(
                    self.kernel_init(
                        key=gen_key(),
                        shape=(self.n_heads, prod(self.n_patches), prod(self.n_patches)),
                        dtype=self.param_dtype,
                        )
                    )

        self.w = nnx.Linear(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                rngs=nnx.Rngs(gen_key()),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                param_dtype=self.param_dtype,
                )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.v(x)
        # batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff
        x = x.reshape(*x.shape[:-1], self.n_heads, -1)
        # batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff
        x = x.swapaxes(-2, -3)

        if self.trans_inv:
            alpha = translations_2d(self.alpha, jnp.arange(self.n_patches[0]),
                                        jnp.arange(self.n_patches[1]))
            alpha = alpha.reshape(self.n_heads, prod(self.n_patches), prod(self.n_patches))
        else:
            alpha = self.alpha

        x = alpha @ x
        # batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff
        x = x.swapaxes(-2, -3)
        # batch n_patches n_heads d_eff ->  batch n_patches (n_heads d_eff)
        x = x.reshape(*x.shape[:-2], -1)
        x = self.w(x)

        return x

class EncoderBlock(nnx.Module):
    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            n_patches: tuple[int, int],
            trans_inv: bool,
            ffn_multiplier: int = default_ffn_multiplier,
            ffn_activation: Callable = default_ffn_activation,
            kernel_init: nnx.Initializer = default_kernel_init,
            bias_init: nnx.Initializer = default_bias_init,
            key: jax.Array = default_key,
            param_dtype: Dtype = default_float_type,
            ):

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.trans_inv = trans_inv
        self.ffn_multiplier = ffn_multiplier
        self.ffn_activation = ffn_activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.key = key
        self.param_dtype = param_dtype

        gen_key = nnx.Rngs(self.key)

        self.attention = Attention(
                embed_dim=self.embed_dim,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                trans_inv=self.trans_inv,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                key=gen_key(),
                param_dtype=self.param_dtype,
                )

        self.norm_0 = nnx.LayerNorm(
                num_features=self.embed_dim,
                rngs=nnx.Rngs(gen_key()),
                param_dtype=self.param_dtype,
                )

        self.norm_1 = nnx.LayerNorm(
                num_features=self.embed_dim,
                rngs=nnx.Rngs(gen_key()),
                param_dtype=self.param_dtype,
                )

        self.non_linear = nnx.Sequential(
                nnx.Linear(
                    in_features=self.embed_dim,
                    out_features= self.ffn_multiplier*self.embed_dim,
                    rngs=nnx.Rngs(gen_key()),
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    param_dtype=self.param_dtype,
                    ),
                self.ffn_activation,
                nnx.Linear(
                    in_features=self.ffn_multiplier*self.embed_dim,
                    out_features=self.embed_dim,
                    rngs=nnx.Rngs(gen_key()),
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    param_dtype=self.param_dtype,
                    ),
                )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x += self.attention(self.norm_0(x))
        x += self.non_linear(self.norm_1(x))

        return x

class Encoder(nnx.Module):
    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            n_patches: tuple[int, int],
            n_layers: int,
            trans_inv: bool,
            ffn_multiplier: int = default_ffn_multiplier,
            ffn_activation: Callable = default_ffn_activation,
            kernel_init: nnx.Initializer = default_kernel_init,
            bias_init: nnx.Initializer = default_bias_init,
            key: jax.Array = default_key,
            param_dtype: Dtype = default_float_type,
            ):

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.ffn_multiplier = ffn_multiplier
        self.n_layers = n_layers
        self.trans_inv = trans_inv
        self.ffn_activation = ffn_activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.key = key
        self.param_dtype = param_dtype

        gen_key = nnx.Rngs(self.key)

        self.layers = [
                EncoderBlock(
                     embed_dim=self.embed_dim,
                     n_heads=self.n_heads,
                     n_patches=self.n_patches,
                     trans_inv=self.trans_inv,
                     ffn_multiplier=self.ffn_multiplier,
                     ffn_activation=self.ffn_activation,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init,
                     key=gen_key(),
                     param_dtype=self.param_dtype,
                    )
                for _ in range(self.n_layers)
                ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)

        return x

class OutputHead(nnx.Module):
    def __init__(
            self,
            embed_dim: int,
            hidden_features_multiplier: int = default_hidden_features_multiplier,
            kernel_init: nnx.Initializer = default_kernel_init,
            bias_init: nnx.Initializer = default_bias_init,
            key: jax.Array = default_key,
            param_dtype: Dtype = default_float_type,
            ):

        self.embed_dim = embed_dim
        self.hidden_features_multiplier = hidden_features_multiplier
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.key = key
        self.param_dtype = param_dtype

        gen_key = nnx.Rngs(self.key)

        self.norm_enc_out = nnx.LayerNorm(
                num_features=self.embed_dim,
                rngs=nnx.Rngs(gen_key()),
                param_dtype=self.param_dtype,
                )

        self.linear = nnx.Linear(
                    in_features=self.embed_dim,
                    out_features= self.hidden_features_multiplier*self.embed_dim,
                    rngs=nnx.Rngs(gen_key()),
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    param_dtype=self.param_dtype,
                )

        _num_features = self.hidden_features_multiplier * self.embed_dim // 2
        self.norm_real = nnx.LayerNorm(
               num_features=_num_features,
               rngs=nnx.Rngs(gen_key()),
               param_dtype=self.param_dtype,
               )
        
        self.norm_imag = nnx.LayerNorm(
               num_features=_num_features,
               rngs=nnx.Rngs(gen_key()),
               param_dtype=self.param_dtype,
               )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.sum(axis=-2)
        x = self.norm_enc_out(x)
        x = self.linear(x)

        x = self.norm_real(x[..., :self.norm_real.num_features]) \
                + 1j * self.norm_imag(x[..., -self.norm_imag.num_features:])
        x = log_cosh(x)
        x = x.sum(axis=-1)

        return x

class ViT(nnx.Module):
    def __init__(
            self,
            input_shape: tuple[int, int],
            embed_dim: int,
            n_heads: int,
            n_layers: int,
            patch_shape: tuple[int, int] = default_patch_shape,
            trans_inv: bool = True,
            ffn_multiplier: int = default_ffn_multiplier,
            hidden_features_multiplier: int = default_hidden_features_multiplier,
            ffn_activation: Callable = default_ffn_activation,
            kernel_init: nnx.Initializer = default_kernel_init,
            bias_init: nnx.Initializer = default_bias_init,
            seed: int = default_seed,
            param_dtype: Dtype = default_float_type,
            ):

        if input_shape[0] % patch_shape[0] or input_shape[1] % patch_shape[1]:
            raise ValueError("Enter valid patch_shape.")

        self.n_patches = (input_shape[0] // patch_shape[0], input_shape[1] // patch_shape[1])

        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.patch_shape = patch_shape
        self.ffn_multiplier = ffn_multiplier
        self.hidden_features_multiplier = hidden_features_multiplier
        self.ffn_activation = ffn_activation
        self.trans_inv = trans_inv
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.seed = seed
        self.param_dtype = param_dtype

        gen_key = nnx.Rngs(self.seed)

        self.embed = Embed(
                embed_dim=self.embed_dim,
                patch_shape=self.patch_shape,
                n_patches=self.n_patches,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                key=gen_key(),
                param_dtype=self.param_dtype,
                )

        self.encoder = Encoder(
                embed_dim=self.embed_dim,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                n_patches=self.n_patches,
                trans_inv=self.trans_inv,
                ffn_multiplier=self.ffn_multiplier,
                ffn_activation=self.ffn_activation,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                key=gen_key(),
                param_dtype=self.param_dtype,
                )

        self.output_head = OutputHead(
                embed_dim=self.embed_dim,
                hidden_features_multiplier=self.hidden_features_multiplier,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                key=gen_key(),
                param_dtype=self.param_dtype,
                )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.embed(x)
        x = self.encoder(x)
        x = self.output_head(x)

        return x
