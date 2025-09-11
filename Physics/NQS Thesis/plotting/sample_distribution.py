import netket as nk
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_sample_dist(vstate: nk.vqs.MCState, hilbert_size: int, bins: int = 50):

    binary_coded_samples = (vstate.samples + 1) // 2
    x = binary_coded_samples * jnp.array([2**n for n in range(hilbert_size)])
    x = x.sum(axis=-1)
    x = x.flatten()


    plt.hist(x, bins=bins, edgecolor='black')
    plt.xlabel("Hilbert Space Index")
    plt.ylabel("Frequency of Appearance in Variational Sample")
    plt.grid(True)
    plt.show()
