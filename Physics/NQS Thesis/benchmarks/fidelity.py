import netket as nk
import jax.numpy as jnp
import matplotlib.pyplot as plt

from operators.fidelity import fidelity_dot_product_squared
from plotting.style import default_fig_ax
from benchmarks.benchmarks_common import *

def benchmark_fidelity_ed(
        gen_ham_hi_g, params, model, k=2,
        n_samples=1024, chunk_size=None, n_discard_per_chain=None, sampler_seed=None,
        sampler="local", d_max=2, n_chains=1024, sweep_size=None,
        optimizer="sgd", lr=1e-2, n_epochs=1000,
        preconditioner="sr", diag_shift=1e-2, qgt=None, chunk_size_bwd=None,
        ):

    if isinstance(params, jnp.ndarray):
        params = params.tolist()

    fidelity_list = []
    energy_ed_list = []
    energy_nqs_list = []

    for param in params:
        ham, hi, g = gen_ham_hi_g(param=param)

        energy_ed, psi_ed = nk.exact.lanczos_ed(ham, k=k, compute_eigenvectors=True)
        energy_ed_list.append(energy_ed)
        psi_ed = jnp.array([psi_ed[:,i] for i in range(k)])

        vstate, driver = get_vstate_driver(
                ham, hi, g, model,
                n_samples, chunk_size, n_discard_per_chain, sampler_seed,
                sampler, d_max, n_chains, sweep_size,
                optimizer, lr,
                preconditioner, diag_shift, qgt, chunk_size_bwd,
                )
        driver.run(n_iter=n_epochs)

        energy_nqs = vstate.expect(ham).mean
        energy_nqs_list.append(energy_nqs)
        psi_nqs = vstate.to_array()

        fidelity = [fidelity_dot_product_squared(psi_ed[i], psi_nqs) for i in range(k)]

        fidelity_list.append(fidelity)

        print(f"Free Parameter: {param}")
        print(f"ED Energies: {energy_ed}")
        print(f"Fidelities: {fidelity}")
        print(f"NQS Energy: {energy_nqs}\n")

    fidelity_list = jnp.array(fidelity_list)
    energy_ed_list = jnp.array(energy_ed_list)
    energy_nqs_list = jnp.array(energy_nqs_list)

    fig1, ax1 = default_fig_ax()
    ax1.set_xlabel("Free Parameter")
    ax1.set_ylabel("Fidelity")
    for i in range(k):
        ax1.plot(params, fidelity_list[:,i], label=f"{i}'th Excited State")
    ax1.plot(params, fidelity_list.sum(axis=-1), label=f"Summed Fidelity")

    fig2, ax2 = default_fig_ax()
    ax2.set_xlabel("Free Parameter")
    ax2.set_ylabel("Energy")
    ax2.plot(params, energy_nqs_list, '.', label="NQS")
    for i in range(k):
        ax2.plot(params, energy_ed_list[:,i], label=f"ED ({i}'th Excited State)")

    ax1.legend()
    ax2.legend()
    plt.show()

    return fidelity_list, energy_ed_list, energy_nqs_list
