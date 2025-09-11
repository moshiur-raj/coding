import netket as nk
import jax.numpy as jnp
import pandas as pd

from benchmarks.benchmarks_common import *

def compare_energy_with_ref(
        gen_ham_hi_g, params, model, energy_ref_list, energy_divisor=1,
        n_samples=1024, chunk_size=None, n_discard_per_chain=None, sampler_seed=None,
        sampler="local", d_max=2, n_chains=1024, sweep_size=None,
        optimizer="sgd", lr=1e-2, n_epochs=1000,
        preconditioner="sr", diag_shift=1e-2, qgt=None, chunk_size_bwd=None,
        ):

    if isinstance(params, jnp.ndarray):
        params = params.tolist()

    if not isinstance(energy_ref_list, jnp.ndarray):
        energy_ref_list = jnp.array(energy_ref_list)

    energy_nqs_list = []
    variance_list = []

    for param in params:
        ham, hi, g = gen_ham_hi_g(param=param)

        vstate, driver = get_vstate_driver(
                ham, hi, g, model,
                n_samples, chunk_size, n_discard_per_chain, sampler_seed,
                sampler, d_max, n_chains, sweep_size,
                optimizer, lr,
                preconditioner, diag_shift, qgt, chunk_size_bwd,
                )
        driver.run(n_iter=n_epochs)

        energy_nqs = vstate.expect(ham)
        energy_nqs_list.append(energy_nqs.mean.real)
        variance_list.append(energy_nqs.variance)
        

        print(f"Free Parameter: {param}")
        print(f"NQS Energy: {energy_nqs.mean.real}")
        print(f"Variance: {energy_nqs.variance}")

    energy_nqs_list = jnp.array(energy_nqs_list) / energy_divisor
    variance_list = jnp.array(variance_list)
    error_list = jnp.abs( (energy_ref_list - energy_nqs_list) / energy_ref_list ) * 100

    data = {
            "Free Parameter": params,
            "Reference Energy": energy_ref_list,
            "NQS Energy": energy_nqs_list,
            "Error (%)": error_list,
            "Variance": variance_list
            }

    df = pd.DataFrame(data)

    return df
