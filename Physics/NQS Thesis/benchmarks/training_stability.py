import netket as nk
import jax.numpy as jnp

from benchmarks.benchmarks_common import *

def view_training_stability(
        gen_ham_hi_g, param, model, obs=None,
        n_samples=1024, chunk_size=None, n_discard_per_chain=None, sampler_seed=None,
        sampler="local", d_max=2, n_chains=1024, sweep_size=None,
        optimizer="sgd", lr=1e-2, n_epochs=1000, diag_shift=1e-2,
        save=False,
        ) -> tuple:

    ham, hi, g = gen_ham_hi_g(param=param)

    vstate, driver = get_vstate_driver(
            ham, hi, g, model,
            n_samples, chunk_size, n_discard_per_chain, sampler_seed,
            sampler, d_max, n_chains, sweep_size,
            optimizer, lr, diag_shift
            )

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_epochs, obs=obs, out=log)

    if save:
        jnp.savez('log', Energy=log['Energy']['Mean'], Variance=log['Energy']['Variance'],
                  Samples=vstate.samples, params=vstate.parameters, allow_pickle=True)

    return log, vstate
