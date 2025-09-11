import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".98"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax.numpy as jnp

from hamiltonians.j1j2 import j1j2_2d_square
from benchmarks.benchmarks_common import *
from models.rbm import RBM

def max_abs_in_dict(d):
    max_vals = []

    for v in d.values():
        if isinstance(v, jnp.ndarray):
            max_vals.append(jnp.max(jnp.abs(v)))

    return jnp.max(jnp.array(max_vals))

l = 6
model = RBM(l*l, alpha=8)
ham, hi, g = j1j2_2d_square(param=0.5, l=l, sign_rule=[False, False], normalize=False)

# %%

vstate, driver = get_vstate_driver(
        ham, hi, g, model,
        n_samples=1024*4, chunk_size=1024*4, n_discard_per_chain=None, sweep_size=None,
        sampler="exchange", d_max=2, n_chains=1024*4, sampler_seed=None,
        optimizer="minsr", lr=4e-3, diag_shift=1e-3
        )
param = [max_abs_in_dict(vstate.parameters)]
energy = [vstate.expect(ham).mean.real]

for iter in range(10):
    driver.run(n_iter=5)
    param.append(max_abs_in_dict(vstate.parameters))
    energy.append(vstate.expect(ham).mean.real)

jnp.savez("log", param=jnp.array(param), energy=jnp.array(energy))
