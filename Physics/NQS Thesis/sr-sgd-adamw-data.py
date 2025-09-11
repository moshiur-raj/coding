import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".98"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax.numpy as jnp
import netket as nk

from hamiltonians.j1j2 import j1j2_1d
from benchmarks.benchmarks_common import *
from models.vit import ViT

def fidelity(psi, gs):
    c = jnp.dot(psi, gs)
    return jnp.acos(jnp.sqrt(c * c.conj()))

optimizer = "adamw"

l = 20
model = ViT(input_shape=(1, l), patch_shape=(1, 2), embed_dim=20, n_heads=5, n_layers=2,
            trans_inv=True)
ham, hi, g = j1j2_1d(param=0.5, l=l, sign_rule=[False, False], normalize=False)

# data = jnp.load('./results/20/log-ed.npz')
# gs = data['eigenstates'][100][:, 0]

vstate, driver = get_vstate_driver(
        ham, hi, g, model,
        n_samples=1024*4, chunk_size=1024*4, n_discard_per_chain=None, sweep_size=None,
        sampler="exchange", d_max=2, n_chains=1024*4, sampler_seed=None,
        optimizer=optimizer, lr=7.5e-3, diag_shift=1e-4
        )
# %%

energy = [vstate.expect(ham).mean.real / 4 / l]
moving_energy = []
# gamma = [fidelity(vstate.to_array(), gs)]
psi = [vstate.to_array()]

for iter in range(20):
    log = nk.logging.RuntimeLog()
    driver.run(n_iter=50, out=log)
    energy.append(log["Energy"]["Mean"][-1].real / 4 / l)
    moving_energy.append(log["Energy"]["Mean"].real.mean() / 4 / l)
    psi.append(vstate.to_array())
    # gamma.append(fidelity(vstate.to_array(), gs))

    print(energy[-1])
    print(moving_energy[-1])
    # print(gamma[-1])

jnp.savez(f'log_{optimizer}', energy=energy, psi=psi, moving_energy=moving_energy)
