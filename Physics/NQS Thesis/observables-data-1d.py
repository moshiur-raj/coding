import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".98"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import netket as nk
import jax.numpy as jnp

from hamiltonians.j1j2 import j1j2_1d
from benchmarks.benchmarks_common import *
from operators.spin_squared import s2
from operators.susceptibility import m2_1d_neel
from models.vit import ViT

l = 20
model = ViT(input_shape=(1, l), patch_shape=(1, 2), embed_dim=20, n_heads=5, n_layers=2,
            trans_inv=True)
j2_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
energy_values = []
variance_values = []
s2_values = []
m2_neel_values = []
# %%

for j2 in j2_list:
    ham, hi, g = j1j2_1d(param=j2, l=l, sign_rule=[False, False], normalize=False)
    n = g.n_nodes
    m2_neel = m2_1d_neel(hi, g)
    s2_op = s2(hi, g)
    vstate, driver = get_vstate_driver(
            ham, hi, g, model,
            n_samples=1024*4, chunk_size=1024*4, n_discard_per_chain=None, sweep_size=None,
            sampler="exchange", d_max=2, n_chains=1024*4, sampler_seed=None,
            optimizer="minsr", lr=7.5e-3, diag_shift=1e-4
            )

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=1000, out=log)
    energy_values.append(log["Energy"]["Mean"][-1].real)
    variance_values.append(log["Energy"]["Mean"][-1].real)
    s2_values.append(vstate.expect(s2_op).mean.real / 4)
    m2_neel_values.append(vstate.expect(m2_neel).mean.real / 4 / (n*(n+2)))

    print(f"j2={j2}")
    print(f"energy={float(energy_values[-1].real)}")
    print(f"variance={float(variance_values[-1].real)}")
    print(f"s2={float(s2_values[-1].real)}")
    print(f"m2_neel={float(m2_neel_values[-1].real)}")

    jnp.savez(f"log_j2_{j2}", param=vstate.parameters, samples=vstate.samples,
              Energy=log["Energy"]["Mean"], Variance=log["Energy"]["Variance"],
              m2_neel=m2_neel_values[-1], s2=s2_values[-1])

jnp.savez("log", energy=jnp.array(energy_values), s2=jnp.array(s2_values),
          m2_neel=jnp.array(m2_neel_values))
