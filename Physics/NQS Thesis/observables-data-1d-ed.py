import jax.numpy as jnp
import netket as nk

from hamiltonians.j1j2 import j1j2_1d
from operators.spin_squared import s2
from operators.susceptibility import m2_1d_neel

l = 20
j2_list = jnp.linspace(0, 1, 201)
energy_values = []
eigenstates = []
s2_values = []
m2_neel_values = []
# %%

for j2 in j2_list:
    ham, hi, g = j1j2_1d(param=float(j2), l=l, sign_rule=[False, False], normalize=False)
    n = g.n_nodes
    m2_neel = m2_1d_neel(hi, g)
    s2_op = s2(hi, g)

    energy, eigenstate = nk.exact.lanczos_ed(ham, compute_eigenvectors=True, k=10)
    gs = eigenstate[:, 0].reshape(-1, 1)

    energy_values.append(energy)
    eigenstates.append(eigenstate)
    s2_values.append((gs.conj().T @ (s2_op @ gs) / 4).item())
    m2_neel_values.append((gs.conj().T @ (m2_neel @ gs) / 4 / (n*(n+2))).item())

    print(f"j2={j2}")
    print(f"energy={float(energy_values[-1][0].real)}")
    print(f"s2={float(s2_values[-1].real)}")
    print(f"m2_neel={float(m2_neel_values[-1].real)}")
# %%

jnp.savez("log", energy=jnp.array(energy_values), eigenstates=jnp.array(eigenstates),
          s2=jnp.array(s2_values), m2_neel=jnp.array(m2_neel_values))
