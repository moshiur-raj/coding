import matplotlib as mpl
import matplotlib.pyplot as plt
import netket as nk
import jax.numpy as jnp
import pandas as pd

from models.vit import ViT
from hamiltonians.j1j2 import j1j2_1d
from plotting.style import default_fig_ax

mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    'font.size': 12,        # Default font size
    'axes.titlesize': 12,   # Title font size
    'axes.labelsize': 12,   # Axis label font size
    'xtick.labelsize': 12,  # X-axis tick label font size
    'ytick.labelsize': 12,  # Y-axis tick label font size
    'legend.fontsize': 12   # Legend font size
})

def gen_fig_ax(frameon=False):
    fig, ax = default_fig_ax(frameon=frameon)
    ax.set_xlabel(r"$J_2$")
    return fig, ax


l = 20
dir = "./results/20/"
model = ViT(input_shape=(1, l), patch_shape=(1, 2), embed_dim=20, n_heads=5, n_layers=2,
            trans_inv=True)
# %%

data = jnp.load(dir + "log.npz")
j2_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
# energy = data['energy']
m2_neel = data['m2_neel']
s2 = data['s2']

data = jnp.load(dir + "log-ed.npz")
j2_ed = jnp.linspace(0, 1, 201)
indices = [0, 20, 40, 60, 80, 100, 110, 120, 130, 140, 160, 180, 200]
energy_ed = data['energy']
eigenstates = data['eigenstates']
m2_neel_ed = data['m2_neel']
s2_ed = data['s2']

e_diff_rel_1 = (energy_ed[:, 0] - energy_ed[:, 1])/energy_ed[:, 0]
# %%

energy = []
var_list = []
overlap_list_0 = []
overlap_list_1 = []
overlap_list_2 = []
for j2, i in zip(j2_list, indices):
    data = jnp.load(dir + f"log_j2_{j2}.npz", allow_pickle=True)
    param = data['param']
    energy.append(data['Energy'][-50:].real.mean())
    var_list.append(data['Variance'][-50:].real.mean() / (4*4))

    ham, hi, g = j1j2_1d(l=l, param=j2)

    sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=2)
    vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1024*4)
    vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1024*4,
                            variables={"params": param.item(), "model_state":
                                       vstate.model_state['model_state']})

    gs = eigenstates[i][:, 0]
    es1 = eigenstates[i][:, 1]
    es2 = eigenstates[i][:, 2]
    psi_nqs = vstate.to_array()

    overlap_list_0.append(jnp.abs(jnp.dot(gs, psi_nqs))**2)
    overlap_list_1.append(jnp.abs(jnp.dot(es1, psi_nqs))**2)
    overlap_list_2.append(jnp.abs(jnp.dot(es2, psi_nqs))**2)

overlap_total = jnp.array(overlap_list_0) + jnp.array(overlap_list_1) + jnp.array(overlap_list_2)
energy = jnp.array(energy)
# %%

fig, ax = gen_fig_ax()

ax.plot(j2_ed, m2_neel_ed, label='ED')
ax.plot(j2_list, m2_neel, '.', label='NQS')
ax.set_ylabel(r'$M^2_N (\pi)$')

ax.legend()
fig.savefig('m2_neel_1d.svg')
plt.close()
# %%

fig, ax = gen_fig_ax()

ax.plot(j2_ed, s2_ed, label=r'$\langle \hat{S}^2 \rangle_{ED}$')
ax.plot(j2_list, s2, '.-', label=r'$\langle \hat{S}^2 \rangle_{NQS}$')
ax.set_ylabel(r'Squared Magnetization, $\langle \hat{S}^2 \rangle$')

ax_t = ax.twinx()
ax_t.plot([], [])
ax_t.plot([], [])
ax_t.plot(j2_list, var_list, '.-', label=r'$\sigma^2_E$')
ax.plot([], [], '.-', label=r'$\sigma^2_E$')
ax_t.set_ylabel(r'Variance, $\sigma^2_E$')

ax.legend()
fig.savefig('s2_var_1d.svg')
plt.close()
# %%

fig, ax = gen_fig_ax()

ax.plot(j2_ed, energy_ed[:, 0] / 4 / l, label='ED')
ax.plot(j2_list, energy / 4 / l, '.', label='NQS')
ax.set_ylabel('Energy per Spin')

ax.legend()
fig.savefig('energy_1d.svg')
plt.close()
# %%

fig, ax = gen_fig_ax()
ax.set_ylabel(r'Overlap, $I$')
ax_t = ax.twinx()
ax_t.set_ylabel(r'Relative Energy Difference, $\epsilon$')

ax.plot(j2_list, overlap_total, '.-', label=r'$I_0 + I_1$')
ax.plot(j2_list, overlap_list_0, '.-', label=r'$I_0$')
ax.plot(j2_list, overlap_list_1, '.-', label=r'$I_1$')

ax_t.plot([], [])
ax_t.plot([], [])
ax_t.plot([], [])
ax_t.plot(j2_ed, e_diff_rel_1, '-', label=r'$\epsilon_1$')
ax.plot([], [], '-', label=r'$\epsilon_1$')


ax.legend(loc='lower left', bbox_to_anchor=(0, .05))
fig.savefig('overlap-1d.svg')
plt.close()
# %%

e = energy_ed[jnp.array(indices)]
df = {'J_2': j2_list, 'Energy (NQS)': energy / (l) / 4, 'Energy (Exact)': e[:, 0] / (l)/ 4,
      'Error (%)': jnp.round((e[:,0] - energy) / e[:, 0] * 100, 3)}
df = pd.DataFrame(df)
