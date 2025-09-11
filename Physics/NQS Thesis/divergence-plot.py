import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Energy per Spin")

    ax_t = ax.twinx()
    ax_t.set_ylabel("Model Parameter")

    return fig, ax, ax_t

l = 6
energy_gs = -0.503810
# %%

data = jnp.load('./results/comparing-nqs/divergence-sr.npz')
param_max_sr = data['param']
energy_sr = data['energy']

fig, ax, ax_t = gen_fig_ax()
ax.set_ylim(-1, 1)
ax_t.set_ylim(0, 1)
ax.axhline(y=energy_gs, label="Ground State Energy", color='green')
energy = jnp.array(energy_sr) / 4 / (l*l)
iters = list(range(0, 5*len(energy), 5))
ax.plot(iters, energy, '.-', label="Variational Energy")
ax_t.plot(iters, param_max_sr, '.-', color='red', label="Largest Parameter")
ax.plot([], [], '.-', color='red', label='Largest Parameter')
ax.legend(loc='upper right', bbox_to_anchor=(.85, 1))
ax.set_xlim(0, ax.get_xlim()[1])

fig.savefig("divergence-sr.svg")
# plt.show()
plt.close()
# %%

data = jnp.load('./results/comparing-nqs/divergence-adamw.npz')
param_max_adamw = data['param']
energy_adamw = data['energy']

fig, ax, ax_t = gen_fig_ax()
ax.axhline(y=energy_gs, label="Ground State Energy", color='green')
energy = jnp.array(energy_adamw) / 4 / (l*l)
iters = list(range(0, 50*len(energy), 50))
ax.plot(iters, energy, '.-', label="Variational Energy")
ax_t.plot(iters, param_max_adamw, '.-', color='red', label="Largest Parameter")
ax.plot([], [], '.-', color='red', label='Largest Parameter')
ax.legend(loc='upper right', bbox_to_anchor=(1, .95))
ax.set_xlim(0, ax.get_xlim()[1])

fig.savefig("divergence-adamw.svg")
# plt.show()
plt.close()
# %%

data = jnp.load('./results/comparing-nqs/divergence-msr-sr.npz')
param_max_msr_sr = data['param']
energy_msr_sr = data['energy']

fig, ax, ax_t = gen_fig_ax()
ax.axhline(y=energy_gs, label="Ground State Energy", color='green')
energy = jnp.array(energy_msr_sr) / 4 / (l*l)
iters = list(range(0, 50*len(energy), 50))
ax.plot(iters, energy, '.-', label="Variational Energy")
ax_t.plot(iters, param_max_msr_sr, '.-', color='red', label="Largest Parameter")
ax.plot([], [], '.-', color='red', label='Largest Parameter')
ax.legend(loc='upper right')
ax.set_xlim(0, ax.get_xlim()[1])

fig.savefig("divergence-msr-sr.svg")
# plt.show()
plt.close()
# %%

data = jnp.load('./results/comparing-nqs/divergence-sr-2.npz')
param_max_sr_2 = data['param']
energy_sr_2 = data['energy']

fig, ax, ax_t = gen_fig_ax()
ax.set_ylim(-1, 1)
ax_t.set_ylim(0, 1)
ax.axhline(y=energy_gs, label="Ground State Energy", color='green')
energy = jnp.array(energy_sr_2) / 4 / (l*l)
param_max_sr_2[16] = 1e10
iters = list(range(0, 50*len(energy), 50))
ax.plot(iters, energy, '.-', label="Variational Energy")
ax_t.plot(iters, param_max_sr_2, '.-', color='red', label="Largest Parameter")
ax.plot([], [], '.-', color='red', label='Largest Parameter')
ax.legend(loc='upper right', bbox_to_anchor=(.85, 1))
ax.set_xlim(0, ax.get_xlim()[1])

fig.savefig("divergence-sr-2.svg")
# plt.show()
plt.close()
