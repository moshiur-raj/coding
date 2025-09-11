import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".98"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotting.style import default_fig_ax
from benchmarks.benchmarks_common import *
from models.vit import ViT

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
    return fig, ax

def fidelity(psi, gs):
    c = jnp.dot(psi, gs)
    return jnp.acos(jnp.sqrt(c * c.conj()))

optimizer_list = ["adamw", "sgd", "minsr"]
label_list = ["AdamW", "SGD", "MinSR"]

l = 20
model = ViT(input_shape=(1, l), patch_shape=(1, 2), embed_dim=20, n_heads=5, n_layers=2,
            trans_inv=True)

data = jnp.load('./results/20/log-ed.npz')
gs = data['eigenstates'][30][:, 0]
# %%

iters = list(range(0, 1000 + 50, 50))
fig1, ax1 = gen_fig_ax()
ax1.set_ylabel("Energy per Spin")
fig2, ax2 = gen_fig_ax()
ax2.set_ylabel(r"Fubini-Study Distance, $\gamma$")

for optimizer, label in zip(optimizer_list, label_list):

    data = jnp.load(f'./results/tuning/log_{optimizer}.npz')
    energy = data["energy"]
    moving_energy = data["moving_energy"]
    psi = data["psi"]
    gamma = [fidelity(vector, gs) for vector in psi]

    ax1.plot(iters, energy, '.-', label=label)
    ax2.plot(iters, gamma, '.-', label=label)

ax1.legend()
ax2.legend()

fig2.savefig('fubini-distance.svg')
