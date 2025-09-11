import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from flax import nnx
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator

from plotting.style import default_fig_ax
from models.ffn import FFN
from curve_fit import nn_fit

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

l = 10
dir = f"./results/10x10/"
# %%

j2_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]

var_list = []
energy = []
m2_neel = []
m2_stripe = []
s2 = [0.17367636711326603, 0.1994806169178837]
for j2 in j2_list:
    data = jnp.load(dir + f"log_j2_{j2}.npz", allow_pickle=True)
    var_list.append(data['Variance'][-50:].real.mean() / (4*4))
    energy.append(data['Energy'][-50:].real.mean())
    m2_neel.append(float(data['m2_neel']))
    m2_stripe.append(float(data['m2_stripe']))

    if j2 != 0 and j2 != 0.1:
        s2.append(float(data['s2']))

energy = jnp.array(energy)
# %%

fig, ax = gen_fig_ax()
ax.set_ylabel(r"$M^2_N(\mathbf{q})$")
ax.plot(j2_list, m2_neel, '.-', label=r"$\mathbf{q} = (\pi, \pi)$")
ax.plot(j2_list, m2_stripe, '.-', label=r"$\mathbf{q} = (\pi, 0)$")

ax.legend()
fig.savefig('magnetic_susceptibility_10x10.svg')
plt.close()
# %%

fig, ax = gen_fig_ax()

ax.plot(j2_list, s2, '.-', label=r'$\langle \hat{S}^2 \rangle$')
ax.set_ylabel(r'Squared Magnetization, $\langle \hat{S}^2 \rangle$')

ax_t = ax.twinx()
ax_t.plot([], [])
ax_t.plot(j2_list, var_list, '.-', label=r'$\sigma^2_E$')
ax.plot([], [], '.-', label=r'$\sigma^2_E$')
ax_t.set_ylabel(r'Variance, $\sigma^2_E$')

ax.legend()
fig.savefig('s2_var_10x10.svg')
plt.close()
# %%

fig, ax = gen_fig_ax()
ax.set_ylabel("Energy Per Spin")
ax.plot(j2_list, energy / (l*l) / 4, '.-')

fig.savefig('energy_10x10.svg')
plt.close()
# %%

df = {'J_2': j2_list, 'Energy (NQS)': energy / (l*l) / 4}
df = pd.DataFrame(df)
