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

l = 6
dir = f"./results/6x6_0/"
# %%

j2_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
data = jnp.load(dir + "log.npz")
m2_neel = data['m2_neel']
m2_stripe = data['m2_stripe']
s2 = data['s2']

# df = pd.read_excel(dir + "neel.xlsx")
# m2_neel_exact = df.to_numpy()
# df = pd.read_excel(dir + "stripe.xlsx")
# m2_stripe_exact = df.to_numpy()
#
energy_exact = jnp.array([-0.678872, -0.638096, -0.599046, -0.562459, -0.529745, -0.503810, -0.495178,
                -0.493239, -0.506588, -0.530001, -0.586487, -0.649052, -0.714360])
m2_neel_exact = jnp.array([0.19879, 0.18893, 0.17601, 0.15800, 0.13109, 0.09236, 0.07062, 0.04378,
                             0.01954, 0.01232, 0.00611, 0.00314, 0.00183])
m2_stripe_exact = jnp.array([0.01322, 0.01326, 0.01335, 0.01354, 0.01404, 0.01594, 0.01965, 0.03822,
                             0.08167, 0.10006, 0.11370, 0.11925, 0.12183])

var_list = []
energy = []
for j2 in j2_list:
    data = jnp.load(dir + f"log_j2_{j2}.npz", allow_pickle=True)
    var_list.append(data['Variance'][-50:].real.mean() / (4*4))
    energy.append(data['Energy'][-50:].real.mean())

for i in [-3, -4]:
    data = jnp.load('./results/6x6_1/log.npz')
    m2_stripe[i] = data['m2_stripe'][i]
    m2_neel[i] = data['m2_neel'][i]
    s2[i] = data['s2'][i]

    j2 = j2_list[i]
    data = jnp.load(dir + f"log_j2_{j2}.npz", allow_pickle=True)
    var_list[i]= data['Variance'][-50:].real.mean() / (4*4)
    energy[i] = data['Energy'][-50:].real.mean()

for i in [-1, -7]:
    data = jnp.load(f'./results/6x6_0/log_j2_{j2_list[i]}_2.npz')
    m2_stripe[i] = data['m2_stripe']
    m2_neel[i] = data['m2_neel']
    s2[i] = data['s2']
    var_list[i]= data['Variance'][-50:].real.mean() / (4*4)
    energy[i] = data['Energy'][-50:].real.mean()

energy = jnp.array(energy)
# %%

fig, ax = gen_fig_ax()
ax.set_ylabel(r"$M^2_N(\mathbf{q})$")
# ax.plot(m2_neel_exact[:, 0][:207], m2_neel_exact[:, 1][:207], color='#1f77b4')
ax.plot(j2_list, m2_neel_exact, color='#1f77b4')
ax.plot(j2_list, m2_neel, '.', label=r"$\mathbf{q} = (\pi, \pi)$", color='#1f77b4')
# ax.plot(m2_stripe_exact[:, 0][109:], m2_stripe_exact[:, 1][109:], color='#ff7f0e')
ax.plot(j2_list, m2_stripe_exact, color='#ff7f0e')
ax.plot(j2_list, m2_stripe, '.', label=r"$\mathbf{q} = (\pi, 0)$", color='#ff7f0e')

ax.legend(loc='upper right', bbox_to_anchor=(1, .85))
fig.savefig('magnetic_susceptibility_6x6_exact.svg')
plt.close()
# %%

# j2_exact = jnp.linspace(0, 1, 1000)
# curve_fit = nn_fit(j2, m2_neel, FFN((1, 128, 128, 1)), 4e-3, 2000)
# m2_neel_exact = curve_fit(j2_exact.reshape(-1, 1))
# curve_fit = nn_fit(j2, m2_stripe, FFN((1, 128, 128, 1)), 1e-3, 10000)
# m2_stripe_exact = curve_fit(j2_exact.reshape(-1, 1))
#
fig, ax = gen_fig_ax()
ax.set_ylabel(r"$M^2_N(\mathbf{q})$")
# ax.plot(j2_exact, m2_stripe_exact)
# ax.plot(j2_exact, m2_neel_exact)
ax.plot(j2_list, m2_neel, '.-', label=r"$\mathbf{q} = (\pi, \pi)$")
ax.plot(j2_list, m2_stripe, '.-', label=r"$\mathbf{q} = (\pi, 0)$")

ax.legend()
fig.savefig('magnetic_susceptibility_6x6.svg')
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
fig.savefig('s2_var_6x6.svg')
plt.close()
# %%

fig, ax = gen_fig_ax()
ax.set_ylabel("Energy Per Spin")
ax.plot(j2_list, energy_exact, '-', label='ED' )
ax.plot(j2_list, energy / (l*l) / 4, '.', label='NQS' )

ax.legend()
fig.savefig('energy_6x6.svg')
plt.close()
# %%

df = {'J_2': j2_list, 'Energy (Exact)': energy_exact, 'Energy (NQS)': energy / (l*l) / 4,
      'Error (%)': (energy_exact - energy/(l*l)/4) / energy_exact * 100}
df = pd.DataFrame(df)
