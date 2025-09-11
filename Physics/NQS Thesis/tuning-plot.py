
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.numpy as jnp

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

dir = './results/tuning/log_j2_0.5'
# %%

energy = []
nl_list = [1, 2, 4]

for nl in nl_list:
    data = jnp.load(f'{dir}_nl_{nl}.npz')
    energy.append(data['Energy'][-50:].real.mean() / 4 / 100)

print(energy)
# %%

energy = []
samp_list = [1, 2, 4]

for samp in samp_list:
    data = jnp.load(f'{dir}_samp_{samp}.npz')
    energy.append(data['Energy'][-50:].real.mean() / 4 / 100)

print(energy)
# %%

data = jnp.load(f'{dir}_d_70.npz')
print(data['Energy'][-50:].real.mean() / 4 / 100)
# %%

data = jnp.load('./results/tuning/log.npz')
energy = []

for i in [1000, 2500, 5000, 7500, 10000]:
    energy.append(data['Energy'][i-50:i].real.mean().item() / 4/ 100)

print(energy)
