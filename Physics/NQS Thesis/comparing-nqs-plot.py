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

l = 6
energy_gs = -0.503810
dir = './results/comparing-nqs'

def gen_fig_ax(frameon=False):
    fig, ax = default_fig_ax(frameon=frameon)
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Energy per Spin")

    return fig, ax

def plot_energy_vs_iteration(file: str, label: str, ax, n_avg = None):
    data = jnp.load(file)
    energy = data["Energy"] / 4 / (l*l)
    iters = list(range(len(energy)))
    if n_avg != None:
        energy = [energy[i: i + n_avg].mean() for i in range(len(energy) - n_avg)]
        iters = list(range(n_avg, len(energy) + n_avg))
        ax.set_ylabel("Energy per Spin (Moving Average)")
    ax.plot(iters, energy, label=label)

def plot_variance_vs_iteration(file: str, label: str, ax, n_avg = None):
    data = jnp.load(file)
    variance = data["Variance"] / (4*4)
    iters = list(range(len(variance)))
    ax.set_ylabel("Variance")
    if n_avg != None:
        variance = [variance[i: i + n_avg].mean() for i in range(len(variance) - n_avg)]
        iters = list(range(n_avg, len(variance) + n_avg))
        ax.set_ylabel("Variance (Moving Average)")
    ax.plot(iters, variance, label=label)
# %%

fig, ax = gen_fig_ax()
ax.set_ylim(-.6, .8)
ax.axhline(y=energy_gs, label="Ground State", color='green')

plot_energy_vs_iteration('./results/comparing-nqs/rbm.npz', 'RBM', ax)
plot_energy_vs_iteration('./results/comparing-nqs/ffn.npz', 'FFN', ax)
ax.plot([], [])
data = jnp.load('./results/comparing-nqs/ffn_complex.npz')
energy = data["Energy"] / 4 / (l*l)
energy[146] = 1e9
iters = list(range(len(energy)))
ax.plot(iters, energy, label='FFNComplex')

ax.legend()
ax.set_xlim(0, ax.get_xlim()[1])
fig.savefig("rbm-ffn-complex.svg")
# plt.show()
plt.close()
# %%

fig, ax = gen_fig_ax()
ax.axhline(y=energy_gs, label="Ground State", color='green')

plot_energy_vs_iteration('./results/comparing-nqs/rbmmodphase.npz', 'RBMModPhase', ax)
plot_energy_vs_iteration('./results/comparing-nqs/ffnmodphase.npz', 'FFNModPhase', ax)
ax.plot([], [])
plot_energy_vs_iteration('./results/comparing-nqs/vit.npz', 'ViT', ax)

ax.legend()
ax.set_xlim(0, ax.get_xlim()[1])
fig.savefig("rbm-ffn-modphase-vit.svg")
# plt.show()
plt.close()
# %%

fig, ax = gen_fig_ax()

plot_energy_vs_iteration('./results/comparing-nqs/rbm_msr.npz', 'RBM (MSR)', ax)
plot_energy_vs_iteration('./results/comparing-nqs/ffn_msr.npz', 'FFN (MSR)', ax)
ax.plot([], [])
plot_energy_vs_iteration('./results/comparing-nqs/ffn_complex_msr.npz', 'FFN (MSR)', ax)

ax.legend()
ax.set_ylim(-.496, -.49)
ax.set_xlim(200, 1000)
fig.savefig("rbm-ffn-complex-msr.svg")
# plt.show()
plt.close()
# %%

fig, ax = gen_fig_ax()

plot_energy_vs_iteration('./results/comparing-nqs/rbm_msr.npz', 'RBM (MSR)', ax, n_avg=50)
plot_energy_vs_iteration('./results/comparing-nqs/ffn_msr.npz', 'FFN (MSR)', ax, n_avg=50)
ax.plot([], [])
plot_energy_vs_iteration('./results/comparing-nqs/ffn_complex_msr.npz', 'FFN (MSR)', ax, n_avg=50)

ax.legend()
ax.set_ylim(-.496, -.49)
ax.set_xlim(200, 1000)
fig.savefig("rbm-ffn-complex-msr-avg.svg")
# plt.show()
plt.close()
# %%

fig, ax = gen_fig_ax()
ax.axhline(y=energy_gs, label="Ground State", color='green')

plot_energy_vs_iteration('./results/comparing-nqs/rbmmodphase_msr.npz', 'RBMModPhase (MSR)', ax, n_avg=50)
plot_energy_vs_iteration('./results/comparing-nqs/ffnmodphase_msr.npz', 'FFNModPhase (MSR)', ax, n_avg=50)
ax.plot([], [])
plot_energy_vs_iteration('./results/comparing-nqs/vit_msr.npz', 'ViT (MSR)', ax, n_avg=50)

ax.legend()
ax.set_ylim(-.505, -.48)
ax.set_xlim(200, 1000)
fig.savefig("rbm-ffn-modphase-vit-msr-avg.svg")
# plt.show()
plt.close()
# %%
fig, ax = gen_fig_ax()
ax.axhline(y=energy_gs, label="Ground State", color='green')

plot_energy_vs_iteration('./results/comparing-nqs/vit.npz', 'ViT', ax, n_avg=50)
plot_energy_vs_iteration('./results/comparing-nqs/vit_msr.npz', 'ViT (MSR)', ax, n_avg=50)
ax.plot([], [])
plot_energy_vs_iteration('./results/comparing-nqs/vit_sym.npz', 'ViTSym', ax, n_avg=50)
plot_energy_vs_iteration('./results/comparing-nqs/vit_sym_msr.npz', 'ViTSym (MSR)', ax, n_avg=50)

ax.legend()
ax.set_xlim(200, 1000)
ax.set_ylim(-.505, -.485)
fig.savefig("vit-all.svg")
# plt.show()
plt.close()
# %%

fig, ax = gen_fig_ax()
ax.axhline(y=energy_gs, label="Ground State", color='green')

plot_energy_vs_iteration('./results/comparing-nqs/rbm_msr_adamw.npz', 'RBM (MSR)', ax, n_avg=50)
plot_energy_vs_iteration('./results/comparing-nqs/ffn_msr_adamw.npz', 'FFN (MSR)', ax, n_avg=50)
ax.plot([], [])
plot_energy_vs_iteration('./results/comparing-nqs/vit_sym_adamw.npz', 'VitSym', ax, n_avg=50)

ax.legend()
ax.set_xlim(50, 1000)
ax.set_ylim(-.52, -.4)
fig.savefig("rbmmodphase-ffn-msr-vit-adamw.svg")
# plt.show()
plt.close()
# %%

fig, ax = gen_fig_ax()

plot_variance_vs_iteration('./results/comparing-nqs/rbm_msr.npz', 'RBM (MSR)', ax, n_avg=50)
plot_variance_vs_iteration('./results/comparing-nqs/ffn_msr.npz', 'FFN (MSR)', ax, n_avg=50)
ax.plot([], [])
plot_variance_vs_iteration('./results/comparing-nqs/vit_sym.npz', 'VitSym', ax, n_avg=50)

ax.legend()
ax.set_xlim(50, 1000)
# ax.set_ylim(-.52, -.4)
fig.savefig("rbm-ffn-msr-vit-variance.svg")
# plt.show()
plt.close()
# %%

energy = []
for model in ['rbm_msr', 'ffn_msr', 'vit_sym']:
    data = jnp.load(f'{dir}/{model}_adamw.npz')
    energy.append(data['Energy'][-50:].real.mean().item() / 4 / (l*l) )

print(energy)

energy = []
for model in ['rbm_msr', 'ffn_msr', 'vit_sym']:
    data = jnp.load(f'{dir}/{model}.npz')
    energy.append(data['Energy'][-50:].real.mean().item() / 4 / (l*l) )

print(energy)

data = jnp.load(f'{dir}/vit_sym.npz')
print(data['Energy'][500-50:500].real.mean().item() / 4 / (l*l) )
# %%

for model in ['rbm', 'rbm_msr', 'rbmmodphase', 'rbmmodphase_msr', 'ffn', 'ffn_msr', 'ffnmodphase',
              'ffnmodphase_msr', 'ffn_complex', 'ffn_complex_msr', 'vit', 'vit_msr', 'vit_sym',
              'vit_sym_msr']:

    data = jnp.load(f'{dir}/{model}.npz')
    energy = data['Energy'][-50:].real.mean().item() / 4 / (l*l)

    print(f'{model}: {energy:.6f}')
# %%

energy_ed = [-0.678872, -0.503810, -0.586487]
for j2, energy_ed in zip(['_j2_0', '', '_j2_8'], energy_ed):
    print(j2)
    for model in ['rbm_msr', 'rbmmodphase_msr', 'ffn_msr', 'ffn_complex_msr', 'ffnmodphase_msr',
                  'vit', 'vit_sym']:

        data = jnp.load(f'{dir}/{model}{j2}.npz')
        energy = data['Energy'][-50:].real.mean().item() / 4 / (l*l)
        error = (energy_ed - energy) / energy_ed * 100

        print(f'{model}: {energy:.6f}')
        print(f'error: {error: .6f}')
