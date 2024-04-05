import os
import numpy as np
from math import floor, log10
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors, animation
from functools import partial

import exact_solutions

markersize = 3
# spin = -1 => white, spin = +1 => black
cmap = colors.ListedColormap(['white', 'black'])
bounds=[-1,0,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

def scientific_notation_latex(number):
    exponent = 0
    if number != 0:
        exponent = int(floor(log10(abs(number))))
    coefficient = number / 10**exponent
    if round(coefficient, 2) == 10:
        coefficient = 1.0
        exponent += 1
    return f"${coefficient:.2f} \\times 10^{{{exponent}}}$"

def gen_2dplot(t, frameon):
    fig = plt.figure(num=1, frameon=frameon)
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', width=0.5, length=3)
    ax.tick_params(which='both', top=True, right=True)
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle='dotted')
    ax.set_xlabel(r"Normalized Temperature, ${T}\,/\,{T_c}$")
    ax.set_ylabel(r"Spontaneous Magnetization, $\langle s \rangle$")
    ax.set_xlim(0, t.max() + 0.05)
    ax.set_ylim(0, 1.25)
    return fig, ax

def plot_spontaneous_magnetization(t, s_f, s_avg, frameon):
    fig, ax = gen_2dplot(t, frameon)
    exact_solutions.plot_spontaneous_magnetization_exact(ax)
    ax.plot(t, s_f, marker='', markersize=markersize, linestyle='-', label="Final State")
    ax.plot(t, s_avg, marker='', markersize=markersize, linestyle='-', label="Ensemble Average")
    ax.legend()
    fig.savefig("./figures/magnetization.svg")
    plt.close()

def uniform_array_cond(t, n):
    res = []
    dt = (t.max() - t.min()) / (n - 1)
    t_ = t.min()
    for i in range(0, t.size):
        cond = (t[i] >= t_)
        res.append(cond)
        t_ += dt * cond
    return np.array(res)

def plot_thermodynamic_functions(t, s_f, s_avg, u_f, frameon):
    plot_spontaneous_magnetization(t, s_f, s_avg, frameon)
    cond = uniform_array_cond(t, int((t.max() - t.min())*40))
    t_ = t[cond]
    u_f_ = u_f[cond]
    fig, ax = gen_2dplot(t_, frameon)
    exact_solutions.plot_average_energy_exact(ax, t.max())
    ax.plot(t_, u_f_, marker='.', markersize=4, linestyle='', label="Average Energy")
    exact_solutions.plot_heat_capacity_exact(ax, t.max())
    c_ = np.gradient(u_f_, t_)
    ax.plot(t_, c_, marker='.', markersize=4, linestyle='', label="Heat Capacity")
    ax.set_ylabel(r"Normalized Thermodyamic Functions")
    ax.set_xlim(0, t_.max())
    ax.set_ylim(-4, 4)
    ax.legend()
    fig.savefig("./figures/thermodynamic_functions.svg")
    plt.close()
    
def plot_thermodynamic_functions_for_different_h(h_norm, t, s_f, u_f, frameon):
    fig, ax = gen_2dplot(t, frameon)
    exact_solutions.plot_spontaneous_magnetization_exact(ax, label=("h = 0 (exact solution)"))
    for i in range(0, h_norm.size):
        ax.plot(t, s_f[i], marker='', markersize=markersize, linestyle='-',
                label="h_norm = "+scientific_notation_latex(h_norm[i]))
    ax.legend()
    fig.savefig("./figures/magnetization_for_different_h_norm.svg")
    plt.close()
    cond = uniform_array_cond(t, int((t.max() - t.min())*75))
    t_ = t[cond]
    fig, ax = gen_2dplot(t_, frameon)
    exact_solutions.plot_average_energy_exact(ax, t.max())
    for i in range(0, h_norm.size):
        u_f_ = u_f[i][cond]
        ax.plot(t_, u_f_, marker='', markersize=markersize, linestyle='-',
                label="h_norm = "+scientific_notation_latex(h_norm[i]))
    ax.set_ylabel(r"Normalized Average Energy")
    ax.set_xlim(0, t_.max())
    ax.set_ylim(-2.5, 0)
    ax.legend()
    fig.savefig("./figures/thermodynamic_functions_for_different_h_norm.svg")
    plt.close()

def gen_cmap_plot(frameon=True):
    fig = plt.figure(num=1, frameon=frameon)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return fig, ax

def plot_magnetic_domains(t, domain, frameon):
    fig, ax = gen_cmap_plot(frameon)
    ax.imshow(domain[0], interpolation='none', origin='upper', cmap=cmap, norm=norm,
              aspect='equal')
    ax.set_xlabel(f"Initial State\t" r"$\langle s \rangle$"
                  f" ={domain[0].sum()/domain[0].size:.2f}")
    fig.savefig(f"./figures/initial_domain.svg")
    # remove previously saved images. this is required as the s=... part of the filename can be
    # different in each run.
    os.system("rm ./figures/magnetic_domain_*.svg")
    for i in range(0, t.size):
        ax.clear()
        ax.imshow(domain[i+1], interpolation='none', origin='upper', cmap=cmap, norm=norm,
                  aspect='equal')
        if frameon:
            ax.set_xlabel(r"${T}\,/\,{T_c}$" f" ={t[i]:.2f}\t\t" r"$\langle s \rangle$" 
                          f" ={domain[i+1].sum()/domain[i+1].size:.2f}")
        fig.savefig(f"./figures/magnetic_domain_t={t[i]:.2f}"
                    + f"_s={domain[i+1].sum()/domain[i+1].size:.2f}.svg")
    plt.close()

def animate(frame, ax, lattice, t_anim, iter):
    ax.clear()
    ax.set_xlabel(r"${T}\,/\,{T_c}$" f" ={t_anim:.2f}\t" + scientific_notation_latex(iter[frame])
                  + f" Iterations\t" r"$\langle s \rangle$"
                  f" ={lattice[frame].sum()/lattice[frame].size:.2f}")
    img = ax.imshow(lattice[frame], interpolation='none', origin='upper', cmap=cmap, norm=norm,
                    aspect='equal')
    return img

def create_animation(lattice, t_anim, iter, fps, res_width):
    fig, ax = gen_cmap_plot()
    dpi = int(res_width / fig.get_size_inches()[0])
    anim = animation.FuncAnimation(fig, partial(animate, ax=ax, lattice=lattice, t_anim=t_anim,
                                   iter=iter), frames=lattice.shape[0], interval=round(1000/fps),
                                   blit=True)
    if os.path.exists("/dev/dri/renderD128"):
        # Use hardware encoder (vaapi) with ffmpeg
        writer = animation.FFMpegWriter(fps=fps, codec="hevc_vaapi",extra_args=["-vaapi_device",
                                        "/dev/dri/renderD128", "-vf", "format=nv12,hwupload"])
    else:
        # No hardware acceleration
        writer = animation.FFMpegWriter(fps=fps, codec="hevc")
    # Save the video
    anim.save('./animation.mkv', writer=writer, dpi=dpi)
    plt.close()

def plot_convergence_iterations(t, s_f, n_iter, frameon):
    fig, ax = gen_2dplot(t, frameon)
    exact_solutions.plot_spontaneous_magnetization_exact(ax)
    for i in range(0, n_iter.size):
        ax.plot(t, s_f[i], marker='', markersize=markersize, linestyle='-',
                label=scientific_notation_latex(n_iter[i]) + f" Iterations")
    ax.legend()
    fig.savefig("./figures/convergence_iterations.svg")
    plt.close()

def plot_convergence_lattice_size(t, s_f, lattice_size_mult, n_rows, n_cols, frameon):
    fig, ax = gen_2dplot(t, frameon)
    exact_solutions.plot_spontaneous_magnetization_exact(ax)
    for i in range(0, lattice_size_mult.size):
        # the curves fluctuate too much. only plot the points, do not join them with lines.
        ax.plot(t, s_f[i], marker='.', markersize=markersize, linestyle='', label=
                f"{lattice_size_mult[i]*n_rows:.0f} x {lattice_size_mult[i]*n_cols:.0f} Lattice")
    ax.legend()
    fig.savefig("./figures/convergence_lattice_size.svg")
    plt.close()
