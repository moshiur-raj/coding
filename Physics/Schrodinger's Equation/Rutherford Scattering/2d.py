# import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".85"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
from scipy.constants import hbar, eV, epsilon_0, e, m_p
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation, colors

jax.config.update('jax_enable_x64', True)

def initialize(
        D: float,
        y_max: float,
        wavelength: float,
        K: float,
        R: float,
        dx: float,
        dy: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    x = jnp.arange(-2.5 * D, 2 * D + dx / 2, dx)
    y = jnp.arange(-y_max, y_max + dy/2, dy)
    y += dy / 2 * jnp.any(y == 0)
    x, y = jnp.meshgrid(x, y, indexing='ij', sparse=True)

    k  = 2 * jnp.pi / wavelength
    ky = jnp.pi / (2 * y_max)
    kx = jnp.sqrt(k**2 - ky**2)

    w = 2 * wavelength
    x_front = - D + w
    x_back  = -2.5 * D + 3 * w
    f = 0.25 * (1 + jnp.tanh((x - x_back) / w)) * (1 - jnp.tanh((x - x_front) / w))
    # plt.plot(x.squeeze() / D, f.squeeze()); plt.show()

    alpha = jnp.cos(ky * y) * jnp.cos(kx * x) * f
    beta  = jnp.cos(ky * y) * jnp.sin(kx * x) * f
    norm = jnp.sqrt((alpha**2 + beta**2).sum() * dx * dy)
    alpha /= norm
    beta /= norm

    r = jnp.sqrt(x**2 + y**2)
    V = jnp.where(r < R, K * (3 * R**2 - r**2) / (2 * R**3), K / r)

    return alpha, beta, x, y, V

def ddx(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    f_pad = jnp.pad(f, ((1, 1), (0, 0)))
    delta = f_pad[2:] - f_pad[:-2]

    return delta / (2 * dx)

def ddy(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    f_pad = jnp.pad(f, ((0, 0), (1, 1)))
    delta = f_pad[:, 2:] - f_pad[:, :-2]

    return delta / (2 * dy)

def laplacian(f: jnp.ndarray, y_inv: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    f_pad = jnp.pad(f, ((1, 1), (1, 1)))

    delta  = ( f_pad[2:, 1:-1] - 2 * f + f_pad[:-2, 1:-1] ) / dx**2
    delta += y_inv * ddy(f, dy)
    delta += ( f_pad[1:-1, 2:] - 2 * f + f_pad[1:-1, :-2] ) / dy**2

    return delta

@jax.jit(donate_argnames=("alpha", "beta", "dbeta"), static_argnames=("m", "dx", "dy", "dt"))
def iterate(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        dbeta: jnp.ndarray,
        y_inv: jnp.ndarray,
        V: jnp.ndarray,
        m: float,
        dx: float,
        dy: float,
        dt: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    beta += dbeta
    alpha = alpha + dt * ( - hbar / 2 / m * laplacian(beta, y_inv, dx, dy) + 1 / hbar * V * beta )
    dbeta = dt / 2 * ( hbar / 2 / m * laplacian(alpha, y_inv, dx, dy) - 1 / hbar * V * alpha )
    beta += dbeta

    return alpha, beta, dbeta

@jax.jit
def calc_prob_density(alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    return alpha**2 + beta**2

def calc_prob_current(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        m: float,
        dx: float,
        dy: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:

    jx = hbar / m * (-beta * ddx(alpha, dx) + alpha * ddx(beta, dx))
    jy = hbar / m * (-beta * ddy(alpha, dy) + alpha * ddy(beta, dy))

    return jx, jy

@jax.jit(static_argnames=("m", "dx", "dy"))
def calc_cross_section(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        r0: float,
        m: float,
        dx: float,
        dy: float
        ) -> jnp.ndarray:

    r = x**2 + y**2

    jx, jy = calc_prob_current(alpha, beta, m, dx, dy)
    jr = (x * jx + y * jy) / jnp.sqrt(r)

    return jr

Z1 = 1
Z2 = 29
m = m_p
energy = 10e6 * eV
wavelength = (2 / m / energy)**.5 * jnp.pi * hbar
K = 1 / 4 / jnp.pi / epsilon_0 * Z1 * Z2 * e**2
a = K / energy
D = 100 * a
dh = wavelength / 20
y_max = 2 * D
energy_offset = (hbar * jnp.pi / 2 * y_max)**2 / 2 / m
v = (2 * (energy - energy_offset) / m)**.5
dt = dh / v / 20
n_iter = int(3 * D / v / dt)

video_len = 10
fps = 60
n_frames = video_len * fps
step = max(n_iter // n_frames, 1)
prob_density_list = []

alpha, beta, x, y, V = initialize(D, y_max, wavelength, K, R=.75*a, dx=dh, dy=dh)
y_inv = 1 / y
dbeta = dt / 2 * ( hbar / 2 / m * laplacian(alpha, y_inv, dx=dh, dy=dh) - 1 / hbar * V * alpha )
density = 1000
nx, ny = alpha.shape
stepy = max(ny // density, 1)
stepx = max(nx // density, 1)

n_avg = int(wavelength / v / dt)
cpu = jax.devices('cpu')[0]
for i in tqdm(range(n_iter)):
    alpha, beta, dbeta = iterate(alpha, beta, dbeta, y_inv, V, m, dx=dh, dy=dh, dt=dt)
    if i % step == 0:
        alpha_ = alpha[::stepx, ::stepy]
        beta_ = beta[::stepx, ::stepy]
        rho = calc_prob_density(alpha_, beta_)
        prob_density_list.append(jnp.asarray(rho, device=cpu))
# %% -----------------------------------------------------------------------------------------------

plt.close()
fig, ax = plt.subplots()
ax.set_xlim(-2.5 * D / a, 2 * D / a)
ax.set_ylim(-y_max / a, y_max / a)
ax.set_xlabel(r'$x \, / \, a$')
ax.set_ylabel(r'$y \, / \, a$')

rho_ref = max(float(u.max()) for u in prob_density_list)
rho = prob_density_list[0]
im = ax.imshow( rho.T / rho_ref, origin='lower',
               extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
               norm=colors.PowerNorm(gamma=0.3, vmin=0, vmax=1))

def animate(rho):
    im.set_data(rho.T / rho_ref)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=prob_density_list, interval=1000/fps, blit=True)

# writer = animation.FFMpegWriter(fps=fps, codec='libx264')
writer = animation.FFMpegWriter(
        fps=fps,
        codec="hevc_vulkan",
        extra_args=[
            "-init_hw_device", "vulkan=vk:0",
            "-filter_hw_device", "vk",
            "-vf", "format=nv12,hwupload",
            "-qp", "22",
            ],
        )
ani.save('animation.mkv', writer=writer, dpi=600)
# %% -----------------------------------------------------------------------------------------------

# plt.close()
# fig, ax = plt.subplots()
# ax.set_xlabel(r'$y \, / \, \lambda$')
# ax.set_ylabel(r'$\mathtt{I} \, / \, \mathtt{I}_0$')
#
# y = jnp.arange(-y_max, y_max + dh/2, dh)
# y = y[starty:stopy]
# plot_step = y.size // 50
#
# r = jnp.sqrt(D**2 + y**2)
# sin_theta = y / r
# cos_theta = D / r
# intensity_predicted = jnp.sinc(a / wavelength * sin_theta)**2 * cos_theta**2
# intensity_predicted /= intensity_predicted.max()
#
# ax.plot(y[::plot_step] / wavelength, intensity[::plot_step] / intensity.max(), '.', zorder=3, label='Simulated')
# ax.plot(y / wavelength, intensity_predicted, label='Predicted', zorder=2)
#
# ax.legend()
# fig.savefig('intensity.pdf')
