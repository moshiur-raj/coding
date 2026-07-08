# import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".85"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
from scipy.constants import hbar, eV, epsilon_0, e, m_u
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation, colors

jax.config.update('jax_enable_x64', True)

@jax.jit(static_argnames=("dxdy"))
def total_prob(alpha: jnp.ndarray, beta: jnp.ndarray, y: jnp.ndarray, dxdy: float) -> jnp.ndarray:
    return 2 * jnp. pi * (y * (alpha**2 + beta**2)).sum() * dxdy

def initialize(
        D: float,
        y_max: float,
        wavelength: float,
        K: float,
        R: float,
        dx: float,
        dy: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    x = jnp.arange(-2 * D, 1.5 * D + dx / 2, dx)
    y = jnp.arange(dy / 2, y_max + dy / 2, dy)
    x, y = jnp.meshgrid(x, y, indexing='ij', sparse=True)

    kx  = 2 * jnp.pi / wavelength
    ky = jnp.pi / (2 * y_max)

    w = 2 * wavelength
    x_front = - D + w
    x_back  = x.min() + 3 * w
    f = 0.25 * (1 + jnp.tanh((x - x_back) / w)) * (1 - jnp.tanh((x - x_front) / w))
    # plt.plot(x.squeeze() / D, f.squeeze()); plt.show()

    alpha = jnp.cos(ky * y) * jnp.cos(kx * x) * f
    beta  = jnp.cos(ky * y) * jnp.sin(kx * x) * f
    norm = jnp.sqrt(total_prob(alpha, beta, y, dx * dy))
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
    f_pad = jnp.pad(f,     ((0, 0), (0, 1)))
    f_pad = jnp.pad(f_pad, ((0, 0), (1, 0)), mode='edge')

    delta = f_pad[:, 2:] - f_pad[:, :-2]

    return delta / (2 * dy)

def laplacian(f: jnp.ndarray, y_inv: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    f_pad = jnp.pad(f,     ((1, 1), (0, 1)))
    f_pad = jnp.pad(f_pad, ((0, 0), (1, 0)), mode='edge')

    delta  = ( f_pad[2:, 1:-1] - 2 * f + f_pad[:-2, 1:-1] ) / dx**2
    delta += y_inv * ddy(f, dy)
    delta += ( f_pad[1:-1, 2:] - 2 * f + f_pad[1:-1, :-2] ) / dy**2

    return delta

@jax.jit(donate_argnames=("alpha", "beta", "dbeta"), static_argnames=("m", "dx", "dy", "dt"))
def iterate_zero_field(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        dbeta: jnp.ndarray,
        y_inv: jnp.ndarray,
        m: float,
        dx: float,
        dy: float,
        dt: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    beta += dbeta
    alpha = alpha - dt * hbar / 2 / m * laplacian(beta, y_inv, dx, dy)
    dbeta = dt / 2 * hbar / 2 / m * laplacian(alpha, y_inv, dx, dy)
    beta += dbeta

    return alpha, beta, dbeta

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
    return (alpha**2 + beta**2)

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

@jax.jit(static_argnames=("dxdy"))
def extract_outgoing_wave(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        alpha0: jnp.ndarray,
        beta0: jnp.ndarray,
        y: jnp.ndarray,
        dxdy: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:

    c_real = 2 * jnp.pi * ((alpha0 * alpha + beta0 * beta)  * y).sum() * dxdy
    c_imag = 2 * jnp.pi * ((alpha0 * beta  - beta0 * alpha) * y).sum() * dxdy

    alpha_out = alpha - (c_real * alpha0 - c_imag * beta0 )
    beta_out  = beta  - (c_real * beta0  + c_imag * alpha0)

    return alpha_out, beta_out

@jax.jit(static_argnames=("r0", "m", "dx", "dy", "n"))
def calc_cross_section(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        m: float,
        r0: float,
        dx: float,
        dy: float,
        n: int,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:

    r = jnp.sqrt(x**2 + y**2)

    jx, jy = calc_prob_current(alpha, beta, m, dx, dy)
    jr = (x * jx + y * jy) / r

    theta = jnp.linspace(0, jnp.pi, n).reshape(-1, 1)
    xc = r0 * jnp.cos(theta)
    yc = r0 * jnp.sin(theta)

    x_indices = jnp.argmin((x[:, 0] - xc)**2, axis=1).flatten()
    y_indices = jnp.argmin((y[0, :] - yc)**2, axis=1).flatten()

    cross_section= jr[x_indices, y_indices] * r[x_indices, y_indices]**2

    return cross_section, theta

Z1 = 1
Z2 = 1
m = .5 * m_u
energy = 10e6 * eV
wavelength = (2 / m / energy)**.5 * jnp.pi * hbar
K = 1 / 4 / jnp.pi / epsilon_0 * Z1 * Z2 * e**2
a = K / energy
D = 1000 * a
dh = wavelength / 20
y_max = 1.5 * D
v = (2 * energy / m)**.5
dt = dh / v / 20
n_iter = int(2.5 * D / v / dt)

video_len = 10
fps = 60
n_frames = video_len * fps
step = max(n_iter // n_frames, 1)

alpha, beta, x, y, V = initialize(D, y_max, wavelength, K, R=.75*a, dx=dh, dy=dh)
y_inv = 1 / y
dbeta = dt / 2 * ( hbar / 2 / m * laplacian(alpha, y_inv, dx=dh, dy=dh) - 1 / hbar * V * alpha )
density = 1000
nx, ny = alpha.shape
stepx = max(nx // density, 1)
stepy = max(ny // density, 1)

cpu = jax.devices('cpu')[0]
alpha0, beta0 = alpha.copy(), beta.copy()
dbeta0 = dt / 2 * hbar / 2 / m * laplacian(alpha0, y_inv, dx=dh, dy=dh)
dxdy = dh**2
frames1 = []
frames2 = []
rho_ref = calc_prob_density(alpha, beta).max()
pbar = tqdm(range(n_iter))
for i in pbar:
    alpha0, beta0, dbeta0 = iterate_zero_field(alpha0, beta0, dbeta0, y_inv, m, dx=dh, dy=dh, dt=dt)
    alpha, beta, dbeta = iterate(alpha, beta, dbeta, y_inv, V, m, dx=dh, dy=dh, dt=dt)
    pbar.set_postfix(
            P=f"{total_prob(alpha, beta, y, dxdy):.2f}",
            P0=f"{total_prob(alpha0, beta0, y, dxdy):.2f}",
            )
    if i % step == 0:
        rho = calc_prob_density(alpha[::stepx, ::stepy], beta[::stepx, ::stepy])
        frames1.append(jnp.asarray(rho / rho_ref, device=cpu, dtype=jnp.float32))
        alpha_out, beta_out = extract_outgoing_wave(alpha, beta, alpha0, beta0, y, dxdy)
        rho = calc_prob_density(alpha_out[::stepx, ::stepy], beta_out[::stepx, ::stepy])
        frames2.append(jnp.asarray(rho / rho_ref, device=cpu, dtype=jnp.float32))

jnp.savez('wavefunction', psi=jnp.stack([alpha, beta]), psi0=jnp.stack([alpha0, beta0]))
# %% -----------------------------------------------------------------------------------------------

cross_section, theta = calc_cross_section(alpha_out, beta_out, x, y, m, r0=300*a, dx=dh, dy=dh, n=1000)

plt.close()
plt.plot(theta, cross_section / cross_section[-1])
plt.show()

# %% -----------------------------------------------------------------------------------------------

plt.close()
fig, ax = plt.subplots()
ax.set_xlim(-2 * D / a, 1.5 * D / a)
ax.set_ylim(-y_max / a, y_max / a)
ax.set_xlabel(r'$x \, / \, a$')
ax.set_ylabel(r'$y \, / \, a$')

rho_ref = max(float(rho.max()) for rho in frames1)
rho = frames1[0]
rho = jnp.concatenate([rho[:, ::-1], rho], axis=1)
im = ax.imshow( rho.T / rho_ref, origin='lower',
               extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
               norm=colors.PowerNorm(gamma=0.3, vmin=0, vmax=1))

def animate(rho):
    rho = jnp.concatenate([rho[:, ::-1], rho], axis=1)
    im.set_data(rho.T / rho_ref)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=frames1, interval=1000/fps, blit=True)

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
ani.save('animation1.mkv', writer=writer, dpi=600)
# %% -----------------------------------------------------------------------------------------------

plt.close()
fig, ax = plt.subplots()
ax.set_xlim(-2 * D / a, 1.5 * D / a)
ax.set_ylim(-y_max / a, y_max / a)
ax.set_xlabel(r'$x \, / \, a$')
ax.set_ylabel(r'$y \, / \, a$')

rho_ref = max(float(rho.max()) for rho in frames2)
rho = frames2[0]
rho = jnp.concatenate([rho[:, ::-1], rho], axis=1)
im = ax.imshow( rho.T / rho_ref, origin='lower',
               extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
               norm=colors.PowerNorm(gamma=0.3, vmin=0, vmax=1))

def animate(rho):
    rho = jnp.concatenate([rho[:, ::-1], rho], axis=1)
    im.set_data(rho.T / rho_ref)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=frames2, interval=1000/fps,
                              blit=True)

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
ani.save('animation2.mkv', writer=writer, dpi=600)
