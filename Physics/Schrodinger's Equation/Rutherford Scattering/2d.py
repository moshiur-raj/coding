# Spherical coordinates are used with reflection symmetry

# import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".85"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
from scipy.special import j0, jn_zeros
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
        a: float,
        dx: float,
        dy: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    x = jnp.arange(-2 * D, y_max + dx / 2, dx)
    y = jnp.arange(dy / 2, y_max + dy / 2, dy)
    x, y = jnp.meshgrid(x, y, indexing='ij', sparse=True)

    kx  = 2 * jnp.pi / wavelength

    w = 2 * wavelength
    x_front = - D + w
    x_back  = x.min() + 3 * w
    fx = 0.25 * (1 + jnp.tanh((x - x_back) / w)) * (1 - jnp.tanh((x - x_front) / w))
    fy = jnp.asarray(j0(jn_zeros(0, 1) / y_max * y))
    # plt.plot(x.squeeze() / D, f.squeeze()); plt.show()

    alpha = jnp.cos(kx * x) * fx * fy
    beta  = jnp.sin(kx * x) * fx * fy
    norm = jnp.sqrt(total_prob(alpha, beta, y, dx * dy))
    alpha /= norm
    beta /= norm

    r = jnp.sqrt(x**2 + y**2)
    r0 = a /  4
    V = jnp.where(r < r0, K * (3 * r0**2 - r**2) / (2 * r0**3), K / r)
    V *= jnp.exp(- r / (100 * a))

    return alpha, beta, x, y, V

def gen_detector_loc(
        x: jnp.ndarray,
        y: jnp.ndarray,
        r0: float,
        n: int,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    theta = jnp.linspace(0, jnp.pi, n)
    xc = r0 * jnp.cos(theta).reshape(-1, 1)
    yc = r0 * jnp.sin(theta).reshape(-1, 1)

    ix = jnp.argmin((x[:, 0] - xc)**2, axis=1).flatten()
    iy = jnp.argmin((y[0, :] - yc)**2, axis=1).flatten()

    return theta, ix, iy

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

@jax.jit(static_argnames=("m", "dx", "dy", "dt", "phi0"))
def calc_cross_section(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        ix: jnp.ndarray,
        iy: jnp.ndarray,
        m: float,
        dx: float,
        dy: float,
        dt: float,
        phi0: float,
        ) -> jnp.ndarray:

    ddx_alpha = (alpha[ix + 1, iy] - alpha[ix - 1, iy]) / 2 / dx
    ddy_alpha = (alpha[ix, iy + 1] - alpha[ix, iy - 1]) / 2 / dy

    ddx_beta = (beta[ix + 1, iy] - beta[ix - 1, iy]) / 2 / dx
    ddy_beta = (beta[ix, iy + 1] - beta[ix, iy - 1]) / 2 / dy

    alpha = alpha[ix, iy]
    beta  = beta[ix, iy]
    x = x[ix, 0]
    y = y[0, iy]

    jx = hbar / m * (-beta * ddx_alpha + alpha * ddx_beta)
    jy = hbar / m * (-beta * ddy_alpha + alpha * ddy_beta)

    r = jnp.sqrt(x**2 + y**2)
    cs = (x * jx + y * jy) * r * dt / phi0

    return cs

Z1 = 1
Z2 = 1
m = m_u
energy = 10e6 * eV
wavelength = (2 / m / energy)**.5 * jnp.pi * hbar
K = 1 / 4 / jnp.pi / epsilon_0 * Z1 * Z2 * e**2
a = K / energy
D = 1000 * a
dh = wavelength / 60
y_max = 1.5 * D
v = (2 * energy / m)**.5
dt = dh / v / 20
n_iter = int(2.5 * D / v / dt)

video_len = 10
fps = 60
n_frames = video_len * fps
step = max(n_iter // n_frames, 1)

alpha, beta, x, y, V = initialize(D, y_max, wavelength, K, a, dx=dh, dy=dh)
y_inv = 1 / y
dbeta = dt / 2 * ( hbar / 2 / m * laplacian(alpha, y_inv, dx=dh, dy=dh) - 1 / hbar * V * alpha )
density = 1000
nx, ny = alpha.shape
stepx = max(nx // density, 1)
stepy = max(ny // density, 1)

r_d = D/2
theta, ix, iy = gen_detector_loc(x, y, r0=r_d, n=100)
cs = jnp.zeros_like(theta)

cpu = jax.devices('cpu')[0]
alpha0, beta0 = alpha.copy(), beta.copy()
dbeta0 = dt / 2 * hbar / 2 / m * laplacian(alpha0, y_inv, dx=dh, dy=dh)
dxdy = dh**2
phi0 = float((alpha[:, 0]**2 + beta[:, 0]**2).sum() * dh)
frames1 = []
frames2 = []
rho_ref = calc_prob_density(alpha, beta).max()
pbar = tqdm(range(n_iter))

for i in pbar:

    alpha0, beta0, dbeta0 = iterate_zero_field(alpha0, beta0, dbeta0, y_inv, m, dx=dh, dy=dh, dt=dt)
    alpha, beta, dbeta = iterate(alpha, beta, dbeta, y_inv, V, m, dx=dh, dy=dh, dt=dt)

    alpha_out, beta_out = alpha - alpha0, beta - beta0
    cs += calc_cross_section(alpha_out, beta_out, x, y, ix, iy, m, dx=dh, dy=dh, dt=dt, phi0=phi0)

    pbar.set_postfix(
            P=f"{total_prob(alpha, beta, y, dxdy):.2f}",
            P0=f"{total_prob(alpha0, beta0, y, dxdy):.2f}",
            )

    if i % step == 0:
        rho = calc_prob_density(alpha[::stepx, ::stepy], beta[::stepx, ::stepy])
        frames1.append(jnp.asarray(rho / rho_ref, device=cpu, dtype=jnp.float32))

        alpha_out, beta_out = alpha - alpha0, beta - beta0
        rho = calc_prob_density(alpha_out[::stepx, ::stepy], beta_out[::stepx, ::stepy])
        frames2.append(jnp.asarray(rho / rho_ref, device=cpu, dtype=jnp.float32))

jnp.save('data', jnp.stack([alpha, beta, alpha0, beta0]))
# %% -----------------------------------------------------------------------------------------------

plt.close()
plt.plot(theta[50:], cs[50:])
# plt.show()
plt.savefig('cross-section.pdf')
# %% -----------------------------------------------------------------------------------------------

plt.close()
fig_width = 10
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(fig_width, fig_width/2*3.5*D/2/y_max))

rho_refs = [max(float(rho.max()) for rho in frames) for frames in (frames1, frames2)]
titles = ['Total Wave', 'Outgoing Wave']

def mirror(rho):
    return jnp.concatenate([rho[:, ::-1], rho], axis=1)

ims = []
axes[0].set_ylabel(r'$y \, / \, a$')
for ax, title, rho, rho_ref in zip(axes, titles, (frames1[0], frames2[0]), rho_refs):
    ax.set_xlim(-2 * D / a, 1.5 * D / a)
    ax.set_ylim(-y_max / a, y_max / a)
    ax.set_xlabel(r'$x \, / \, a$')
    ax.set_title(title)
    im = ax.imshow(mirror(rho).T / rho_ref, origin='lower',
                   extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
                   norm=colors.PowerNorm(gamma=0.3, vmin=0, vmax=1))
    ims.append(im)

fig.tight_layout()

def animate(pair):
    for im, rho, rho_ref in zip(ims, pair, rho_refs):
        im.set_data(mirror(rho).T / rho_ref)
    return ims

ani = animation.FuncAnimation(fig, animate, frames=zip(frames1, frames2),
                              interval=1000/fps, blit=True)
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
ani.save('animation.mkv', writer=writer, dpi=3840//fig_width)
