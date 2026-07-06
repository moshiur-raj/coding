# import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".98"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
from scipy.constants import hbar, eV, epsilon_0, e, m_u
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
        dz: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    x = jnp.arange(-2.5 * D, D + 2 * wavelength + dx / 2, dx)
    y = jnp.arange(-y_max, y_max + dy/2, dy)
    z = y

    x, y, z = jnp.meshgrid(x, y, z, indexing='ij', sparse=True)
    k  = 2 * jnp.pi / wavelength
    ky = jnp.pi / (2 * y_max)
    kz = ky
    kx = jnp.sqrt(k**2 - ky**2 - kz**2)

    w =  2 * wavelength
    x_front = -D + w
    x_back  = -2.5 * D + 3 * w
    f = 0.25 * (1 + jnp.tanh((x - x_back) / w)) * (1 - jnp.tanh((x - x_front) / w))

    base = jnp.cos(ky * y) * jnp.cos(kz * z) * f
    alpha = base * jnp.cos(kx * x)
    beta = base * jnp.sin(kx * x)
    norm = jnp.sqrt((alpha**2 + beta**2).sum() * dx * dy * dz)
    alpha /= norm
    beta /= norm

    r = jnp.sqrt(x**2 + y**2 + z**2)
    V = jnp.where(r < R, K * (3 * R**2 - r**2) / (2 * R**3), K / r)

    return alpha, beta, V / hbar

def ddx(x: jnp.ndarray, dx: float) -> jnp.ndarray:
    x_pad = jnp.pad(x, ((1, 1), (0, 0), (0, 0)))
    delta_x = x_pad[2:] - x_pad[:-2]

    return delta_x / (2 * dx)

def ddy(x: jnp.ndarray, dy: float) -> jnp.ndarray:
    x_pad = jnp.pad(x, ((0, 0), (1, 1), (0, 0)))
    delta_x = x_pad[:, 2:] - x_pad[:, :-2]

    return delta_x / (2 * dy)

def ddz(x: jnp.ndarray, dz: float) -> jnp.ndarray:
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (1, 1)))
    delta_x = x_pad[:, :, 2:] - x_pad[:, :, :-2]

    return delta_x / (2 * dz)

def laplacian(x: jnp.ndarray, dx: float, dy: float, dz: float) -> jnp.ndarray:
    x_pad = jnp.pad(x, ((1, 1), (1, 1), (1, 1)))

    delta  = ( x_pad[2:, 1:-1, 1:-1] - 2 * x + x_pad[:-2, 1:-1, 1:-1] ) / dx**2
    delta += ( x_pad[1:-1, 2:, 1:-1] - 2 * x + x_pad[1:-1, :-2, 1:-1] ) / dy**2
    delta += ( x_pad[1:-1, 1:-1, 2:] - 2 * x + x_pad[1:-1, 1:-1, :-2] ) / dz**2

    return delta

@jax.jit(donate_argnames=("alpha", "beta", "dbeta"), static_argnames=("m", "dx", "dy", "dz", "dt"))
def iterate(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        dbeta: jnp.ndarray,
        V_norm: jnp.ndarray,
        m: float,
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    beta += dbeta
    alpha = alpha + dt * ( - hbar / 2 / m * laplacian(beta, dx, dy, dz) + V_norm * beta )
    dbeta = dt / 2 * ( hbar / 2 / m * laplacian(alpha, dx, dy, dz) - V_norm * alpha )
    beta += dbeta

    return alpha, beta, dbeta

@jax.jit(static_argnames=("m", "dx", "dy", "dz"))
def calc_prob_current(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        m: float,
        dx: float,
        dy: float,
        dz: float
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    jx = hbar / m * (-beta * ddx(alpha, dx) + alpha * ddx(beta, dx))
    jy = hbar / m * (-beta * ddy(alpha, dy) + alpha * ddy(beta, dy))
    jz = hbar / m * (-beta * ddz(alpha, dz) + alpha * ddz(beta, dz))

    return jx, jy, jz

@jax.jit
def calc_prob_density(alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    return alpha**2 + beta**2

@jax.jit(static_argnames=("m", "dx"))
def calc_intensity(alpha: jnp.ndarray, beta: jnp.ndarray, m: float, dx: float) -> jnp.ndarray:
    jx = hbar / m * (-beta * ddx(alpha, dx) + alpha * ddx(beta, dx))

    return jx

Z1 = 2
Z2 = 79
m = 4.0015 * m_u
energy = 20e6 * eV
wavelength = (2 / m / energy)**.5 * jnp.pi * hbar
v = (2 * energy/ m)**.5
K = 1 / 4 / jnp.pi / epsilon_0 * Z1 * Z2 * e**2
a = K / energy
D = 6 * a
dh = wavelength / 5
y_max = D + 2 * wavelength
dt = dh / v / 20
n_iter = int((2 * D + 2 * wavelength) / v / dt)

video_len = 10
fps = 60
n_frames = video_len * fps
step = max(n_iter // n_frames, 1)
prob_density_list = []

alpha, beta, V_norm = initialize(D, y_max, wavelength, K, R=.75*a, dx=dh, dy=dh, dz=dh)
dbeta = dt / 2 * ( hbar / 2 / m * laplacian(alpha, dx=dh, dy=dh, dz=dh) - V_norm * alpha )
density = 1000
startx = int(round(1.5 * D / dh))
stopx  = int(round(3.5 * D / dh))
nx, ny, nz = alpha.shape
mid = nz // 2
starty = 0
stopy = ny - starty
stepy = max(ny // density, 1)
stepx = max(nx // density, 1)

n_avg = int(wavelength / v / dt)
intensity = jnp.zeros((ny, nz))
for i in tqdm(range(n_iter)):
    alpha, beta, dbeta = iterate(alpha, beta, dbeta, V_norm, m, dx=dh, dy=dh, dz=dh, dt=dt)
    if i % step == 0:
        alpha_ = alpha[startx:stopx:stepx, starty:stopy:stepy, mid]
        beta_ = beta[startx:stopx:stepx, starty:stopy:stepy, mid]
        prob_density_list.append(calc_prob_density(alpha_, beta_))
    # if i >= n_iter - n_avg:
    #     alpha_ = alpha[stopx-1:stopx+2]
    #     beta_ = beta[stopx-1:stopx+2]
    #     intensity += calc_intensity(alpha_, beta_, m, dx=dh)[1] / n_avg

# jnp.save('intensity', intensity)
# jnp.save('prob_density_list', prob_density_list)
# %% -----------------------------------------------------------------------------------------------

plt.close()
fig, ax = plt.subplots()
ax.set_xlim(-D / wavelength, D / wavelength)
ax.set_ylim(-y_max / wavelength, y_max / wavelength)
ax.set_xlabel(r'$x \, / \, \lambda$')
ax.set_ylabel(r'$y \, / \, \lambda$')

u_ref = max(float(u.max()) for u in prob_density_list)
u = prob_density_list[0]
im = ax.imshow( u.T / u_ref, origin='lower',
               extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
               norm=colors.PowerNorm(gamma=0.3, vmin=0, vmax=1))

def animate(u):
    im.set_data(u.T / u_ref)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=prob_density_list, interval=1000/fps, blit=True)

writer = animation.FFMpegWriter(fps=fps, codec='libx264')
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
