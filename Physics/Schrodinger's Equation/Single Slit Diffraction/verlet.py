import jax
import jax.numpy as jnp
from scipy.constants import hbar, m_e, eV
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation, colors

jax.config.update('jax_enable_x64', True)

def apply_boundary_cond(
        x: jnp.ndarray,
        slit_loc: tuple[jnp.ndarray, int],
        ) -> jnp.ndarray:

    rows, col = slit_loc

    x = x.at[rows, col].set(0)

    return x

def gen_initial_cond(
        D: float,
        y_max: float,
        dx: float,
        dy: float,
        wavelength: float,
        a: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, int]]:

    x = jnp.arange(-2 * D, 1.5 * D + 2 * wavelength + dx/2, dx)
    y = jnp.arange(y_max, -y_max - dy/2, -dy)
    
    slit_index_x = int(round(2 * D / dx))
    n = int( (y.size - a / dy - 1) / 2 )
    slit_index_y = jnp.concatenate([jnp.arange(n), jnp.arange(y.size - n, y.size)])
    slit_loc = (slit_index_y, slit_index_x)

    x, y = jnp.meshgrid(x, y)
    k  = 2 * jnp.pi / wavelength
    ky = jnp.pi / (2 * y_max)
    kx = jnp.sqrt(k**2 - ky**2)

    w =  2 * wavelength
    x_front = -.5 * D + w
    x_back  = -2 * D + 3 * w
    f = 0.25 * (1 + jnp.tanh((x - x_back) / w)) * (1 - jnp.tanh((x - x_front) / w))
    f = f.at[:, slit_index_x:].set(0)

    alpha = jnp.cos(ky * y) * jnp.cos(kx * x) * f
    beta = jnp.cos(ky * y) * jnp.sin(kx * x) * f
    z = jnp.sqrt((alpha**2 + beta**2).sum() * dx * dy)
    alpha /= z
    beta /= z

    return alpha, beta, slit_loc

def ddx(x: jnp.ndarray, dx: float) -> jnp.ndarray:
    x_pad = jnp.pad(x, ((0, 0), (1, 1)))
    delta_x = x_pad[:, 2:] - x_pad[:, :-2]

    return delta_x / (2 * dx)

def ddy(x: jnp.ndarray, dy: float) -> jnp.ndarray:
    x_pad = jnp.pad(x, ((1, 1), (0, 0)))
    delta_x = x_pad[:-2] - x_pad[2:]

    return delta_x / (2 * dy)

def d2dx2(x: jnp.ndarray, dx: float) -> jnp.ndarray:
    x_pad = jnp.pad(x, ((0, 0), (1, 1)))
    delta = x_pad[:, 2:] - 2 * x_pad[:, 1:-1] + x_pad[:, :-2]

    return delta / dx**2

def d2dy2(x: jnp.ndarray, dy: float) -> jnp.ndarray:
    x_pad = jnp.pad(x, ((1, 1), (0, 0)))
    delta = x_pad[:-2] - 2 * x_pad[1:-1] + x_pad[2:]

    return delta / dy**2

def laplacian(x: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    return hbar / 2 / m_e * (d2dx2(x, dx) + d2dy2(x, dy))

@jax.jit(donate_argnames=("alpha", "beta", "dbeta"), static_argnames=("dx", "dy", "dt"))
def iterate(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        dbeta: jnp.ndarray,
        dx: float,
        dy: float,
        dt: float,
        slit_loc: tuple[jnp.ndarray, int],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    beta += dbeta
    alpha = apply_boundary_cond(alpha - dt * laplacian(beta, dx, dy), slit_loc)
    dbeta = apply_boundary_cond(dt / 2 * laplacian(alpha, dx, dy), slit_loc)
    beta += dbeta

    return alpha, beta, dbeta

@jax.jit
def calc_prob_current(alpha: jnp.ndarray, beta: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    jx = hbar / m_e * (-beta * ddx(alpha, dx) + alpha * ddx(beta, dx))
    jy = hbar / m_e * (-beta * ddy(alpha, dy) + alpha * ddy(beta, dy))

    return jnp.stack([jx, jy])

@jax.jit
def calc_prob_density(alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    return alpha**2 + beta**2

@jax.jit
def calc_intensity(alpha: jnp.ndarray, beta: jnp.ndarray, dx: float) -> jnp.ndarray:
    jx = hbar / m_e * (-beta * ddx(alpha, dx) + alpha * ddx(beta, dx))

    return jx

energy = 1e3 * eV
wavelength = (2 / m_e / energy)**.5 * jnp.pi * hbar
v = (2 * energy/ m_e)**.5
a = 3 * wavelength
D = 25 * wavelength
dh = wavelength / 20
y_max = (1.5**2 - 1)**.5 * D
dt = dh / v / 10
n_iter = int((2 * D + 2 * wavelength) / v / dt)

video_len = 10
fps = 60
n_frames = video_len * fps
step = max(n_iter // n_frames, 1)
prob_density_list = []

alpha, beta, slit_loc = gen_initial_cond(D=D, y_max=y_max, dx=dh, dy=dh, wavelength=wavelength, a=a)
dbeta = apply_boundary_cond(dt / 2 * laplacian(alpha, dx=dh, dy=dh), slit_loc)
density = 1000
ny, nx = alpha.shape 
startx = int(round(D / dh))
stopx  = int(round(3 * D / dh))
starty = 0
stopy = ny - starty
stepy = max(ny // density, 1)
stepx = max(nx // density, 1)

n_avg = int(wavelength / v / dt)
intensity = jnp.zeros(alpha.shape[0])
for i in tqdm(range(n_iter)):
    alpha, beta, dbeta = iterate(alpha, beta, dbeta, dx=dh, dy=dh, dt=dt, slit_loc=slit_loc)
    if i % step == 0:
        alpha_ = alpha[starty:stopy:stepy, startx:stopx:stepx]
        beta_ = beta[starty:stopy:stepy, startx:stopx:stepx]
        prob_density_list.append(calc_prob_density(alpha_, beta_))
    if i >= n_iter - n_avg:
        alpha_ = alpha[:, stopx-1:stopx+2]
        beta_ = beta[:, stopx-1:stopx+2]
        intensity += calc_intensity(alpha_, beta_, dx=dh)[:, 1] / n_avg

jnp.save('intensity', intensity)
# jnp.save('prob_density_list', prob_density_list)

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
# %% -----------------------------------------------------------------------------------------------

plt.close()
fig, ax = plt.subplots()
ax.set_xlabel(r'$y \, / \, \lambda$')
ax.set_ylabel(r'$\mathtt{I} \, / \, \mathtt{I}_0$')

y = jnp.arange(y_max, -y_max - dh/2, -dh)
y = y[starty:stopy]
plot_step = y.size // 50

r = jnp.sqrt(D**2 + y**2)
sin_theta = y / r
cos_theta = D / r
intensity_predicted = jnp.sinc(a / wavelength * sin_theta)**2 * cos_theta**2
intensity_predicted /= intensity_predicted.max()

ax.plot(y[::plot_step] / wavelength, intensity[::plot_step] / intensity.max(), '.', zorder=3, label='Simulated')
ax.plot(y / wavelength, intensity_predicted, label='Predicted', zorder=2)

ax.legend()
fig.savefig('intensity.pdf')
# %% -----------------------------------------------------------------------------------------------

plt.close()
fig, ax = plt.subplots()
ax.set_xlim(-D / wavelength, D / wavelength)
ax.set_ylim(-y_max / wavelength, y_max / wavelength)
ax.set_xlabel(r'$x \, / \, \lambda$')
ax.set_ylabel(r'$y \, / \, \lambda$')
ax.axvline(x=0, ymax=.5 - a/y_max/4, linewidth=1.5, color='black')
ax.axvline(x=0, ymin=.5 + a/y_max/4, linewidth=1.5, color='black')

u_ref = max(float(u.max()) for u in prob_density_list)
u = prob_density_list[0]
im = ax.imshow( u / u_ref,
               extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
               norm=colors.PowerNorm(gamma=0.3, vmin=0, vmax=1))

def animate(u):
    im.set_data(u / u_ref)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=prob_density_list, interval=1000/fps, blit=True)

ani.save('animation1.mkv', writer=writer, dpi=600)
# %% -----------------------------------------------------------------------------------------------

plt.close()
fig, ax = plt.subplots()
ax.set_xlim(0, D / wavelength)
ax.set_ylim(-y_max / wavelength, y_max / wavelength)
ax.set_xlabel(r'$x \, / \, \lambda$')
ax.set_ylabel(r'$y \, / \, \lambda$')

cutoff_t = int(D / 2 / v / dt / step)
u_ref = max(float(u.max()) for u in prob_density_list[cutoff_t:])
u = prob_density_list[cutoff_t]
cutoff_x = u.shape[1] // 2
u = u[:, cutoff_x:]
im = ax.imshow(u / u_ref, extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
               norm=colors.LogNorm(vmin=1e-4, vmax=1))

def animate(u):
    u = u[:, cutoff_x:]
    im.set_data(u / u_ref)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=prob_density_list[cutoff_t: ],
                              interval=1000/fps, blit=True)

ani.save('animation2.mkv', writer=writer, dpi=600)
