import jax
import jax.numpy as jnp
from scipy.constants import hbar, m_e, eV
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation, colors

jax.config.update('jax_enable_x64', True)

def apply_boundary_cond(
        psi: jnp.ndarray,
        slit_inner: tuple[jnp.ndarray, int],
        ) -> jnp.ndarray:

    rows, col = slit_inner

    psi = psi.at[:, rows, col - 1].set(0)
    psi = psi.at[:, rows, col].set(0)
    psi = psi.at[:, rows, col + 1].set(0)

    return psi

def ddx(psi: jnp.ndarray, dx: float) -> jnp.ndarray:
    psi = jnp.atleast_3d(psi)
    psi_pad = jnp.pad(psi, ((0, 0), (0, 0), (1, 1)))
    delta_x = psi_pad[:, :, 2:] - psi_pad[:, :, :-2]

    return delta_x / (2 * dx)

def ddy(psi: jnp.ndarray, dy: float) -> jnp.ndarray:
    psi = jnp.atleast_3d(psi)
    psi_pad = jnp.pad(psi, ((0, 0), (1, 1), (0, 0)))
    delta_x = psi_pad[:, :-2] - psi_pad[:, 2:]

    return delta_x / (2 * dy)

def d2dx2(psi: jnp.ndarray, dx: float) -> jnp.ndarray:
    psi_pad = jnp.pad(psi, ((0, 0), (0, 0), (1, 1)))
    delta = psi_pad[:, :, 2:] - 2 * psi_pad[:, :, 1:-1] + psi_pad[:, :, :-2]

    return delta / dx**2

def d2dy2(psi: jnp.ndarray, dy: float) -> jnp.ndarray:
    psi_pad = jnp.pad(psi, ((0, 0), (1, 1), (0, 0)))
    delta = psi_pad[:, :-2] - 2 * psi_pad[:, 1:-1] + psi_pad[:, 2:]

    return delta / dy**2

def ddt(psi: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    dpsi_dt = hbar / 2 / m_e * (d2dx2(psi, dx) + d2dy2(psi, dy))

    return jnp.stack([-dpsi_dt[1], dpsi_dt[0]])

def gen_initial_cond(
        D: float,
        y_max: float,
        dx: float,
        dy: float,
        dt: float,
        wavelength: float,
        a: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, int]]:

    x = jnp.arange(-2 * D, 1.5 * D + dx/2, dx)
    y = jnp.arange(y_max + dy/2, -y_max, -dy)
    
    slit_index_x = int((x.size - 1) * 4/7)
    n = int( (y.size - a / dy - 1) / 2 )
    slit_index_y = jnp.concatenate([jnp.arange(n), jnp.arange(y.size - n, y.size)])
    slit_inner = (slit_index_y, slit_index_x)

    # x, y = jnp.meshgrid(x, y)
    # k = 2 * jnp.pi / wavelength
    # psi = jnp.exp(1j * k * x)
    #
    # cutoff_index = int( (1.5 * D // wavelength) * wavelength / dx )
    # psi = psi.at[:, cutoff_index:].set(0)

    x, y = jnp.meshgrid(x, y)
    k  = 2 * jnp.pi / wavelength
    ky = jnp.pi / (2 * y_max)
    kx = jnp.sqrt(k**2 - ky**2)

    w = 3 * wavelength
    x_front = -13 * wavelength
    x_back  = -2 * D + 3 * wavelength
    f = 0.25 * (1 + jnp.tanh((x - x_back) / w)) * (1 - jnp.tanh((x - x_front) / w))

    psi_prev = jnp.stack([
        jnp.cos(ky * y) * jnp.cos(kx * x) * f,
        jnp.cos(ky * y) * jnp.sin(kx * x) * f,
        ])

    ddt_psi = ddt(psi_prev, dx, dy)
    psi = psi_prev + ddt_psi * dt + ddt(ddt_psi, dx, dy) * dt**2 / 2
    psi = apply_boundary_cond(psi, slit_inner)

    return psi, psi_prev, slit_inner

@jax.jit(donate_argnames=("psi", "psi_prev"))
def iterate(
        psi: jnp.ndarray,
        psi_prev: jnp.ndarray,
        dx: float,
        dy: float,
        dt: float,
        slit_inner: tuple[jnp.ndarray, int],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:

    psi_next = psi_prev + ddt(psi, dx, dy) * 2 * dt
    psi_next = apply_boundary_cond(psi_next, slit_inner)

    return psi_next, psi

@jax.jit
def calc_prob_current(psi: jnp.ndarray, dx: float, dy: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    real = psi[0]
    imag = psi[1]
    jx = hbar / m_e * (-imag * ddx(real, dx) + real * ddx(imag, dx))
    jy = hbar / m_e * (-imag * ddy(real, dy) + real * ddy(imag, dy))

    return jx, jy

@jax.jit
def calc_prob_density(psi: jnp.ndarray) -> jnp.ndarray:
    return psi[0]**2 + psi[1]**2

@jax.jit
def calc_intensity(psi: jnp.ndarray, dx: float) -> jnp.ndarray:
    real = psi[0]
    imag = psi[1]
    jx = hbar / m_e * (-imag * ddx(real, dx) + real * ddx(imag, dx))

    return jx

energy = 10e3 * eV
wavelength = (2 / m_e / energy)**.5 * jnp.pi * hbar
v = (2 * energy/ m_e)**.5
a = 3 * wavelength
D = 25 * wavelength
dh = wavelength / 25
y_max = (1.5**2 - 1)**.5 * D
dt = 0.5 * m_e * dh**2 / (4 * hbar)
n_iter = int(2 * D / v / dt)

video_len = 10
fps = 60
n_frames = video_len * fps
step = max(n_iter // n_frames, 1)
# prob_current_list = []
prob_density_list = []

psi, psi_prev, slit_inner = gen_initial_cond(D=D, y_max=y_max, dx=dh, dy=dh, dt=dt,
                                             wavelength=wavelength, a=a)
density = 1000
ny, nx = psi[0].shape 
ny = max(ny // density, 1)
nx = max(nx // density, 1)
nrows, ncols = psi[0].shape
startx = int(ncols * 2/7)
stopx = ncols - int(ncols * 1/7)
starty = 0
stopy = nrows - starty

n_avg = int(10 * wavelength / v / dt)
intensity = jnp.zeros(psi.shape[1])
for i in tqdm(range(n_iter)):
    psi, psi_prev = iterate(psi, psi_prev, dx=dh, dy=dh, dt=dt, slit_inner=slit_inner)
    if i % step == 0:
        psi_ = psi[starty:stopy:ny, startx:stopx:nx]
        # prob_current_list.append(calc_prob_current(psi, dx=dh, dy=dh)[starty:stopy:ny, startx:stopx:nx])
        prob_density_list.append(calc_prob_density(psi_))
    if i >= n_iter - n_avg:
        intensity += calc_intensity(psi[:, stopx-1:stopx+2], dx=dh)[:, 1] / n_avg

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

y = jnp.arange(y_max + dh/2, -y_max, -dh)
y = y[starty:stopy]
step = y.size // 50

r = jnp.sqrt(D**2 + y**2)
sin_theta = y / r
cos_theta = D / r
intensity_predicted = jnp.sinc(a / wavelength * sin_theta)**2 * cos_theta**2
intensity_predicted /= intensity_predicted.max()

ax.plot(y[::step] / wavelength, intensity[::step] / intensity.max(), '.', label='Simulated')
ax.plot(y / wavelength, intensity_predicted, label='Predicted')

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
