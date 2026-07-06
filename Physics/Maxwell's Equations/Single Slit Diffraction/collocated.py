import jax
import jax.numpy as jnp
from scipy.constants import c, epsilon_0, mu_0
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation, colors

jax.config.update('jax_enable_x64', True)

def apply_electric_boundary_cond(
        Ex: jnp.ndarray,
        Ey: jnp.ndarray,
        slit_inner: tuple[jnp.ndarray, int],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:

    rows, col = slit_inner

    Ex = Ex.at[rows, col].set(0)
    Ex = Ex.at[0, :].set(0)
    Ex = Ex.at[-1, :].set(0)

    Ey = Ey.at[rows, col].set(0)
    Ey = Ey.at[rows, col - 1].set(0)
    Ey = Ey.at[rows, col + 1].set(0)
    Ey = Ey.at[:, 0].set(0)
    Ey = Ey.at[:, -1].set(0)

    return Ex, Ey

def apply_magnetic_boundary_cond(
        Bz: jnp.ndarray,
        slit_inner: tuple[jnp.ndarray, int],
        ) -> jnp.ndarray:

    rows, col = slit_inner

    Bz = Bz.at[rows, col].set(0)

    return Bz


def gen_initial_cond(
        D: float,
        y_max: float,
        dx: float,
        dy: float,
        wavelength: float,
        a: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, int]]:

    x = jnp.arange(-2 * D, 1.5 * D + dx/2, dx)
    y = jnp.arange(y_max, -y_max - dy/2, -dy)
    
    slit_index_x = int((x.size - 1) * 4/7)
    n = int( (y.size - a / dy - 1) / 2 )
    slit_index_y = jnp.concatenate([jnp.arange(n), jnp.arange(y.size - n, y.size)])
    slit_inner = (slit_index_y, slit_index_x)

    x, y = jnp.meshgrid(x, y)
    k = 2 * jnp.pi / wavelength
    Ey = jnp.sin(k * x)
    Bz = Ey / c

    cutoff_index = int( (1.5 * D // wavelength) * wavelength / dx )
    Ey = Ey.at[:, cutoff_index:].set(0)
    Bz = Bz.at[:, cutoff_index:].set(0)

    Ex = jnp.zeros(Ey.shape)

    return Ex, Ey, Bz, slit_inner

def ddx(x: jnp.ndarray, dx: float) -> jnp.ndarray:
    x_padded = jnp.pad(x, ((0, 0), (1, 1)))
    delta_x = x_padded[:, 2:] - x_padded[:, :-2]

    return delta_x / (2 * dx)

def ddy(x: jnp.ndarray, dy: float) -> jnp.ndarray:
    x_padded = jnp.pad(x, ((1, 1), (0, 0)))
    delta_x = x_padded[:-2] - x_padded[2:]

    return delta_x / (2 * dy)

@jax.jit(donate_argnames=("Ex", "Ey", "Bz", "dBz"))
def iterate(
        Ex: jnp.ndarray,
        Ey: jnp.ndarray,
        Bz: jnp.ndarray,
        dBz: jnp.ndarray,
        dx: float,
        dy: float,
        dt: float,
        slit_inner: tuple[jnp.ndarray, int],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    Bz += dBz
    Ex = Ex + c**2 * ddy(Bz, dy) * dt
    Ey = Ey - c**2 * ddx(Bz, dx) * dt
    Ex, Ey = apply_electric_boundary_cond(Ex, Ey, slit_inner)
    dBz = apply_magnetic_boundary_cond(dt / 2 * (ddy(Ex, dy) - ddx(Ey, dx)), slit_inner)
    Bz += dBz

    return Ex, Ey, Bz, dBz

@jax.jit
def calc_poynting_vector(
        Ex: jnp.ndarray,
        Ey: jnp.ndarray,
        Bz: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:

    Sx = 1 / mu_0 * Ey * Bz
    Sy = - 1 / mu_0 * Ex * Bz

    return Sx, Sy

@jax.jit
def calc_energy_density(Ex: jnp.ndarray, Ey: jnp.ndarray, Bz: jnp.ndarray) -> jnp.ndarray:
    u = epsilon_0/2 * (Ex**2 + Ey**2) + 1/(2*mu_0) * Bz**2
    return u

@jax.jit
def calc_intensity(Ey: jnp.ndarray, Bz: jnp.ndarray) -> jnp.ndarray:
    return 1 / mu_0 * Ey * Bz

wavelength = 1e-6
a = 3 * wavelength
D = 25 * wavelength
dh = wavelength / 50
y_max = (1.5**2 - 1)**.5 * D
dt = dh / c / 10
n_iter = int(2 * D / c / dt)

video_len = 10
fps = 60
n_frames = video_len * fps
step = max(n_iter // n_frames, 1)
poynting_vector_list = []
energy_density_list = []

Ex, Ey, Bz, slit_inner = gen_initial_cond(D=D, y_max=y_max, dx=dh, dy=dh, wavelength=wavelength, a=a)
dBz = apply_magnetic_boundary_cond(dt / 2 * (ddy(Ex, dy=dh) - ddx(Ey, dx=dh)), slit_inner)

density = 1000
ny, nx = Ex.shape 
ny = max(ny // density, 1)
nx = max(nx // density, 1)
nrows, ncols = Ex.shape
startx = int(ncols * 2/7)
stopx = ncols - int(ncols * 1/7)
starty = 0
stopy = nrows - starty

n_avg = int(10 * wavelength / c / dt)
intensity = jnp.zeros(Ex.shape[0])
for i in tqdm(range(n_iter)):
    Ex, Ey, Bz, dBz = iterate(Ex=Ex, Ey=Ey, Bz=Bz, dBz=dBz, dx=dh, dy=dh, dt=dt, slit_inner=slit_inner)
    if i % step == 0:
        Ex_ = Ex[starty:stopy:ny, startx:stopx:nx]
        Ey_ = Ey[starty:stopy:ny, startx:stopx:nx]
        Bz_ = Bz[starty:stopy:ny, startx:stopx:nx]
        # poynting_vector_list.append(poynting_vector(Ex_, Ey_, Bz_))
        energy_density_list.append(calc_energy_density(Ex_, Ey_, Bz_))
    if i >= n_iter - n_avg:
        intensity += 1/n_avg * calc_intensity(Ey[starty:stopy, stopx], Bz[starty:stopy, stopx])

jnp.save('intensity', intensity)
# jnp.save('energy_density_list', energy_density_list)

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

ax.plot(y[::plot_step] / wavelength, intensity[::plot_step] / intensity.max(), '.', label='Simulated')
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

u_ref = max(float(u.max()) for u in energy_density_list)
u = energy_density_list[0]
im = ax.imshow( u / u_ref,
               extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
               norm=colors.PowerNorm(gamma=0.3, vmin=0, vmax=1))

def animate(u):
    im.set_data(u / u_ref)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=energy_density_list, interval=1000/fps, blit=True)

ani.save('animation1.mkv', writer=writer, dpi=600)
# %% -----------------------------------------------------------------------------------------------

plt.close()
fig, ax = plt.subplots()
ax.set_xlim(0, D / wavelength)
ax.set_ylim(-y_max / wavelength, y_max / wavelength)
ax.set_xlabel(r'$x \, / \, \lambda$')
ax.set_ylabel(r'$y \, / \, \lambda$')

cutoff_t = int(D / 2 / c / dt / step)
u_ref = max(float(u.max()) for u in energy_density_list[cutoff_t:])
u = energy_density_list[cutoff_t]
cutoff_x = u.shape[1] // 2
u = u[:, cutoff_x:]
im = ax.imshow(u / u_ref, extent=(*ax.get_xlim(), *ax.get_ylim()), cmap='Oranges',
               norm=colors.LogNorm(vmin=1e-4, vmax=1))

def animate(u):
    u = u[:, cutoff_x:]
    im.set_data(u / u_ref)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=energy_density_list[cutoff_t: ],
                              interval=1000/fps, blit=True)

ani.save('animation2.mkv', writer=writer, dpi=600)
