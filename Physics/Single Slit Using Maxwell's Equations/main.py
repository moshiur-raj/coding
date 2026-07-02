import jax
import jax.numpy as jnp
from scipy.constants import c, epsilon_0, mu_0
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

jax.config.update('jax_enable_x64', True)

def apply_boundary_cond(
        Ex: jnp.ndarray,
        Ey: jnp.ndarray,
        Bz: jnp.ndarray,
        dBz: jnp.ndarray,
        boundary: tuple[jnp.ndarray, int],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    rows, col = boundary
    Ex  = Ex.at[rows, col].set(0)
    Ey  = Ey.at[rows, col].set(0)
    Bz  = Bz.at[rows, col].set(0)
    dBz = dBz.at[rows, col].set(0)

    return Ex, Ey, Bz, dBz

def gen_initial_cond(
        D: float,
        l: float,
        dx: float,
        dy: float,
        wavelength: float,
        slit_pos_rel: float ,
        a: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, int]]:

    x = jnp.arange(0, D + dx/2, dx)
    y = jnp.arange(0, l + dy/2, dy)
    
    slit_index_x = int((x.size - 1) * slit_pos_rel)
    n = int( (y.size - a / dy - 1) / 2 )
    slit_index_y = jnp.concatenate([jnp.arange(n), jnp.arange(y.size - n, y.size)])
    boundary = (slit_index_y, slit_index_x)

    x, y = jnp.meshgrid(x, y)
    k = 2 * jnp.pi / wavelength
    Ey = jnp.sin(k * x)
    Bz = Ey / c

    cutoff_index = int( (0.75 * slit_pos_rel * D // wavelength) * wavelength / dx )
    Ey = Ey.at[:, cutoff_index:].set(0)
    Bz = Bz.at[:, cutoff_index:].set(0)

    Ex = jnp.zeros(Ey.shape)

    return Ex, Ey, Bz, boundary

def ddx(x: jnp.ndarray, dx: float) -> jnp.ndarray:
    x_padded = jnp.pad(x, ((0, 0), (1, 1)))
    delta_x = x_padded[:, 2:] - x_padded[:, :-2]

    return delta_x / (2 * dx)

def ddy(x: jnp.ndarray, dy: float) -> jnp.ndarray:
    x_padded = jnp.pad(x, ((1, 1), (0, 0)))
    delta_x = x_padded[2:] - x_padded[:-2]

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
        boundary: tuple[jnp.ndarray, int],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    Bz += dBz
    Ex += c**2 * ddy(Bz, dy) * dt
    Ey += - c**2 * ddx(Bz, dx) * dt
    dBz = (dt / 2) * (ddy(Ex, dy) - ddx(Ey, dx))
    Bz += dBz

    return apply_boundary_cond(Ex, Ey, Bz, dBz, boundary)

def crop(
        Ex: jnp.ndarray,
        Ey: jnp.ndarray,
        Bz: jnp.ndarray,
        nx: int,
        ny: int,
        ratio: float,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    nrows, ncols = Ex.shape
    startx = int(ncols * ratio)
    stopx = ncols - startx
    starty = int(nrows * ratio)
    stopy = nrows - starty

    Ex = Ex[starty:stopy:ny, startx:stopx:nx]
    Ey = Ey[starty:stopy:ny, startx:stopx:nx]
    Bz = Bz[starty:stopy:ny, startx:stopx:nx]

    return Ex, Ey, Bz

@jax.jit
def poynting_vector(
        Ex: jnp.ndarray,
        Ey: jnp.ndarray,
        Bz: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:

    Sx = 1 / mu_0 * Ey * Bz
    Sy = - 1 / mu_0 * Ex * Bz

    return Sx, Sy

@jax.jit
def energy_density(Ex: jnp.ndarray, Ey: jnp.ndarray, Bz: jnp.ndarray) -> jnp.ndarray:
    u = epsilon_0/2 * (Ex**2 + Ey**2) + 1/(2*mu_0) * Bz**2
    return u

wavelength = 1e-6
D = 2 * 50 * wavelength
dh = wavelength / 20
l = 2 * 50 * wavelength
a = 3 * wavelength
dt = dh / c
slit_pos_rel = .5
n_iter = int(slit_pos_rel * D / c / dt)

video_len = 10
fps = 60
n_frames = video_len * fps
step = max(n_iter // n_frames, 1)
poynting_vector_list = []
energy_density_list = []

Ex, Ey, Bz, boundary = gen_initial_cond(D=D, l=l, dx=dh, dy=dh, wavelength=wavelength,
                                        slit_pos_rel=slit_pos_rel, a=a)
dBz = (dt / 2) * ((ddy(Ex, dy=dh) - ddx(Ey, dx=dh)))

density = 1000
ny, nx = Ex.shape 
ny = max(ny // density, 1)
nx = max(nx // density, 1)

for i in tqdm(range(n_iter)):
    Ex, Ey, Bz, dBz = iterate(Ex=Ex, Ey=Ey, Bz=Bz, dBz=dBz, dx=dh, dy=dh, dt=dt, boundary=boundary)
    if i % step == 0:
        Ex_, Ey_, Bz_ = crop(Ex, Ey, Bz, nx, ny, ratio=.25)
        # poynting_vector_list.append(poynting_vector(Ex_, Ey_, Bz_))
        energy_density_list.append(energy_density(Ex_, Ey_, Bz_))

energy_density_list = jnp.array(energy_density_list)
# %% -----------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

u = energy_density_list[0]
im = ax.imshow(u / u.max(), cmap='Oranges', vmin=0, vmax=1)

def animate(u):
    im.set_data(u / u.max())
    return im,

ani = animation.FuncAnimation(fig, animate, frames=energy_density_list, interval=1000/fps, blit=True)

writer = animation.FFMpegWriter(fps=fps, codec='libx264')
# writer = animation.FFMpegWriter(fps=fps, codec="hevc_vaapi", extra_args=["-vaapi_device",
#                                                                          "/dev/dri/renderD128",
#                                                                          "-vf",
#                                                                          "format=nv12,hwupload"])

ani.save('animation.mkv', writer=writer, dpi=600)
plt.clf()
# %% -----------------------------------------------------------------------------------------------

n_avg = int((wavelength / c) / dt)

fig, ax = plt.subplots()
intensity_simulated = energy_density_list[-n_avg:, :, -1].mean(axis=0)
ax.plot(intensity_simulated)
plt.show()
plt.clf()
