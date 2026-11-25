# This method exploits the spherical symmetry of Coulomb potential. The wavefunction has the form
# sum_l u_l(r, t)/r * P_l(cos_theta) and u_l(r, t) follows a one dimensional Schrodinger's equation
# with a l-dependent effective potential. Due to memory constraints large grid size could not be
# simulated, thus detector results at infinity is extrapolated from the fourier transform. Due to
# the usage of a narrow beam, measurements at small angles are not expected to reproduce the
# Rutherford cross section formula. Crank–Nicolson and Numerov were used for numerical methods. 

import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from scipy.constants import hbar, eV, epsilon_0, e, m_u, pi

from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib import animation, colors

jax.config.update('jax_enable_x64', True)

m = 4 * m_u
K = 2 * 79 * e**2 / (4 * pi * epsilon_0) # Coulomb's constant
E = 5e6 * eV
r_n = 7e-15 # nuclear radius of target

v = np.sqrt(2 * E / m)
k = m * v / hbar
wlen = 2 * pi / k # wavelength
r_c = K / E # distance of closest approach

dr = wlen / 20
dt = dr / v

theta_min = 15 / 180 * pi # cross section should be accurate for theta > theta_min
z0 = -100 * r_c
k0 = np.sqrt(2 * m * (E - K / np.abs(z0))) / hbar
sigma_z = 10 * wlen
b_min = r_c / 2 / np.tan(theta_min / 2)
w = 4 * b_min
l_max = int(1.75 * k * w)
r_max = 1.25 * np.abs(z0)

n_plot = 1000
fps = 30
time = 20
n_frames = fps * time
steps_per_frame = int((0.9 * (r_max + np.abs(z0)) / v / dt) / n_frames)

# Numerov / Crank-Nicolson constants:  i hbar M du/dt = A u,  M = tridiag(1, 10, 1)/12,
# A = -hbar^2/(2 m dr^2) tridiag(1, -2, 1) + M diag(V_eff).  (M + c A) u_new = (M - c A) u
M0, M1 = 10 / 12, 1 / 12
T0 = hbar**2 / (2 * m * dr**2)
c = 0.5j * dt / hbar

def initialize() -> tuple[jnp.ndarray, ...]:
    r = jnp.arange(dr, r_max + dr/2, dr)
    cos_theta, w_theta = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(2 * l_max + 200))
    P = sp.special.legendre_p_all(l_max, np.array(cos_theta))[0]
    Pw = jnp.array(P * w_theta[None])

    z = r[:, None] * cos_theta[None]
    rho = r[:, None] * jnp.sqrt((1 - cos_theta[None]**2))

    psi_0 = jnp.exp(- (z - z0)**2 / 4 / sigma_z**2 - (rho / w)**6 + 1j * k0 * z)
    norm = np.sqrt((2 * pi)**1.5 * sigma_z * w**2 * sp.special.gamma(1 / 3) / (6 * 2**(1 / 3)))
    psi_0 /= norm

    l = jnp.arange(l_max + 1)
    u = r[None] * (2 * l + 1)[:, None] / 2 * (Pw @ psi_0.T)

    V = jnp.where(r < r_n, K * (3 * r_n**2 - r**2) / (2 * r_n**3), K / r).reshape(1, -1)
    V_eff = V + hbar**2 / (2 * m) * l[:, None] * (l[:, None] + 1) / r**2

    return r, u, V_eff

def crank_nicolson_numerov_operators(V_eff: jnp.ndarray) -> dict:
    n = V_eff.shape[-1]
    V_pad = jnp.pad(V_eff, ((0, 0), (1, 1)))
    i = jnp.arange(n)[None, :]

    return dict(
            V_pad=V_pad,
            diag=M0 + c * (2 * T0 + M0 * V_eff),
            upper=jnp.where(i != n - 1, M1 + c * (-T0 + M1 * V_pad[:, 2:]) , 0),
            lower=jnp.where(i != 0    , M1 + c * (-T0 + M1 * V_pad[:, :-2]), 0)
            )

# requires u to be zero at boundaries
def crank_nicolson_numerov_step(u: jnp.ndarray, ops: dict) -> jnp.ndarray:
    u_pad = jnp.pad(u, ((0, 0), (1, 1)))
    Vu = ops["V_pad"] * u_pad
    Mu = M1 * (u_pad[:, 2:] + u_pad[:, :-2]) + M0 * u
    Au = -T0 * (u_pad[:, 2:] - 2 * u + u_pad[:, :-2]) + M1 * (Vu[:, 2:] + Vu[:, :-2]) \
         + M0 * Vu[:, 1:-1]
    rhs = Mu - c * Au

    return jax.lax.linalg.tridiagonal_solve(
            ops["lower"], ops["diag"], ops["upper"], rhs[..., None]
            )[..., 0]

@jax.jit
def next_frame(u: jnp.ndarray, ops: dict) -> jnp.ndarray:
    return jax.lax.scan(
            lambda u, _: (crank_nicolson_numerov_step(u, ops), None),
            u, None, length=steps_per_frame
            )[0]

@jax.jit(static_argnames=('log_prob_min'))
def log_prob_density(
        u: jnp.ndarray, r: jnp.ndarray, P: jnp.ndarray, ref: jnp.ndarray, log_prob_min: float = -12,
        ) -> jnp.ndarray:
    psi = (u / r[None, :]).T @ P
    psi2 = psi.real**2 + psi.imag**2

    # smoothen the plot by taking average
    r_step = r.size // n_plot
    n_rows = r.size // r_step
    psi2 = psi2[: n_rows * r_step].reshape(n_rows, r_step, -1).mean(axis=1)
    psi2 = psi2.reshape(n_rows, n_plot, -1).mean(axis=-1)

    log_prob = (jnp.log10(psi2) - ref).astype(jnp.float32)

    return jnp.maximum(log_prob, log_prob_min)

def cross_section(u: jnp.ndarray | np.ndarray, r: jnp.ndarray | np.ndarray) -> tuple[np.ndarray, ...]:
    u = np.array(u)
    r = np.array(r)

    theta_k = np.linspace(0, pi, n_plot)
    P = sp.special.legendre_p_all(l_max, np.cos(theta_k))[0]
    sigma_k = 1 / 2 / sigma_z
    # particle loses energy near nucleus so lower k should be prioritized more
    k_list, dk = np.linspace(k-10*sigma_k, k+5*sigma_k, max(os.cpu_count()*3, 25), retstep=True)
    l = np.arange(l_max + 1)

    ur = u * r[None]
    def get_int_jur(k_):
        return np.trapezoid(sp.special.spherical_jn(l[:, None], k_ * r[None]) * ur, dx=dr)

    with ThreadPoolExecutor(os.cpu_count()) as ex:
        cols = list(tqdm(ex.map(get_int_jur, k_list), total=k_list.size, desc="momentum distribution"))
    int_jur = np.stack(cols, axis=1)

    phase = np.array([1, -1j, -1, 1j])[l % 4] # (-i)^l
    phi = 4 * pi * (phase[:, None] * P).T @ int_jur
    phi2 = phi.real**2 + phi.imag**2

    fluence = 2**(1 / 3) / (pi * w**2 * sp.special.gamma(4 / 3)) # calculated at t=0. not so accurate
    cs = 1/(2*pi)**3 * np.trapezoid(phi2 * k_list**2, dx=dk) / fluence

    return cs, theta_k

def simulate() -> tuple[list[np.ndarray], *tuple[np.ndarray, ...]]:
    r, u, V_eff = initialize()
    theta = np.linspace(0, pi, 3 * n_plot)
    P = jnp.array(sp.special.legendre_p_all(l_max, np.cos(theta))[0])
    ops = crank_nicolson_numerov_operators(V_eff)

    ref = jnp.max(log_prob_density(u, r, P, 0))
    frames = [np.array(log_prob_density(u, r, P, ref))]
    for _ in tqdm(range(n_frames), desc="simulation"):
        u = next_frame(u, ops)
        frames.append(np.array(
            log_prob_density(u, r, P, ref)
            ))

    r_step = r.size // n_plot
    r_img = r[:(r.size // r_step) * r_step].reshape(-1, r_step).mean(axis=1)
    theta_step = theta.size // n_plot
    theta_img = theta[:(theta.size//theta_step) * theta_step].reshape(-1, theta_step).mean(axis=1)

    cs, theta_k = cross_section(u, r)

    return frames, cs, np.array(r_img), theta_img, theta_k

def rutherford(theta: np.ndarray) -> np.ndarray:
    return (r_c / 4)**2 / np.sin(theta / 2)**4

def plot_cross_section(cs: np.ndarray, theta_k: np.ndarray) -> None:
    barn = 1e-28
    deg = np.degrees(theta_k)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax0.semilogy(deg[1:], rutherford(theta_k[1:]) / barn, "k-", label="Rutherford Formula")
    ax0.semilogy(deg, cs / barn, ".", label="Simulation", markevery=25,)
    ax1.plot(deg[1:], cs[1:] / rutherford(theta_k[1:]), "-")
    for ax in (ax0, ax1):
        ax.axvline(np.degrees(theta_min), color="gray", lw=0.8, ls=":")
    ax0.annotate("angles properly covered by beam width", xy=(np.degrees(theta_min), 0.04),
                 xycoords=("data", "axes fraction"), xytext=(np.degrees(theta_min) + 12, 0.04),
                 textcoords=("data", "axes fraction"),
                 va="center", fontsize=9, color="gray",
                 arrowprops=dict(arrowstyle="<-", color="gray", lw=0.8))
    ax1.axhline(1, color="k", lw=0.8)
    ax1.set_ylim(0.9, 1.1)
    ax0.set_ylabel(r"$\frac{d\sigma}{d\Omega}$ (b/sr)")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel(r"$\theta$ (deg)")
    ax0.legend()
    fig.tight_layout()
    fig.savefig("cross_section.pdf")
    plt.close(fig)

def animate(frames: list[np.ndarray], r_img: np.ndarray, theta_img: np.ndarray) -> None:
    view = 1.03 * r_max
    pix = view / 600
    x = np.arange(-view, view, pix) + pix / 2
    y = np.arange(pix / 2, view, pix)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # (row, column) of the polar image for every Cartesian pixel; outside r_max -> below the colour scale
    coords = np.array([np.interp(np.hypot(X, Y), r_img, np.arange(r_img.size), right=-10),
                       np.interp(np.arctan2(Y, X), theta_img, np.arange(theta_img.size))])

    def to_cartesian(frame: np.ndarray) -> np.ndarray:
        half = 10.0 ** map_coordinates(frame.astype(np.float32), coords, order=1, cval=-12)
        return np.concatenate([half[:, ::-1], half], axis=1).T # mirror to rho < 0

    cmap = plt.get_cmap("magma").copy()
    cmap.set_under("black")
    fig, ax = plt.subplots(figsize=(8, 7.4))
    im = ax.imshow(to_cartesian(frames[0]), origin="lower", cmap=cmap,
                   extent=(-view / wlen, view / wlen, -view / wlen, view / wlen),
                   norm=colors.LogNorm(vmin=1e-7, vmax=1))
    phi = np.linspace(0, 2 * pi, 400)
    ax.plot(r_max / wlen * np.cos(phi), r_max / wlen * np.sin(phi), "w--", lw=0.8, label="boundary")
    ax.set_xlabel(r"$z / \lambda$")
    ax.set_ylabel(r"$\rho / \lambda$")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")
    title = ax.set_title("")
    fig.tight_layout()

    def update(i: int):
        im.set_data(to_cartesian(frames[i]))
        title.set_text(rf"$t = {i * steps_per_frame * dt * 1e21:.1f} \times 10^{{-21}}$ s")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset",
                                                "medium"])
    ani.save("rutherford.mp4", writer=writer, dpi=300)
    plt.close(fig)

frames, cs, r_img, theta_img, theta_k = simulate()
plot_cross_section(cs, theta_k)
animate(frames, r_img, theta_img)
