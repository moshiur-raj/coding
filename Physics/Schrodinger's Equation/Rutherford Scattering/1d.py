# This method exploits the spherical symmetry of Coulomb potential. The wavefunction has the form
# sum_l u_l(r, t)/r * P_l(cos_theta) and u_l(r, t) follows a one dimensional Schrodinger's equation
# with a l-dependent effective potential. Due to memory constraints large grid size could not be
# simulated, thus detector results at infinity is extrapolated from the fourier transform. Due to
# the usage of a narrow beam, measurements at small angles are not expected to reproduce the
# Rutherford cross section formula. Crank–Nicolson and Numerov were used for numerical methods. 

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from scipy.constants import hbar, eV, epsilon_0, e, m_u, pi
import matplotlib.pyplot as plt
from matplotlib import animation, colors

jax.config.update('jax_enable_x64', True)

E = 5e6 * eV
m = 4 * m_u
K = 2 * 79 * e**2 / (4 * pi * epsilon_0) # Coulomb's constant
r_n = 7e-15 # nuclear radius of target

v = np.sqrt(2 * E / m)
k = m * v / hbar
wlen = 2 * pi / k # wavelength
r_c = K / E # distance of closest approach

dr = wlen / 20
dt = dr / v

theta_min = 15 / 180 * pi
z0 = -100 * r_c
sigma_z = 10 * wlen
b_min = r_c / 2 / np.tan(theta_min / 2)
w = 4 * b_min
l_max = int(2 * k * w)
r_max = 1.25 * np.abs(z0)

n_plot = 1000
step = int(r_max / dr / n_plot)
theta_plot = jnp.linspace(0, pi, n_plot)
fps = 30
time = 10
n_frames = fps * time

def legendre_poly(l_max: int, x: jnp.ndarray) -> jnp.ndarray:
    def body(carry, l):
        p_prev, p_cur = carry
        return (p_cur, ((2 * l + 1) * x * p_cur - l * p_prev) / (l + 1)), p_cur
    return jax.lax.scan(body, (jnp.zeros_like(x), jnp.ones_like(x)), jnp.arange(l_max + 1))[1]

def initialize() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    r = jnp.arange(dr, r_max + dr/2, dr)
    cos_theta, w_theta = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(2 * l_max + 200))
    P = legendre_poly(l_max, cos_theta)
    Pw = P * w_theta[None]

    z = r[:, None] * cos_theta[None]
    rho = r[:, None] * jnp.sqrt((1 - cos_theta[None]**2))

    psi_0 = jnp.exp(- (z - z0)**2 / 4 / sigma_z**2 - (rho / w)**6 + 1j * k * z)
    norm = np.sqrt((2 * pi)**1.5 * sigma_z * w**2 * sp.special.gamma(1 / 3) / (6 * 2**(1 / 3)))
    psi_0 /= norm

    l = jnp.arange(l_max + 1)
    u = r[None] * (2 * l + 1)[:, None] / 2 * (Pw @ psi_0.T)

    V = jnp.where(r < r_n, K * (3 * r_n**2 - r**2) / (2 * r_n**3), K / r).reshape(1, -1)
    V_eff = V + hbar**2 / (2 * m) * l[:, None] * (l[:, None] + 1) / r**2

    return r, u, V_eff, norm

def prob_density_img(u: jnp.ndarray, r: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
    psi = (u / r[None, :]).T @ P
    psi2 = psi.real**2 + psi.imag**2

    n_rows = r.size // step
    psi2 = psi2[: n_rows * step].reshape(n_rows, step, n_plot).mean(axis=1)

    return psi2.astype(jnp.float32)

r, u, V_eff, norm = initialize()
r_plot = r[:(r.size // step) * step].reshape(-1, step).mean(axis=1)
P = legendre_poly(l_max, jnp.cos(theta_plot))
