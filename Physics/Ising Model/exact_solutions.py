import numpy as np
from scipy.special import ellipk, ellipe

J = 1
k_B = 1
m = np.arcsinh(1)

def k(phi):
    return 2*np.sinh(phi)/np.cosh(phi)**2

def kp(phi):
    return 2*np.tanh(phi)**2 - 1

def energy_exact(t):
    phi = m/t
    return - J / np.tanh(phi) * (1 + 2/np.pi * kp(phi) * ellipk(k(phi)**2))

def heat_capacity_exact(t):
    phi = m/t
    return k_B * 2/np.pi * (phi/2)**2 / np.tanh(phi)**2 * (2*ellipk(k(phi)**2) - 2*ellipe(k(phi)**2) - (1 - kp(phi)) * (np.pi/2 + kp(phi)*ellipk(k(phi)**2)))

def plot_spontaneous_magnetization_exact(ax, label="Exact Solution"):
    tol = 1e-2
    t = np.concatenate([np.linspace(tol, 0.95, 1000, endpoint=False),
                         np.linspace(0.95, 1, 1000, endpoint=False)])
    s = (1 - np.sinh(np.arcsinh(1) / t)**(-4))**(1/8)
    t[[0, -1]] = [0, 1]
    s[[0, -1]] = [1, 0]
    ax.plot(t, s, marker='', linestyle='-', label=label)

def plot_average_energy_exact(ax, t_max):
    tol = 1e-2
    t = np.linspace(tol, t_max, 1000)
    u = energy_exact(t)
    t[0] = 0
    u[0] = u[1]
    ax.plot(t, u, label="Average Energy (exact)")

def plot_heat_capacity_exact(ax, t_max):
    tol1 = 1e-2
    tol2 = 1e-6
    if t_max > 1:
        t = np.concatenate([np.linspace(tol1, 1-tol2, 5000), np.linspace(1+tol2, t_max, 5000)])
    else:
        t = np.linspace(tol1, t_max, 10000)
    c = heat_capacity_exact(t)
    t[0] = 0
    c[0] = 0
    ax.plot(t, c, label="Heat Capacity (exact)")
