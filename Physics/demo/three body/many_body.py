import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pathlib

jax.config.update('jax_enable_x64', True)

G = 1

def calculate_acceleration(r, m, eps=1e-12):
    r_ij = r.reshape(-1, 1, 2) - r
    m_j = m.reshape(-1, 1)

    r_ij_squared = (r_ij**2 + eps**2).sum(axis=-1, keepdims=True)
    g_ij = - G * m_j * r_ij *  r_ij_squared**(-3/2)

    return g_ij.sum(axis=-2)

@jax.jit
def update_state(r, v, g, dt):
    v = v + 1/2 * g * dt
    r = r + v * dt
    g = calculate_acceleration(r, m)
    v = v + 1/2 * g * dt

    return r, v, g

def iterate(r_0, v_0, m, dt, n_iter, n_frames):
    r = r_0
    v = v_0
    g = calculate_acceleration(r, m)
    r_list = [r]
    step = max(n_iter // n_frames, 1)
    for iter in tqdm(range(n_iter)):
        r, v, g = update_state(r, v, g, dt)
        if iter % step == 0:
            r_list.append(r)

    return jnp.asarray(r_list)
# %%

# ---------------- initial conditions ----------------
M0, M1, M2 = 50.0, 1.0, 1e-3            # star >> planet >> moon (strong hierarchy)
m = jnp.array([M0, M1, M2])

a_p, e_p = 5.0, 0.4                       # planet: semi-major axis & eccentricity
v_apo = (G*M0/a_p * (1-e_p)/(1+e_p))**0.5  # speed at apoapsis
r1 = jnp.array([a_p*(1+e_p), 0.0]);  v1 = jnp.array([0.0, v_apo])

a_m = .5
v_m = (G*M1/a_m)**0.5                       # circular speed around the planet
r2 = r1 + jnp.array([a_m, 0.0]);  v2 = v1 + jnp.array([0.0, v_m])

v0 = -(M1*v1 + M2*v2)/M0                    # star velocity cancels net momentum

r_0 = jnp.stack([jnp.zeros(2), r1, r2])
v_0 = jnp.stack([v0, v1, v2])

T = float(2*jnp.pi*(a_p**3/(G*(M0+M1)))**0.5)
# -----------------------------------------------------

dt = 1e-7
n_iter = int(T/dt)
fps = 60
video_len = 10
n_frames = fps * video_len
r_list = iterate(r_0, v_0, m, dt, n_iter, n_frames)

for particle in range(r_list.shape[1]):
    plt.plot(r_list[:, particle, 0], r_list[:, particle, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-3.5, 8)
plt.ylim(-5, 5.5)
plt.savefig('three-body', dpi=600)
plt.clf()
# %%

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-3.5, 8)
ax.set_ylim(-5, 5.5)
lines = [ax.plot([], [], 'o')[0] for _ in range(r_list.shape[1])]

def animate(i):
    for j, line in enumerate(lines):
        x = jnp.atleast_1d(r_list[i, j, 0])
        y = jnp.atleast_1d(r_list[i, j, 1])
        line.set_data(x, y)
    return lines

ani = animation.FuncAnimation(fig, animate, frames=len(r_list), interval=1000/fps, blit=True)

writer = animation.FFMpegWriter(fps=fps, codec='libx264')
ani.save('animation.mkv', writer=writer, dpi=600)
