import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)

def iterate(v_0, g, t_f, dt):
    t = 0
    r = jnp.zeros((2,))
    v = v_0
    r_list = [r]
    while t < t_f:
        r += v * dt
        v += g * dt
        t += dt
        r_list.append(r)

    return jnp.array(r_list)

g = jnp.array([0, -9.8])
v_0 = jnp.array([10, 10])
t_f = - 2 * v_0[1] / g[1]
t = jnp.linspace(0, t_f, 1000).reshape(-1, 1)
r_exact = v_0.reshape(1, 2) * t + 1/2 * g.reshape(1, 2) * t**2

r_1 = iterate(v_0, g, t_f, dt=.2)
r_2 = iterate(v_0, g, t_f, dt=.1)
r_3 = iterate(v_0, g, t_f, dt=.05)

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.plot(r_exact[:, 0], r_exact[:, 1], label='Exact')
plt.plot(r_1[:, 0], r_1[:, 1], label='dt=.2s')
plt.plot(r_2[:, 0], r_2[:, 1], label='dt=.1s')
plt.plot(r_3[:, 0], r_3[:, 1], label='dt=.05s')
plt.legend()
plt.savefig('projectile_1.png', dpi=300)

r_4 = iterate(v_0, g, t_f, dt=.001)
plt.clf()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.plot(r_exact[:, 0], r_exact[:, 1], label='Exact')
plt.plot(r_4[:, 0][::100], r_4[:, 1][::100], '.', label='dt=.001s')
plt.legend()
plt.savefig('projectile_2.png', dpi=300)
