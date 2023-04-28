import torch
import numpy as np
from math import gamma
import matplotlib.pyplot as plt
from matplotlib import animation
# for better looking graphs
import scienceplots
plt.style.use(['science', 'notebook'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
float_dtype = torch.get_default_dtype()

# dimension of the box containing the gas particles
n_dim = 3
# number of particles
n_particles = int(1e4)
# length of the sides of the cubic box
l = 1
# rms speed
v_rms = 500
# radius of gas particles
radius = 5e-3
# time interval for simulation
dt = 1e-5

# number of times position and velocity of particles will be updated
n_iter = int(1e3)
# number of points used to plot MB distribution
n_points = int(1e3)
# max value of v for plotting
v_max_plot = 4*v_rms
# scaling factor for y axis when plotting the probability distribution
hist_factor = 1e3
# number of bins to use for histogram
n_bins = 80

# parameters for rendering the video
fps = 60
aspect_ratio = (16, 9)
dpi = 120

# array for storing position and velocity
r = torch.rand((n_dim, n_particles), device=device, dtype=float_dtype)*l
v = torch.zeros((n_dim, n_particles), device=device, dtype=float_dtype) 

# determining if the particle is on the left side of the box or the right side wrt x axis
is_r = r[0] > 0.5*l
is_l = r[0] <= 0.5*l
# make particles on the left side travel to the right and vice versa
v[0][is_r] = -v_rms
v[0][is_l] = v_rms
# store the array in ram
is_r = is_r.cpu()
is_l = is_l.cpu()

# calculate square of the distance between pairs of particle
def dist_sq_pairs(r, id_pairs):
    d_pairs = r[:, id_pairs[:, 0]] - r[:, id_pairs[:, 1]]

    return torch.sum(d_pairs**2, dim=0)


# update velocity
def compute_new_v(v1, v2, r1, r2):
    dr = r1 - r2
    v1_new = v1 - torch.sum((v1 - v2) * dr, dim=0) / torch.sum(dr**2, dim=0) * dr
    v2_new = v1 + v2 - v1_new

    return v1_new, v2_new


# reflect particles if the collide with any wall
def check_wall_collision(r, v, n_dim):
    for dim in range(0, n_dim):
        id_wall_collision = r[dim] < 0
        v[dim, id_wall_collision] = torch.abs(v[dim, id_wall_collision])
        id_wall_collision =  r[dim] > l
        v[dim, id_wall_collision] = -torch.abs(v[dim, id_wall_collision])


def motion(r, v, n_dim, radius, n_particles, n_iter, dt, device):
    # arrays for storing points for rendering the video
    rs = torch.zeros((n_iter, n_dim, n_particles),  dtype=float_dtype)
    vs = torch.zeros((n_iter, n_dim, n_particles),  dtype=float_dtype)
    rs[0] = r
    vs[0] = v

    # generate all pairs of particle
    id_pairs = torch.combinations(torch.arange(n_particles), 2).to(device)

    for i in range(1, n_iter):
        # check if the pairs are in collision
        id_pc = id_pairs[dist_sq_pairs(r, id_pairs) < (2*radius)**2]
        # update the velocity of colliding particles
        v[:, id_pc[:, 0]], v[:, id_pc[:, 1]] = compute_new_v(v[:, id_pc[:, 0]], v[:, id_pc[:, 1]],
                                                             r[:, id_pc[:, 0]], r[:, id_pc[:, 1]])
        check_wall_collision(r, v, n_dim)
        r = r + v*dt
        rs[i] = r
        vs[i] = v
    
    return rs, vs


rs, vs = motion(r, v, n_dim, radius, n_particles, n_iter, dt, device)


# plot the Maxwell Boltzmann distribution
v = np.linspace(0, v_max_plot, n_points)
a = (n_dim / 2)**(n_dim / 2) / v_rms**n_dim * 2 / gamma(n_dim / 2)
b = n_dim / 2 / v_rms**2
fv = a * v**(n_dim - 1) * np.exp(-b*v**2)
fv *= hist_factor

fig, ax = plt.subplots(1, 2, figsize=aspect_ratio)
ax[1].plot(v, fv, color='g', label="Maxwell-Boltzmann Distribution")

def init_fig():
    ax[0].set_xlim(0, l)
    ax[0].set_ylim(0, l)
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[1].set_xlim(0, v_max_plot)
    ax[1].set_ylim(0, 1.5*fv.max())
    ax[1].set_xlabel(r"Speed (m/s)")
    ax[1].set_ylabel(f"Probability Density ({hist_factor:.1e})")
    ax[1].legend()

    global red, blue, bins, patches
    # set the particle size shown in animation
    markersize = 1/4 * 2*radius/l * ax[0].get_window_extent().width * 72./fig.dpi
    red, = ax[0].plot([], [], marker='o', linestyle='', color='r', markersize=markersize)
    blue, = ax[0].plot([], [], marker='o', linestyle='', color='royalblue', markersize=markersize)

    n, bins, patches = ax[1].hist(torch.sqrt(torch.sum(vs[0]**2, dim=0)).cpu(), color='royalblue',
                                  bins=np.linspace(0, v_max_plot, n_bins), density=True)
    return red, blue


def animate(i):
    xred, yred = rs[i][0][is_r], rs[i][1][is_r]
    xblue, yblue = rs[i][0][is_l],rs[i][1][is_l]
    red.set_data(xred, yred)
    blue.set_data(xblue, yblue)
    hist, _ = np.histogram(torch.sqrt(torch.sum(vs[i]**2, dim=0)).cpu(), bins=bins, density=True)
    hist = hist * hist_factor
    for i, patch in enumerate(patches):
        patch.set_height(hist[i])
    return red, blue


ani = animation.FuncAnimation(fig, animate, init_func=init_fig, frames=n_iter,
                              interval=round(1000/fps), blit=True)
# Use vaapi encoder with ffmpeg
writer = animation.FFMpegWriter(fps=fps, codec="hevc_vaapi", extra_args=["-vaapi_device", "/dev/dri/renderD128", "-vf",
                                "format=nv12,hwupload"])
ani.save('animation.mkv', writer=writer, dpi=dpi)
