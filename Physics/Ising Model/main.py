## Simulation of Ising model using the metropolis algorithm. The pseudo-code for metropolis
## algorithm can be found in the book An Introduction to Thermal Physics by Daniel V. Schroeder,
## chapter 8.

import numpy as np
import ctypes
import os

import ising
import plots

# Enable to plot avg spin, avg energy, heat capacity for the first value of h_norm.
plt_thermodynamic_functions = True
# Enable to plot avg spin, avg energy for different values of h_norm.
plt_thermodynamic_functions_for_different_h = True
# Enable to plot the magnetic domains.
plt_domains = True
# Enable to plot the convergence of the algorithm for infinite iterations.
plt_conv_inf_iter = True
# Enable to plot the convergence of the algorithm for infinite lattice.
plt_conv_inf_lattice = True
# Enable to animate the state transitions in metropolis algorithm.
anim_state_transitions = True
# Enable for frameon=True for matplotlib figures. Default should be True (assuming you want to
# generate standalone images) unless the figures are going to be embedded in a pdf or other types of
# document.
frameon = True
# Enable to save data in .npz format after execution
save_data = True

# Critical temperacture divided by J/k_B.
t_c = 2/np.arcsinh(1)
# Temperatures for plotting magnetic domains.
# Temperatures in this program are normalized such that, beta * absolute temperature = 1/t.
t_domain = np.array([0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.50])*t_c
# Temperatures for plotting thermodynamic functions. These are derived from t_domain.
t = np.concatenate([np.linspace(t_domain[i], t_domain[i+1], 25, endpoint=False,
                                dtype=ctypes.c_double) for i in range(0, t_domain.size - 1)])
# Append the last value of t_domain.
t = np.append(t, t_domain[-1])

# Avg spin of the initial lattice. It seems this needs to be positive for the algorithm to work.
# Otherwise the algorithm does not converge. Most likely due to the negative spin states being
# equally probable.
s_0 = 0.20
# Normalized magnetic field strength. The first value is used for plotting magnetic domains.
h_norm = np.array([0, 0.025, 0.05], dtype=ctypes.c_double)
# Length of the square lattice.
length = 250
# Parameter used for determining the number of iterations. The algorithm chooses a random lattice
# point in each iteration. The random point is determined by a random number stored in an array.
# The size of that array is number of lattice points * iteration multiplier. You can think of
# iteration_multiplier indicating the number of iternations per lattice point.
iteration_multiplier = 500
# The algorithm is iterated n_pass * iteration_multiplier * number of lattice points number of times
# in total. n_pass is used to avoid using excess memory by the array used to store random variables.
# Basically the array is used n_pass times.
n_pass = 1
# Frames per second for the animation.
fps = 60
# Total number of frames for the animation.
n_frames = fps*30
# Temperature of the lattice for animation.
t_anim = 0.90*t_c

# Number of iteration is modified by multiplication with elements of this array.
# Must be in decreasing order due to the array storing random variables being generated for
# h_norm[0]. Otherwise there will be buffer overflow.
n_iter_mult = np.array([1/4, 1/16], dtype=ctypes.c_double)
# Lattice size is modified by multiplication with elements of this array.
# Must be in decreasing order due to the array storing random variables being generated for
# h_norm[0]. Otherwise there will be buffer overflow.
lattice_size_mult = np.array([1/4, 1/16], dtype=ctypes.c_double)

# Parameters to pass to C function as argument
param = ising.parameters()
param.flag.plt_thermodynamic_functions = plt_thermodynamic_functions
param.flag.plt_thermodynamic_functions_for_different_h = plt_thermodynamic_functions_for_different_h
param.flag.plt_domains = plt_domains 
param.flag.plt_conv_inf_iter = plt_conv_inf_iter 
param.flag.plt_conv_inf_lattice = plt_conv_inf_lattice 
param.flag.anim_state_transitions = anim_state_transitions 
param.s_0 = s_0
param.t_anim = t_anim
param.n_frames = n_frames
# n_var indicates the size of var.
# Number of threads to use for the algorithm. By default it is equal to number of cpu threads.
param.n_threads = os.cpu_count()
param.n_h_norm = h_norm.size
param.n_t_domain = t_domain.size
param.n_t = t.size
param.n_rows = length
param.n_cols = length
param.n_spins = param.n_rows * param.n_cols
param.n_pass = n_pass
param.n_iter_0 = iteration_multiplier * param.n_spins
param.n_n_iter_mult = n_iter_mult.size
param.n_lattice_size_mult = lattice_size_mult.size

# Initialize arrays to pass to c library.
# To store magnetic domains.
domain = np.zeros((param.n_t_domain + 1, param.n_rows, param.n_cols), dtype=ising.spin_dtype)
# To generate video frames for animation. Allocate a dummy variable if animation is disabled as
# lattice_anim can use up a lot of memory
if anim_state_transitions:
    lattice_anim = np.zeros((param.n_frames, param.n_rows, param.n_cols), dtype=ising.spin_dtype)
else:
    lattice_anim = np.zeros(1, dtype=ising.spin_dtype)
# To store the avg spin of the final state
s_f = np.zeros((h_norm.size, param.n_t), dtype=ctypes.c_double)
# To store the ensemble average of avg spin
s_avg = np.zeros((h_norm.size, param.n_t), dtype=ctypes.c_double)
# To store the avg energy of final state
# Energies and Heat capacities are also normalized by dividing with J and k_B respectively.
u_f = np.zeros((h_norm.size, param.n_t), dtype=ctypes.c_double)
# To store the avg spin of the final state for different number of iternations
s_f_i = np.zeros((n_iter_mult.size, param.n_t), dtype=ctypes.c_double)
# To store the avg spin of the final state for different lattice sizes
s_f_l = np.zeros((lattice_size_mult.size, param.n_t), dtype=ctypes.c_double)

# Pointers to pass to the C function as argument. This allows the C library to access the numpy
# arrays.
ptr = ising.pointers()
ptr.domain = ising.get_pointer(domain)
ptr.lattice_anim = ising.get_pointer(lattice_anim)
ptr.h_norm = ising.get_pointer(h_norm)
ptr.t_domain = ising.get_pointer(t_domain)
ptr.t = ising.get_pointer(t)
ptr.s_f = ising.get_pointer(s_f)
ptr.s_avg = ising.get_pointer(s_avg)
ptr.u_f = ising.get_pointer(u_f)
ptr.n_iter_mult = ising.get_pointer(n_iter_mult)
ptr.lattice_size_mult = ising.get_pointer(lattice_size_mult)
ptr.s_f_i = ising.get_pointer(s_f_i)
ptr.s_f_l = ising.get_pointer(s_f_l)

# Pass the arguments to the C function and start the algorithm
ising.start_simulation(param, ptr)

# Check if figures directory exists
if not os.path.exists("./figures"):
    os.mkdir("./figures")

# All the functions below take normalized temperature as argument. So, use t/t_c insread of t
# Plot thermodynamic functions
if plt_thermodynamic_functions:
    plots.plot_thermodynamic_functions(t/t_c, s_f[0], s_avg[0], u_f[0], frameon)

# Plot thermodynamic functions at different field strengths
if plt_thermodynamic_functions_for_different_h:
    plots.plot_thermodynamic_functions_for_different_h(h_norm, t/t_c, s_f, u_f, frameon)

# Plot magnetic domains
if plt_domains:
    plots.plot_magnetic_domains(t_domain/t_c, domain, frameon)

# Animate state transitions
if anim_state_transitions:
    res_width = 1920
    iter = np.append(np.arange(0, n_pass*param.n_iter_0,
                               int(n_pass*param.n_iter_0/(param.n_frames-1) + 1), dtype=int),
                     param.n_iter_0);
    plots.create_animation(lattice_anim, param.t_anim/t_c, iter, fps, res_width)

# Plot the convergence of the algorithm for increasing iterations
if plt_conv_inf_iter:
    s_f_ = np.vstack([s_f[0], s_f_i])
    n_iter_mult_ = np.concatenate([[1], n_iter_mult])
    plots.plot_convergence_iterations(t/t_c, s_f_, n_iter_mult_*param.n_pass*param.n_iter_0,
                                      frameon)

# Plot the convergence of the algorithm for increasing lattice size
if plt_conv_inf_lattice:
    s_f_ = np.vstack([s_f[0], s_f_l])
    lattice_size_mult_ = np.concatenate([[1], lattice_size_mult])
    plots.plot_convergence_lattice_size(t/t_c, s_f_, lattice_size_mult_, param.n_rows, param.n_cols,
                                        frameon)

if save_data:
    np.savez("data", t_domain=t_domain, domain=domain, t=t, s_f=s_f, s_avg=s_avg, u_f=u_f,
             s_f_i=s_f_i, s_f_l=s_f_l, h_norm=h_norm, iter=n_pass*param.n_iter_0*n_iter_mult,
             len=length*lattice_size_mult)
