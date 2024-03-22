import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit, fsolve
from scipy.misc import derivative
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# EXPERIMENTAL DATA

# Drain Characteristics
# in Volts
V_GS = [0.00, -0.05, -0.10, -0.15]
# in mA
I_D = [ np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                  15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 21.5, 22.0, 22.5]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                  15.0, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.5,
                  14.0, 14.5, 15.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5,
                  12.0])
]
# Convert current to Amperes
I_D = [ i*1e-3 for i in I_D]
# in Volts
V_DS = [ np.array([0.000, 0.020, 0.038, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.180, 0.210,
                   0.240, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.70, 1.10, 2.20, 4.80, 6.60, 9.4,
                   13.0]),
         np.array([0.000, 0.020, 0.038, 0.062, 0.080, 0.100, 0.120, 0.155, 0.180, 0.240, 0.250,
                   0.30, 0.35, 0.40, 0.50, 0.75, 2.0, 3.4, 5.0, 8.6, 10.0, 14]),
         np.array([0.000, 0.020, 0.040, 0.060, 0.090, 0.115, 0.150, 0.180, 0.250, 0.25, 0.30, 0.45,
                   0.80, 1.80, 2.4, 6.4, 11, 14]),
         np.array([0.000, 0.040, 0.050, 0.075, 0.110, 0.140, 0.195, 0.250, 0.400, 0.950, 1.15, 3.2,
                   4.4, 7.4, 10.0, 14])
]

# Transfer Characteristics
# in Volts
V_GS2 = np.arange(0, -0.5 - 0.05/2, -0.05)
# in Amperes
I_D2 = np.array([24, 20, 16.5, 13.0, 9.5, 7.0, 4.5, 2.5, 1.0, 0.5, 0.0])*1e-3




# CURVE FITTING

# Drain Characteristics
def func(x, m1, m2, m3, m4, m5):
    return m1*x - m2*np.log(np.exp(m3*x) + m4) + m5

# Initialize array of empty arrays for storing optimal value of parameters for func
popt = [ [] for i in range(0, len(V_DS)) ]
for i in range(0, len(popt)):
    popt[i], pcov = curve_fit(func, V_DS[i], I_D[i], maxfev=10000)

def get_axes(fig):
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', top=True, right=True)
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle='dotted')
    return ax

# Plot I_D vs V_DS curves
fig1 = plt.figure(num=1)
ax1 = get_axes(fig1)
for i in range(0, len(V_DS)):
    # I_D in mA
    ax1.plot(V_DS[i], I_D[i]*1e3, label=r"$V_{GS}$ = " + f"{V_GS[i]:.2f}" + " V", linestyle='',
             marker='.', markersize=5)
    x = np.linspace(V_DS[i].min(), V_DS[i].max(), 1000)
    ax1.plot(x, func(x, popt[i][0], popt[i][1], popt[i][2], popt[i][3], popt[i][4])*1e3,
             linestyle='-', color='grey', linewidth=1.2)

ax1.set_xlabel(r"$V_{DS}$ (Volt)")
ax1.set_ylabel(r"$I_{D}$ (mA)")
ax1.legend()

# Transfer Characteristics
fig2 = plt.figure(num=2)
ax2 = get_axes(fig2)
# I_D in mA
ax2.plot(V_GS2, I_D2*1e3, label=r"$V_{DS}$ = 14.0V", linestyle='', marker='.', markersize=5)
poln = Polynomial.fit(V_GS2, I_D2, deg=5)
x = np.linspace(V_GS2.min(), V_GS2.max(), 1000)
y = poln(x)
ax2.plot(x, y*1e3, linestyle='-', linewidth=1.2, color='grey')

ax2.set_xlabel(r"$V_{GS}$ (Volt)")
ax2.set_ylabel(r"$I_D$ (mA)")
ax2.legend()

# Print optimal parameters used in curve fitting
print(f"\nOptimal parameters for fitting I_G vs V_DS: ", popt)
print(f"\nOptimal parameters for fitting I_G vs V_GS: ", poln)
# Save Plots
fig1.savefig("image1.png", dpi=300)
fig2.savefig("image2.png", dpi=300)


# Determining r_d, g_m and mu
# make func a single variable function
def func2(x):
    return func(x, popt[0][0], popt[0][1], popt[0][2], popt[0][3], popt[0][4])

# vds was 14V for transfer characteristics
vds0 = 14
ig0 = func2(vds0)

def poln2(x):
    return poln(x) - ig0

# step size for derivative
dx = 1e-6
rd=1/derivative(func2, x0=vds0, dx=dx)
# Solve when ig = ig0 for ig vs vgs curve
x0 = fsolve(poln2, 0)[0]
gm = derivative(poln, x0=x0, dx=dx)
mu = gm * rd
print(f"\nV_DS = {vds0} V")
print(f"I_G = {ig0*1e3 : .2f} mA")
print(f"r_d = {rd: .2f} Ohm")
print(f"g_m = {gm : .2f} mho")
print(f"mu = {mu : .2f}")
