import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

f0 = np.array([7.42, 6.67, 5.94, 5.49, 5.19])*1e14
V0 = np.array([1.698, 1.430, 1.140, 0.954, 0.820])

f = np.array([7.32, 6.67, 5.94, 5.50, 5.22])*1e14
V = np.array([1.04, 0.91, 0.75, 0.64, 0.57])

e =  1.602176634e-19
def stopping_potential(frequency, h_, phi_):
    return h_*1e-34 / e * frequency - phi_

popt0, pcov0 = curve_fit(stopping_potential, f0, V0)
h0 = popt0[0]*1e-34
phi0 = popt0[1]
print("For Holmarc data")
print("Optimal value of h :", h0, " Js")
print("Optimal value of phi :", phi0, " eV")

popt, pcov = curve_fit(stopping_potential, f, V)
h = popt[0]*1e-34
phi = popt[1]
print("For experimental data")
print("Optimal value of h :", h, " Js")
print("Optimal value of phi :", phi, " eV")

fig = plt.figure(num=1)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', width=0.5, length=3)
ax.tick_params(which='both', top=True, right=True)
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle='dotted')

ax.plot(f, V, marker='.', linestyle='', label="Experimental Data")
ax.plot(f0, V0, marker='.', linestyle='', label="Hallmark's Data")
X0 = np.linspace(0, 1.05*f0.max(), 1000)
Y0 = stopping_potential(X0, popt0[0], popt0[1])
ax.plot(X0, Y0)
X = np.linspace(0, 1.05*f.max(), 1000)
Y = stopping_potential(X, popt[0], popt[1])
ax.plot(X, Y)

ax.set_xlim(-0.5, 1.05*max(X0.max(), X.max()))
ax.set_ylim(min(Y0.min(), Y.min())*1.05, max(Y0.max(), Y.max())*1.05)
ax.set_xlabel(r"$\nu \, \,(10^{14} Hz)$")
ax.set_ylabel("V (Volt)")
plt.legend()
plt.savefig("image.png", dpi=600)
