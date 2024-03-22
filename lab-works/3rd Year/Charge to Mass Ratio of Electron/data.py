import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Start and stop are in cm
def Range(start, stop):
    step = 0.5
    # r = diameter / 2
    return np.arange(start, stop + step/2, step)*1e-2 / 2

# Voltage difference between cathode and anode
Va = [np.array([ 85, 94, 108, 123, 134, 151, 165]),
      np.array([ 85, 97, 112, 125, 139, 155, 177, 194]),
      np.array([ 85, 96, 112, 126, 144, 164, 182, 205, 228]),
      np.array([ 93, 107, 125, 144, 164, 185, 205, 233, 262]),
      np.array([ 88, 105, 122, 140, 160, 188, 210, 238, 266]) ]

# Current applied in the helmholtz coil
I = np.arange(1.20, 1.60 + 0.10/2, 0.10)

# Radius of the circle
r = [ Range(7.0, 10.0), Range(6.5, 10.0), Range(6.0, 10.0), Range(6.0, 10.0), Range(5.5, 9.5) ]

# Permeability of free space
mu_0 = 1.25663706212e-6
# Number of turns in each helmholtz coil
N = 130
# Radius of the helmholtz coil
R = 0.15
B = mu_0 * (4/5)**(3/2) * N * I / R
Br2 = [B[i]**2 * r[i]**2 for i in range(0, len(r))]

em = [  2 * Va[i] / (B[i]**2 * r[i]**2) for i in range(0, len(r)) ]

# print("B")
# print(B)
# print("Br2")
# print(Br2)
# print("e/m")
# print(em)
# print("mean")
# print([em[i].mean() / 1e11 for i in range(0, len(em))])
# print("std")
# print([em[i].std() * np.sqrt(em[i].size / (em[i].size + 1)) / 1e11 for i in range(0, len(em))])

Va = np.concatenate(Va)
Br2 = np.concatenate(Br2)
em = np.concatenate(em)
print("Mean value of e/m is: ", em.mean()/1e11, "x10^11 C/kg")
print("Standard error: ", em.std() / 1e11, "x10^11 C/kg")

def voltage(Br2, em):
    return 1e11*em/2 * Br2

popt, pcov = curve_fit(voltage, Br2, Va)

em_optimal = popt[0]*1e11
print("Optimal Value of e/m is: ", em_optimal/1e11, "x10^11 C/kg")

# plt.rcParams['text.usetex'] = True
fig = plt.figure(num=1)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', width=0.5, length=3)
ax.tick_params(which='both', top=True, right=True)
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle='dotted')

ax.plot(Br2, Va, marker='.', linestyle='', label="Experimental Data")
X = np.linspace(Br2.min(), Br2.max(), 1000)
Y = em_optimal/2 * X
ax.plot(X, Y, label="Best Fit Curve")
ax.set_xlim(Br2.min()*0.95, Br2.max()*1.05)
ax.set_ylim(Va.min()*0.95, Va.max()*1.05)
ax.set_xlabel(r"$B^2 r^2 \, (T^2 \, m^2)$")
ax.set_ylabel("Va (Volt)")
plt.legend()
plt.savefig("image.png", dpi=600)
