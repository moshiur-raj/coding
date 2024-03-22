import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

bg = (30 + 25 + 30 + 26 + 31 + 24) / 6
t = 100e-6
x = np.array([0, 4.5, 6.5, 14.1, 28.1, 59.1, 102, 129, 161, 206, 258, 328, 419, 516, 590, 645, 849,
              1230, 1890, 3632, 7435])
r = np.array([5103+5100+5015, 4509+4464+4411, 4244+4219+4153, 3596+3618+3563, 2882+2878+2758,
              1323+1306+1351, 981+957+988, 671+624+678, 469+473+474, 360+337+349, 291+325+314,
              296+312+295, 298+313+318, 286+283+311, 288+304+316, 263+303+283, 278+268+309,
              315+323+282, 267+275+309, 246+258+252, 160+167+171]) / 3

r = (r - bg) / (1 - (r - bg)/60 * t)

def func(x, mu1, mu2, r10, r20):
    return r10 * np.exp(-mu1 * x) + r20 * np.exp(-mu2 * x)

popt, pcov = curve_fit(func, x, r, maxfev=1000000)

print("Absorption Coefficient for Particle 1: ", popt[0])
print("Absorption Coefficient for Particle 2: ", popt[1])

print("Initial Count Rate for Particle 1: ", popt[2])
print("Initial Count Rate for Particle 2: ", popt[3])

fig = plt.figure(num=1)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', width=0.5, length=3)
ax.tick_params(which='both', top=True, right=True)
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle='dotted')

ax.set_xlabel(r"x ($mg/cm^2$)")
ax.set_ylabel("R ($counts/min$)")

ax.plot(x, r, marker='.', linestyle='', label="Experimental Points")
x_ = np.linspace(x.min(), x.max(), 1000)
ax.plot(x_, func(x_, popt[0], popt[1], popt[2], popt[3]), label="Best Fit Curve")

plt.legend()
plt.savefig("image.png", dpi=300)
