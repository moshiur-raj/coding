import numpy as np
from scipy.optimize import curve_fit, fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

t = 100e-6

x = np.array([0, 14.3, 16.2, 19.4, 21.85, 26.80, 35.70, 41.70, 47.00, 56.50, 67.50, 79.80, 102, 125, 143, 191, 241, 332, 432, 548, 702, 875, 875+102, 875+241, 875+332])

r = np.array([1914, 1660, 1627, 1639, 1611, 1514, 1443, 1359, 1356, 1212, 1154, 1100, 1000, 929, 902, 731, 600, 407, 236, 131, 56, 33, 27, 28, 28])

r_uk = 588

r = r / (1 - r/60 * t)

def func(x, mu_1, mu_2, r0_1, r0_2, c=r[-1]):
    return 1e3*r0_1 * np.exp(-mu_1*1e-3 * x) + 1e2*r0_2 * np.exp(-mu_2*1e-2 * x) + c

popt, pcov = curve_fit(func, x, r, maxfev=1000000)

mu_1 = popt[0]*1e-3
mu_2 = popt[1]*1e-2
r0_1 = popt[2]*1e3
r0_2 = popt[3]*1e2

R = fsolve(func, 800, (mu_1, mu_2, r0_1, r0_2, -1))
X = fsolve(func, 200, (mu_1, mu_2, r0_1, r0_2, - (r_uk - r[-1])))

print("Absorption Coefficients: ", mu_1, mu_2)
print("Initial Count Rates:", r0_1, r0_2)
print("R = ", R)
print("X = ", X)

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
ax.semilogy(x_, func(x_, popt[0], popt[1], popt[2], popt[3]), label="Beta + Background")

ax.semilogy(x_, func(x_, popt[0], popt[1], popt[2], popt[3], 0), label="Only Beta")
ax.semilogy(x_, np.full(x_.shape, r[-1]), label="Only Background")

plt.legend()
plt.savefig("image.png", dpi=300)
