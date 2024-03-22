import numpy as np
from scipy.optimize import curve_fit
from scipy import constants as c
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

T1 = np.arange(35, 120 + 5/2, 5) + 273.15
T2 = np.arange(35, 85 + 5/2, 5) + 273.15

V1 = np.array([413, 404, 394, 384, 374, 363, 352, 340, 330, 318, 306, 294, 283, 270, 259, 245, 231, 215]) * 1e-3
V2 = np.array([159, 150, 143, 130, 121, 112, 103, 95, 86, 79, 69]) * 1e-3

I_F = 10 / 99e3

def func(T, Eg, B, n):
    return Eg - np.log(B*T**3/I_F) * c.k / c.e * n * T

popt1, pcov1 = curve_fit(func, T1, V1, bounds=([0.8, 0, 1], [1.3, 1, 1.1]), maxfev=1000000)
print(popt1)
popt2, pcov2 = curve_fit(func, T2, V2, bounds=([0.5, 0, 1], [1.0, 0.1, 1.1]), maxfev=1000000)
print(popt2)

fig = plt.figure(num=1)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', width=0.5, length=3)
ax.tick_params(which='both', top=True, right=True)
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle='dotted')

ax.set_xlabel(r"Temperature ($K$)")
ax.set_ylabel("Voltage ($V$)")

ax.plot(T1, V1, marker='.', linestyle='', label="Silicon")
ax.plot(T2, V2, marker='.', linestyle='', label="Germanium")
x_ = np.linspace(1, 487, 1000)
ax.plot(x_, func(x_, popt1[0], popt1[1], popt1[2]))
x_ = np.linspace(1, 395, 1000)
ax.plot(x_, func(x_, popt2[0], popt2[1], popt2[2]))

plt.legend()
plt.savefig("image.png", dpi=300)
