import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy import constants as c
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

T = np.arange(300, 65 - 5/2, -5)[::-1]
V = np.array([ 3.42, 3.42, 3.43, 3.45, 3.46, 3.48, 3.50, 3.52, 3.56, 3.62, 3.68, 3.76, 3.82, 3.91, 4.02, 4.14, 4.27, 4.46, 4.65, 4.93, 5.30, 5.74, 6.24, 6.63, 6.98, 7.28, 7.58, 7.86, 8.11, 8.35, 8.55, 8.74, 9.04, 9.25, 9.50, 9.65, 9.84, 10.05, 10.21, 10.37, 10.54, 10.70, 10.87, 11.04, 11.25, 11.32, 11.42, 11.64 ])[::-1]

# def func(T, T_c, a0, a1, a2, a3, b1, b2, b3):
#     return a0 + a1*T + a2*T**2 + a3*T**3 + b1/T + b2/T**2 + b3/T**3

def func(x, m1, m2, m3, m4, m5):
    return m1*x - m2*np.log(np.exp(m3*x) + m4) + m5

popt, pcov = curve_fit(func, T, V, maxfev=1000000)
T_c = popt[0]
print(popt)

# deg = 30
# popt = np.polyfit(T, V, deg)

# cs = CubicSpline(T, V)

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

ax.plot(T, V, marker='.', linestyle='', label="Experimental Points")
T_ = np.linspace(T.min(), T.max(), 1000)

# ax.plot(T_, cs(T_))

# ax.plot(T_, np.polyval(popt, T_))

# T_ = np.concatenate((np.linspace(T.min(), T_c, 1000, endpoint=False), np.linspace(T.max(), T_c, 1000, endpoint=False)))
ax.plot(T_, func(T_, *popt), label="Best Fit Curve")

plt.legend()
# plt.savefig("image.png", dpi=300)
plt.show()
