import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# bg = (19+19+14) / 3
r = np.array([4.8, 5.8, 6.8, 7.8, 8.8, 10.8, 12.8, 14.8])
R1 = np.array([277, 213, 153, 125, 97, 66, 52, 29])
R2 = np.array([251, 185, 168, 124, 89, 65, 52, 33])

# t = 100e-6
# r = (r - bg) / (1 - (r - bg)/60 * t)

def func_(x, m, n):
    return m / x**n

def func(x, m):
    return m/x**2

popt_, pcov_ = curve_fit(func_, r, R1, maxfev=1000000)
popt1, pcov1 = curve_fit(func, r, R1, maxfev=1000000)
popt2, pcov2 = curve_fit(func, r, R2, maxfev=1000000)

print("Power of r: ", popt_[1])
print("Activity of Unknown Source: ", popt2[0]/popt1[0]*3.62, " uC")

def get_axes(fig):
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', width=0.5, length=3)
    ax.tick_params(which='both', top=True, right=True)
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle='dotted')
    return ax

fig_ = plt.figure(num=1)
ax_ = get_axes(fig_)
ax_.set_xscale("log", base=10)
ax_.set_yscale("log", base=10)
ax_.set_xlabel(r"r($cm$)")
ax_.set_ylabel("R ($counts/min$)")
ax_.plot(r, R1, marker='.', linestyle='', label="Experimental Points")
x_ = np.linspace(r.min(), r.max(), 1000)
ax_.plot(x_, func_(x_, popt_[0], popt_[1]), label="Best Fit Curve")

fig = plt.figure(num=2)
ax = get_axes(fig)
ax.set_xlabel(r"1/r^2 ($cm^{-2}$)")
ax.set_ylabel("R ($counts/min$)")

ax.plot(1/r**2, R1, marker='.', linestyle='', label="Known Source")
ax.plot(1/x_**2, func(x_, popt1[0]), label="Best Fit Curve For Known Source")
ax.plot(1/r**2, R2, marker='.', linestyle='', label="Unknown Source")
ax.plot(1/x_**2, func(x_, popt2[0]), label="Best Fit Curve For Unknown Source")


plt.legend()
fig_.savefig("image_.png", dpi=300)
fig.savefig("image.png", dpi=300)
