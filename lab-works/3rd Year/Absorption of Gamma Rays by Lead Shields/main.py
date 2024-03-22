import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

# in mm
width = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.2, 1.2, 1.2, 1.1, 1.0, 1.0, 1.2, 1.0, 1.2, 1.2, 1.1, 1.1],
                  [1.0, 0.9, 1.0, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 0.9, 0.9, 1.0],
                  [1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0],
                  [1.0, 0.9, 1.0, 1.0, 0.9, 0.9, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0],
                  [0.9, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 1.0, 0.9],
                  [1.0, 1.0, 1.0, 0.9, 0.8, 0.9, 0.9, 1.0, 0.8, 1.0, 0.9, 0.9],
                  [1.1, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.1, 1.1, 1.1, 0.9, 1.0]])

x = width.mean(axis=1).cumsum()

# Count per minute
cpm = np.array([[10225, 10270, 10258], [7371, 7432, 7259], [6423, 6513, 6431], [5576, 5499, 5590],
              [5071, 5003, 5172], [4612, 4679, 4666], [4242, 4264, 4264], [3872, 3801, 3862]])
R = cpm.mean(axis=1)

def func1(x, R0, mu):
    return R0 * np.exp(-mu * x)

def func2(x, R10, mu1, R20, mu2):
    return R10 * np.exp(-mu1 * x) + R20 * np.exp(-mu2 * x)

popt1, pcov1 = curve_fit(func1, x, R, maxfev=100000)
popt2, pcov2 = curve_fit(func2, x, R, maxfev=100000)

fig = plt.figure(num=1)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', width=0.5, length=3)
ax.tick_params(which='both', top=True, right=True)
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle='dotted')

ax.set_xlabel("x (mm)")
ax.set_ylabel("R (Counts/min)")

ax.plot(x, R, marker='.', linestyle='', label="Experimental Points")
x_ = np.linspace(x.min(), x.max(), 1000)
ax.plot(x_, func1(x_, popt1[0], popt1[1]), label="Fit for Only Gamma Rays")
ax.plot(x_, func2(x_, popt2[0], popt2[1], popt2[2], popt2[3]), label="Fit for Both Alpha and Gamma Rays")

plt.legend()
plt.savefig("image.png", dpi=300)
# plt.show()
print(x)
print(R)
print(popt1)
print(popt2)
