import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

V = np.arange(840, 1220 + 20/2, 20)
R = np.array([ [0, 0, 0], [0, 1, 0], [5769, 5668, 5997], [6459, 6372, 6408], [6894, 6976, 6751],
               [7067, 6981, 7076], [7314, 7343, 7526], [7464, 7431, 7471], [7588, 7782, 7560],
               [7843, 7842, 7701], [7900, 7991, 7974], [8123, 8011, 8163], [8170, 8134, 8159],
               [8430, 8240, 8062], [8310, 8277, 8297], [8381, 8482, 8405], [8554, 8379, 8608],
               [8498, 8582, 8650], [8634, 8771, 8746], [8747, 8824, 8745] ])
R = np.array([ np.average(R[i]) for i in range(0, len(R)) ])
print(R)

polyfit = Polynomial(np.polynomial.polynomial.polyfit(V, R, 5))

fig = plt.figure(num=1)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', width=0.5, length=3)
ax.tick_params(which='both', top=True, right=True)
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle='dotted')

ax.plot( V, R, marker='.', linestyle='', label="Experimental Data")
V_ = np.linspace(V.min(), V.max(), 1000)
R_ = polyfit(V_)
ax.plot(V_, R_, label="Best Fit Curve")

xmin = min(V.min(), V_.min())
xmin *= (xmin > 0)*0.95 + ( xmin < 0)*1.05
xmax = max(V.max(), V_.max())
xmax *= (xmax > 0)*1.05 + ( xmax < 0)*0.95

ymin = min(R.min(), R_.min())
ymin *= (ymin > 0)*0.95 + ( ymin < 0)*1.05
ymax = max(R.max(), R_.max())
ymax *= (ymax > 0)*1.05 + ( ymax < 0)*0.95

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r"V")
ax.set_ylabel(r"R")
plt.legend()
plt.show()
# plt.savefig("image.png", dpi=600)
