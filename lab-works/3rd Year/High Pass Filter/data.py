import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import data_ngspice

f = np.array([10, 50, 100, 500, 1e3, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3, 150e3, 200e3, 250e3, 300e3, 400e3, 500e3, 600e3, 700e3, 800e3, 900e3])

v_in = 0.5 * np.sqrt(2)
v_out = np.array([0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 10, 28, 46, 48, 46, 42, 40, 40, 39, 38, 38, 38, 38, 38, 38, 38, 38, 38]) / 40 * v_in

fig = plt.figure(num=1)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor', width=0.5, length=3)
ax.tick_params(which='both', top=True, right=True)
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle='dotted')

ax.set_xscale('log')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel(" Normalized Voltage Gain (dB)")
ax.plot(f, 20*np.log10(v_out / v_in), marker='.', linestyle='', label='Experimental Points')
ax.plot(data_ngspice.f_sim, 20*np.log10(np.abs(data_ngspice.v_out_sim) / v_in), label='Simulated Curve')
plt.legend()
# plt.show()
plt.savefig("image.png", dpi=600)
