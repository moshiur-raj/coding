import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LinearLocator, AutoLocator

import data_ngspice

f = np.array([10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1.0e3, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.0e3, 5.0e3, 6.0e3, 7.0e3, 8.0e3, 9.0e3, 10e3, 15e3, 20e3, 25e3, 30e3, 32e3, 34e3, 36e3, 38e3, 40e3, 42e3, 44e3, 46e3, 48e3, 50e3, 55e3, 60e3, 70e3, 80e3, 90e3, 100e3, 150e3, 200e3, 250e3, 300e3])

v_in = 0.5 * np.sqrt(2)
v_out = np.array([36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 34, 32, 31, 30, 28, 27, 26, 24, 23, 22, 21, 18, 15, 12, 9, 7, 6, 3, 2, 1, 1]) / 40 * v_in

fig1 = plt.figure(num=1)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_major_locator(AutoLocator())
ax1.tick_params(which='minor', width=0.5, length=3)
ax1.tick_params(which='both', top=True, right=True)
ax1.grid(which='major', linestyle='--')
ax1.grid(which='minor', linestyle='dotted')

ax1.set_xscale('log')
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Normalized Voltage Gain (dB)")
ax1.plot(f, 20*np.log10(v_out / v_in), marker='.', linestyle='', label='Experimental Points')
ax1.plot(data_ngspice.f_sim, 20*np.log10(np.abs(data_ngspice.v_out_sim) / v_in), label='Ngspice Simulation')

ax2 = ax1.twinx()
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_major_locator(AutoLocator())
# ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
# nticks = 7
# ax1.yaxis.set_major_locator(LinearLocator(nticks))
# ax2.yaxis.set_major_locator(LinearLocator(nticks))
ax2.tick_params(which='minor', width=0.5, length=3)
ax2.tick_params(which='both', top=True, right=True)
ax2.grid(which='major', linestyle='--')
ax2.grid(which='minor', linestyle='dotted')

ax2.set_xscale('log')
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Voltage Gain")
ax2.plot(f, v_out / v_in, marker='.', linestyle='', label='Experimental Points')
ax2.plot(data_ngspice.f_sim, np.abs(data_ngspice.v_out_sim) / v_in, label='Ngspice Simulation')

ax1.legend()
ax2.legend()
# fig1.savefig("image1.png", dpi=300)
plt.show()
