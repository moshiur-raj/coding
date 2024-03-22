import numpy as np
import matplotlib.pyplot as plt

voltage = np.array([ 0.600, 0.626, 0.648, 0.660, 0.671, 0.681, 0.691, 0.698, 0.706, 0.713, 0.719,
                     0.725, 0.731, 0.737, 0.741, 0.747, 0.752, 0.757, 0.761, 0.766, 0.770, 0.775,
                     0.780, 0.784, 0.789, 0.793, 0.798, 0.801, 0.806, 0.810, 0.814, 0.818, 0.822,
                     0.826, 0.829, 0.833, 0.837, 0.841, 0.845, 0.849, 0.852, 0.856, 0.860, 0.865,
                     0.867, 0.871, 0.875, 0.878, 0.883, 0.887, 0.890 ])

current = np.arange(0, 100 + 2, 2)

poly_coeff = np.polyfit(voltage, current, 5)
voltage_polyfit = np.linspace(voltage.min(), voltage.max(), 1000)
current_polyfit = np.poly1d(poly_coeff)(voltage_polyfit)

plt.plot(voltage, current, linestyle='', marker='.')
plt.plot(voltage_polyfit, current_polyfit)
plt.xlabel("Voltage (V)")
plt.ylabel("Current (mA)")
plt.legend()
plt.savefig("fig.png")
plt.show()
