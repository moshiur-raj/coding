import numpy as np
import matplotlib.pyplot as plt

resistance = 1000
capacitance = 1e-6
inductance = 0.1

frequency = np.concatenate((np.arange(100, 250 + 50, 50), np.arange(300, 1000 + 100, 100)))

lr_voltage_r = np.array([ [6.71, 6.71, 6.70, 6.68, 6.67, 6.62, 6.56, 6.50, 6.44, 6.37, 6.30, 6.22],
                          [6.71, 6.71, 6.69, 6.68, 6.65, 6.61, 6.56, 6.50, 6.43, 6.36, 6.28, 6.20] ])

lr_voltage_l = np.array([ [0.44, 0.65, 0.85, 1.05, 1.26, 1.66, 2.06, 2.45, 2.81, 3.16, 3.48, 3.84],
                          [0.43, 0.64, 0.84, 1.04, 1.26, 1.65, 2.05, 2.44, 2.82, 3.18, 3.52, 3.85] ])

rc_voltage_r = np.array([ [4.92, 5.72, 6.12, 6.32, 6.45, 6.57, 6.63, 6.66, 6.67, 6.68, 6.68, 6.68],
                          [4.92, 5.71, 6.12, 6.32, 6.44, 6.56, 6.62, 6.66, 6.67, 6.68, 6.68, 6.68] ])

rc_voltage_c = np.array([ [7.46, 5.83, 4.66, 3.88, 3.29, 2.53, 2.04, 1.71, 1.48, 1.29, 1.15, 1.03],
                          [7.46, 5.83, 4.66, 3.88, 3.29, 2.54, 2.03, 1.70, 1.47, 1.28, 1.14, 1.03] ])

lr_voltage_r_avg = (lr_voltage_r[0] + lr_voltage_r[1]) / 2
lr_voltage_l_avg = (lr_voltage_l[0] + lr_voltage_l[1]) / 2
rc_voltage_r_avg = (rc_voltage_r[0] + rc_voltage_r[1]) / 2
rc_voltage_c_avg = (rc_voltage_c[0] + rc_voltage_c[1]) / 2

xl = lr_voltage_l_avg/lr_voltage_r_avg * resistance
xc = rc_voltage_c_avg/rc_voltage_r_avg * resistance

plt.plot(frequency, xl, linestyle='-', marker='.', label="X_L")
plt.plot(frequency, xc, linestyle='-', marker='.', label ="X_C")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Impedance (Ohm)")
plt.legend()
plt.savefig("fig.png")
plt.show()
