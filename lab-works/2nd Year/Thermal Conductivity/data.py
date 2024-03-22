import matplotlib.pyplot as plt
import numpy as np

temperature = np.arange(70, 48 - 1, -1)
time = [ [0, 0], [0, 46], [1, 37], [2, 25], [3, 20], [4, 11], [5, 7], [6, 5], [7, 4], [8, 4],
         [9, 8], [10, 14], [11, 19], [12, 31], [13, 46], [15, 3], [16, 26], [17, 55], [19, 21],
         [20, 56], [22, 30], [24, 13], [26, 3] ]
time = np.array([item[0] for item in time])*60 + np.array([item[1] for item in time])

coeff = np.polyfit(time, temperature, 3)
print(coeff)
time_polyfit = np.linspace(0, time[time.size - 1], 1000)
poly = np.poly1d(coeff)
temperature_polyfit = poly(time_polyfit)

plt.plot(time, temperature, linestyle='', marker='.')
plt.plot(time_polyfit, temperature_polyfit)
plt.xlabel("Time (s)")
plt.ylabel("Temperature (C)")
plt.legend()
plt.savefig('fig.png')
plt.show()
