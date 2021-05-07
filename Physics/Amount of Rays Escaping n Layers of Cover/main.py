#!/bin/python

import subprocess
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

num_layers_min = str(1)
num_layers_max = str(5)
step_for_num_layers = str(1)
transparency_min = str(0)
transparency_max = str(1)
num_rays = str(1e6)
num_output_points = str(int(1e3))

ans = input("Would you like to use the default values ? (Y/n) [defaults to yes] : ")
if (ans == 'n'):
    num_layers_min = str(int(input("Enter the minimum number of layers : ")))
    num_layers_max = str(int(input("Enter the maximum number of layers : ")))
    step_for_num_layers = str(int(input("Enter the step for changing number of layers : ")))
    transparency_min = str(float(input("Enter minimum value for transparency : ")))
    transparency_max = str(float(input("Enter maximum value for transparency : ")))
    num_rays = str(int(input("Enter the number of rays for simulation : ")))
    num_output_points = str(int(input("Enter the number of output points : ")))

output = subprocess.run(["./calculate", num_layers_min, num_layers_max, step_for_num_layers, transparency_min, transparency_max, num_rays, num_output_points], capture_output=True, text=True, check=True)
data = np.loadtxt(StringIO(output.stdout))
transparency = data[0]
num_layers_used = int((int(num_layers_max) - int(num_layers_min) + 1)/int(step_for_num_layers))
simulated_trap_capacity = data[1 : num_layers_used + 1]
predicted_trap_capacity = data[num_layers_used + 1 : ]

for i in range(0, num_layers_used):
    plt.plot(transparency, simulated_trap_capacity[i])
    plt.plot(transparency, predicted_trap_capacity[i])
plt.show()
