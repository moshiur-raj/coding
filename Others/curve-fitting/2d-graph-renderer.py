#!/usr/bin/python

from sys import argv
import numpy as np
import matplotlib.pyplot as plt

shm_block1_path = "/dev/shm" + argv[1]
point_count = int(argv[2])
shm_block2_path = "/dev/shm" + argv[3]
observed_point_count = int(argv[4])
xlabel = argv[5]
ylabel = argv[6]
legend = argv[7]

data1 = np.memmap(shm_block1_path, dtype=np.float64, mode='r', shape=point_count*2)
plt.plot(data1[0:point_count], data1[point_count:2*point_count])

data2 = np.memmap(shm_block2_path, dtype=np.float64, mode='r', shape=observed_point_count*2)
plt.plot(data2[0:observed_point_count], data2[observed_point_count:2*observed_point_count], 'r.')

plt.xlabel(xlabel)
plt.ylabel(ylabel)
if legend != "":
    plt.legend(legend)
plt.show()
