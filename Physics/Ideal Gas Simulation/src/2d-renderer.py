#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.loadtxt("xdata.csv", delimiter=',', dtype=float)
y = np.loadtxt("ydata.csv", delimiter=',', dtype=float)

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)

ln, = ax.plot([], [], 'r.')

def init():
    # ax.set_xlim(x.min(), x.max())
    # ax.set_ylim(y.min(), y.max())
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    return ln,

def update(frame):
    ln.set_data(x[frame], y[frame])
    return ln,

animation = FuncAnimation(fig, update, frames=range(0, x.shape[0]), init_func=init,
                          interval=int(1000/30), blit=True)
animation.save("animation.mkv", writer="ffmpeg", dpi=300)
