from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

whatToPlot = int(argv[1])
fps = float(argv[2])
nframes = int(argv[3])
ndim = int(argv[4])
nparticles = int(argv[5])
xmin, xmax = (float(argv[7]), float(argv[8]))

fig = plt.figure()

if (ndim == 2):
    ymin, ymax = (float(argv[9]), float(argv[10]))
    points = np.memmap("/dev/shm/" + argv[6], dtype=np.float64, mode='r',
                       shape=(nframes, ndim, nparticles))
    ax  = fig.add_subplot(1, 1, 1)
    ln, = ax.plot([], [])

    def init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return ln,

    def update(frame):
        ln.set_data(points[frame][0], points[frame][1])
        return ln,

    animation = FuncAnimation(fig, update, frames=range(0, nframes), init_func=init,
                              interval=int(1000.0/fps), blit=True)
    animation.save("animation.mkv", writer="ffmpeg", dpi=300)

elif (ndim == 3):
    zmin, zmax = (float(argv[11]), float(argv[12]))
