import subprocess
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Fps = 40
Alpha = "0.01"
NumFrames = "1000"
NumPositions = "10000"
Nmax = "50"
TimeMin = "0.0"
TimeMax = str(int(NumFrames)/Fps)
Length = "1.0"

output = subprocess.run(["./calculate",
                        Alpha, NumFrames, NumPositions, Nmax, TimeMin, TimeMax, Length],
                        capture_output=True, text=True, check=True).stdout
data = np.loadtxt(StringIO(output))
position = data[0]
Time = np.linspace(float(TimeMin), float(TimeMax), int(NumFrames))

fig = plt.figure()
ax = fig.add_subplot()
ln, = plt.plot([], [])

def init():
    ax.set_xlim(0, float(Length))
    ax.set_ylim(np.amin(data[1:int(NumFrames)+1])*0.98, np.amax(data[1:int(NumFrames)+1])*1.02)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Temperature (K)")
    ln.set_label("time = {:.2f}".format(Time[0]))
    ax.legend()
    return ln,

def update(i):
    ln.set_data(data[0], data[i])
    ln.set_label("time = {:.2f}".format(Time[i]))
    ax.legend()
    return ln,

animation = FuncAnimation(fig, update, frames=range(0, 600), init_func=init, interval=int(1000/Fps), blit=True)

animation.save("animation.mkv", writer="ffmpeg", dpi=300)
