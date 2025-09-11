import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def default_fig_ax(frameon: bool = False):
    fig = plt.figure(frameon=frameon)
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', width=0.5, length=3)
    ax.tick_params(which='both', top=True, right=True)
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle='dotted')

    return fig, ax
