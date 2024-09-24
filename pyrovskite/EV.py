from ase.io import read
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm
from glob import glob

def plot_EV(time, E_atom, V, fig_name):

    fig, ax1 = plt.subplots(figsize=[3,2.25])
    #ax1.set_aspect(1)

    color = 'tab:red'
    ax1.set_xlabel('time (ps)')
    ax1.set_ylabel('total energy (eV/atom)', color=color)
    ax1.plot(time, E_atom, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r'volume (nm$^3$)', color=color)  # we already handled the x-label with ax1
    ax2.plot(time, V/1000, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(25))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(fig_name, dpi=300)
    plt.close()

if __name__ == '__main__':
    time_step = 2
    for traj_name in tqdm(sorted(glob('*.traj'))):
        traj = read(traj_name, ':')
        time = time_step * np.arange(len(traj)) #timestep in ps
        E_atom = np.array([atoms.get_total_energy() for atoms in traj])/len(traj[0])
        V = np.array([atoms.get_volume() for atoms in traj])
        plot_EV(time, E_atom, V, './EV/'+traj_name[:-5]+'.png')
