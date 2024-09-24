import matplotlib.pyplot as plt
import numpy as np
#from glob import glob

if __name__ == '__main__':
    from ase.io import read
    import sys
    from pyrovskite.plotter import plot_EV
    time_step = 2
    #for traj_name in sorted(glob('*.traj')):
    traj_name = sys.argv[1]
    traj = read(traj_name, ':')
    fig, ax1 = plt.subplots(figsize=[3,2.25])
    time = time_step * np.arange(len(traj)) #timestep in ps
    E_atom = np.array([atoms.get_total_energy() for atoms in traj])/len(traj[0])
    V = np.array([atoms.get_volume() for atoms in traj])
    plot_EV(time, E_atom, V, ax1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('./EV.png', dpi=300)