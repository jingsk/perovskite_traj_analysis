from ase.io import read
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
from pyrovskite.plotter import plot_g_r, plot_acf_2d, plot_EV, plot_MSD
import numpy as np
from pyrovskite.g_r import g_r
from pyrovskite.acf import hv_Gd
from pyrovskite.msd import MSD_self_window
from itertools import compress

def dashboard(traj_name, image_name = './dashboard.png'):
    traj = read(traj_name, ':')
    mask = [atom.index for atom in traj[0] if atom.symbol=='H']
    traj_H = [atoms[mask] for atoms in traj]
    time_step = 2
    time = time_step * np.arange(len(traj)) #ps
    atom_pairs = list(combinations_with_replacement(['H', 'O','Ni', 'Nd'],2))
    fig, axs = plt.subplots(
        2,2,
        figsize=[6,4.5], 
        width_ratios = [1,1],
        height_ratios = [1,1],
        layout="constrained")

    E_atom = np.array([atoms.get_total_energy() for atoms in traj])/len(traj[0])
    V = np.array([atoms.get_volume() for atoms in traj])
    plot_EV(time, E_atom, V, axs[0,0])

    # msd
    #axis decomposed msd 
    pos_list = np.array([atoms.get_positions() for atoms in traj_H])
    msd_axes_t = MSD_self_window(pos_list)
    plot_MSD(msd_axes_t, time, axs[0,1])
    
    #Gd
    mask_t = np.logical_and((time>25), (time<time[-1]-20))
    r, acf_t = hv_Gd(list(compress(traj_H, mask_t)))
    plot_acf_2d(time[mask_t], r, acf_t, axs[1,0])
    
    #g_r
    r_grid, g_r_pairs = g_r(traj[25:], atom_pairs, 4.0, 201)
    plot_g_r(r_grid, g_r_pairs, atom_pairs, axs[1,1])

    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2, right=0.8, top=0.8, wspace=None, hspace=None)
    fig.savefig(image_name, dpi=300)