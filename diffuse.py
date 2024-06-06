from ase.md.analysis import DiffusionCoefficient
from ase.units import fs
from ase.io import Trajectory
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze(traj, timestep):
    D = DiffusionCoefficient(traj, timestep *fs, [atom.index for atom in traj[0] if atom.symbol=='H'])
    D.calculate()
    MSD = D.xyz_segment_ensemble_average[0,0,:,:] # in Ang^2
    D = fs * 0.1 * D.slopes[0,0,:] # now in cm2/s
    return MSD, D

def plot_MSD(traj, timestep):
    fig, ax = plt.subplots(figsize = [3.3,2.5])
    MSD, D = analyze(traj, timestep)
    t = timestep * np.arange(len(MSD[0]))/1000 #ps
    for msd, m, c in zip(MSD, ['o', 's', '^'], ['tab:blue', 'tab:orange', 'tab:green']):
        ax.scatter(t, msd, 
                   marker = m, 
                   linewidths = 0.5,
                   color=c,
                   edgecolor='slategrey')
    ax.set_ylabel(r'MSD ($\AA{}^2$)')
    ax.set_xlabel('Time (ps)')
    ax.legend([r'H$_x$', r'H$_y$', r'H$_z$'])
    fig.suptitle('D = ' + np.array2string(D, precision=2, separator=', ')+r' cm$^2$s$^{-1}$', wrap=True, fontsize = 10)
    fig.tight_layout()
    fig.savefig('./imgs/'+filename[5:-5]+'_diffusion.png', dpi=300)


if __name__ == "__main__":
    filename = sys.argv[1]
    timestep = float(sys.argv[2])
    
    traj = Trajectory(filename, mode='r')
    plot_MSD(traj, timestep)

