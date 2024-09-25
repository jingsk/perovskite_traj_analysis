from ase.io import read
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

def MSD_self_window(pos_list):
    lagtimes = np.arange(1, pos_list.shape[0])
    particle_msd = np.zeros((pos_list.shape[0], pos_list.shape[1],  pos_list.shape[2]))
    for lag in lagtimes:
        disp = pos_list[:-lag, :, :] - pos_list[lag:, :, :]
        sqdist = np.square(disp) #.sum(axis=-1)
        particle_msd[lag, :,:] = np.mean(sqdist, axis=0)
    return particle_msd.mean(axis=1)

if __name__ == "__main__":
    from ase.io import read
    import sys
    from pyrovskite.plotter import plot_MSD
    #for traj_name in sorted(glob('*.traj')):
    traj_name = sys.argv[1]
    traj = read(traj_name, ':')
    time_step = 2
    time = time_step * np.arange(len(traj)) #ps
    
    #axis decomposed msd 
    mask = [atom.index for atom in traj[0] if atom.symbol=='H']
    pos_list = np.array([atoms[mask].get_positions() for atoms in traj])
    msd_axes_t = MSD_self_window(pos_list)

    fig, ax = plt.subplots(figsize = [3,2.25])  
    plot_MSD(msd_axes_t, time, ax)
    fig.tight_layout()
    #plt.show()
    fig.savefig('./msd.png', dpi=300)
    #for traj_name in tqdm(sorted(glob('*.traj'))):
    #plot_MSD(traj, time_step, './MSD/'+traj_name[:-5]+'.png')
