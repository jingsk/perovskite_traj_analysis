from ase.io import read
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
#import tidynamics

# def MSD_self_fft(pos_list):
#     particle_msd = np.zeros((pos_list.shape[0], pos_list.shape[1]))
#     for n in range(pos_list.shape[1]):
#         particle_msd[:, n] = tidynamics.msd(pos_list[:,n,:])
#     return particle_msd.mean(axis=1)

def MSD_self_window(pos_list):
    lagtimes = np.arange(1, pos_list.shape[0])
    particle_msd = np.zeros((pos_list.shape[0], pos_list.shape[1],  pos_list.shape[2]))
    for lag in lagtimes:
        disp = pos_list[:-lag, :, :] - pos_list[lag:, :, :]
        sqdist = np.square(disp) #.sum(axis=-1)
        particle_msd[lag, :,:] = np.mean(sqdist, axis=0)
    return particle_msd.mean(axis=1)

def analyze(traj, timestep):
    mask = [atom.index for atom in traj[0] if atom.symbol=='H']
    pos_list = np.array([atoms[mask].get_positions() for atoms in traj])
    msd_axes_t = MSD_self_window(pos_list)
    return msd_axes_t

def linear_fit(t,msd):
    [m,b], cov = np.polyfit(t, msd, 1, cov=True)
    x_grid = np.linspace(t[0], t[-1], 5*t.size, endpoint=True)
    return x_grid, x_grid * m + b

def plot_MSD(traj, timestep, fig_name):
    fig, ax = plt.subplots(figsize = [3,2.25])
    msd_axes_t = analyze(traj, timestep)
    msd_t = np.sum(msd_axes_t, axis=-1)
    t = timestep * np.arange(len(traj)) #ps
    mask1 = np.logical_and((t>25), (t<t[-1]-10))
    t_fit, msd_fit = linear_fit(t[mask1],msd_t[mask1])
    #print(t.shape, msd_self.shape)
    for msd_axis, m, c in zip(msd_axes_t.T, ['o', 's', '^'], ['tab:blue', 'tab:orange', 'tab:green']):
        ax.scatter(t, msd_axis,
                    marker=m,
                    linewidths = 0.5,
                    color=c,
                    edgecolor='slategrey')
    ax.scatter(t, msd_t,
                marker='P',
                linewidths = 0.5,
                color='tab:purple',
                edgecolor='slategrey')
    ax.plot(t_fit, msd_fit, '-', linewidth=2, color='k')
    ax.set_ylabel(r'MSD ($\AA{}^2$)')
    ax.set_xlabel('time (ps)')
    ax.legend([r'H$_x$',r'H$_y$',r'H$_z$', 'H', 'fit'], fontsize=8)
    fig.tight_layout()
    #plt.show()
    fig.savefig(fig_name, dpi=300)
    plt.close()
    #axes[1].set_yscale('log')
    #axes[1].set_xscale('log')

if __name__ == "__main__":
    time_step = 2
    for traj_name in tqdm(sorted(glob('*.traj'))):
    #for traj_name in ['md_seed_0_conc0.0625_600K.traj']:
        traj = read(traj_name, ':')
        plot_MSD(traj, time_step, './MSD/'+traj_name[:-5]+'.png')
