from matscipy.neighbours import neighbour_list
from tqdm import tqdm
from matscipy.neighbours import mic 
from ase.io import read
from ase.visualize import view

def gaussian(x, mu, sig):
    return 1/sig/np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/(2*sig**2))
import numpy as np
import matplotlib.pyplot as plt

def convolution(r_it0_j0, r_cut = 5.0,grid_size=201):
    r_grid = np.linspace(0,r_cut,grid_size)
    acf = np.zeros(grid_size)
    mask1 = r_it0_j0<r_cut
    mask2 = r_it0_j0>0.
    hist, bin_edges = np.histogram(r_it0_j0[np.logical_and(mask1,mask2)], r_grid)
    bin_center = (bin_edges[:-1] + bin_edges[1:])/2
    for mu, count in zip(bin_center,hist):
         acf += 1/mu**2 *count * gaussian(r_grid, mu, sig=0.1)
    return r_grid, acf

def plot_g_r(r_grid, g_r, plot_name):
    
    fig, ax = plt.subplots(figsize=[3,2.25])
    
    for gr in g_r_pairs:
        gr /= np.sum(gr)
        
    leg = [i+'-'+j for (i,j) in atom_pairs]
    for i, gr in enumerate(g_r_pairs):
        ax.plot(r_grid, gr)
    ax.legend(leg, fontsize=5)
    ax.set_ylabel(r'g(r) norm.')
    ax.set_xlabel(r'r ($\AA{}$)')
    ax.set_xlim([0,3.5])
    fig.tight_layout()
    fig.savefig(plot_name, dpi=300)

def g_r(traj, atom_pairs, r_cut,r_grid_size):
    n_time = len(traj)
    g_pairs_t_r = np.zeros([len(atom_pairs), n_time, r_grid_size])
    #t_lag_idx = 0
    #r_t_it0_j0 = np.zeros([n_time-t_lag_idx, n_atoms1,n_atoms2])
        #g_r_t = np.zeros([n_time, r_grid_size])
    for t_idx in range(n_time):
        for atom_pair_idx, atom_pair in enumerate(atom_pairs):
            src, dst, d = neighbour_list('ijd', atoms=traj[t_idx], cutoff={atom_pair:r_cut})
            # mask1 = [traj[t_idx][i].symbol == atom_pair[0] for i in src]
            # mask2 = [traj[t_idx][j].symbol == atom_pair[1] for j in dst]
            #print(np.logical_and(mask1,mask2))
            #r_grid, g_pairs_t_r[atom_pair_idx,t_idx] = convolution(d[np.logical_and(mask1, mask2)], r_cut,r_grid_size)
            r_grid, g_pairs_t_r[atom_pair_idx,t_idx] = convolution(d, r_cut,r_grid_size)
        #g_r_pairs.append(np.average(g_r_t, axis=0))
    return r_grid, np.average(g_pairs_t_r, axis=1)

if __name__ == "__main__":
    from ase.io import read
    from itertools import combinations_with_replacement
    from glob import glob
    atom_pairs = list(combinations_with_replacement(['H', 'O','Ni', 'Nd'],2))
    
    for traj_name in tqdm(sorted(glob('*.traj'))):
        traj = read(traj_name, '25:')
        r_grid, g_r_pairs = g_r(traj, atom_pairs, 4.0, 201)
        plot_g_r(r_grid, g_r_pairs, './g_r/'+traj_name[:-5]+'.png')
