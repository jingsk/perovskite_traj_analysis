from matscipy.neighbours import neighbour_list
from .utils import d_to_acf
import numpy as np
import matplotlib.pyplot as plt

def g_r(traj, atom_pairs, r_cut = 4.0,r_grid_size = 201):
    n_time = len(traj)
    g_pairs_t_r = np.zeros([len(atom_pairs), n_time, r_grid_size])
    for t_idx in range(n_time):
        for atom_pair_idx, atom_pair in enumerate(atom_pairs):
            src, dst, d = neighbour_list('ijd', atoms=traj[t_idx], cutoff={atom_pair:r_cut})
            r_grid, g_pairs_t_r[atom_pair_idx,t_idx] = d_to_acf(np.array([d]), r_cut,r_grid_size)
    return r_grid, np.average(g_pairs_t_r, axis=1)

if __name__ == "__main__":
    from ase.io import read
    from itertools import combinations_with_replacement
    import sys
    from pyrovskite.plotter import plot_g_r
    atom_pairs = list(combinations_with_replacement(['H', 'O','Ni', 'Nd'],2))
    
    #for traj_name in tqdm(sorted(glob('*.traj'))):
    traj_name = sys.argv[1]
    traj = read(traj_name, '25:')
    r_grid, g_r_pairs = g_r(traj, atom_pairs, 4.0, 201)
    fig, ax = plt.subplots(figsize=[3,2.25])
    plot_g_r(r_grid, g_r_pairs, atom_pairs, ax)
    fig.tight_layout()
    fig.savefig('./g_r.png', dpi=300)
    
