import numpy as np
from tqdm import tqdm
from matscipy.neighbours import mic
from .utils import d_to_acf
from pyrovskite.plotter import plot_acf_2d
        
def hv_Gd(traj):
    n_atoms = len(traj[0])
    n_time = len(traj)
    acf_t = []
    for t_lag_idx in tqdm(range(n_time)):
        r_t_it0_j0 = np.zeros([n_time-t_lag_idx, n_atoms,n_atoms])
        for t_idx in range(n_time-t_lag_idx):
            atoms_t = traj[t_idx]
            atoms_t0 = traj[t_idx + t_lag_idx]
            for i in range(n_atoms):
                dis_vec = np.vstack([
                    atoms_t0.get_positions()[:i], 
                    atoms_t.get_positions()[i], 
                    atoms_t0.get_positions()[i+1:]]
                    ) - atoms_t.get_positions()[i]
                r_t_it0_j0[t_idx, i] = np.linalg.norm(mic(dis_vec, atoms_t.cell),axis=1)
        r_grid, acf_t_idx = d_to_acf(r_t_it0_j0, r_cut = 4.0,grid_size=201)
        acf_t.append(acf_t_idx/r_t_it0_j0.shape[0])
    return r_grid, acf_t

if __name__ == '__main__':
    from ase.io import read
    import matplotlib.pyplot as plt
    import sys
    traj_name = sys.argv[1]
    traj = read('./traj/'+traj_name, '20:121')
    time_step = 2 #ps
    fig, ax = plt.subplots(figsize=[3,2.25])
    r, acf_t = hv_Gd(
        [atoms[[atom.index for atom in atoms if atom.symbol=='H']] for atoms in traj]
        )
    time = 2* np.arange(len(traj))
    plot_acf_2d(time, r, acf_t, ax)
    fig.tight_layout()
    fig.savefig('acf.png', dpi=300)