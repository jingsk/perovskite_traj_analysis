import numpy as np
import matplotlib.pyplot as plt
import sys
from ase.io import read
from H_r import H_r
from oct_vol import oct_vol

if __name__ == "__main__":
    fig_name = sys.argv[1]
    atoms = read(sys.argv[2],sys.argv[3])
    Ni_idx = np.array([atom.index for atom in atoms if atom.symbol=='Ni'])
    oct_vol = oct_vol(Ni_idx = Ni_idx,
                      atoms = atoms,
                      r_max = 7)
    H_r = H_r(Ni_idx = Ni_idx,
              H_idx = np.array([atom.index for atom in atoms if atom.symbol=='H']),
              pos = atoms.get_positions()
        )
    
    
    fig, ax = plt.subplots(figsize=[4,3])
    ax.plot(H_r, oct_vol, '.')
    ax.set_xlabel(r'$H_r$')
    ax.set_ylabel(r'oct. vol. ($\AA{}^3$)')
    fig.tight_layout()
    fig.savefig(fig_name, dpi=300)
