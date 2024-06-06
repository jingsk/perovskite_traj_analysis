import numpy as np
import matplotlib.pyplot as plt
import sys
from ase.io import read
from H_r import H_r
from oct_vol import oct_vol

if __name__ == "__main__":
    fig_name = str(sys.argv[1])
    timestep = 0.2 #ps
    fig, ax = plt.subplots(figsize=[4,3])
    for time in [sys.argv[3], sys.argv[4]]:
        atoms = read(sys.argv[2],time)
        Ni_idx = np.array([atom.index for atom in atoms if atom.symbol=='Ni'])
        y = oct_vol(Ni_idx = Ni_idx,
                          atoms = atoms,
                          r_max = 7)
        x = H_r(Ni_idx = Ni_idx,
                  H_idx = np.array([atom.index for atom in atoms if atom.symbol=='H']),
                  pos = atoms.get_positions(),
                  sig = 4
            )
        ax.plot(x, y, '.')
    ax.set_title(fig_name[-8:-4])
    ax.legend([f'{float(sys.argv[3])*timestep:.01f} ps', f'{float(sys.argv[4])*timestep:.01f} ps'])
    ax.set_xlabel(r'$H_r$')
    ax.set_ylabel(r'oct. vol. ($\AA{}^3$)')
    fig.tight_layout()
    fig.savefig(fig_name, dpi=300)
