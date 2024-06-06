from ase.io import read
import numpy as np
import sys
from tqdm import tqdm

#previously used sigma=3 and obtained smooth line density
#def gau(x, mu, sig):
#    return 1/sig/np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/(2*sig**2))

def cart_squared_distance(p1, p2):
    squared_dist = np.sum((p1-p2)**2, axis=0)
    return squared_dist

def gaussian_at_r(r2,sig):
    return 1/sig/np.sqrt(2*np.pi)*np.exp(-r2/(2*sig**2))

def H_r(Ni_idx, H_idx, pos):
    H_r = []
    for i in tqdm(Ni_idx):
        H_conc = 0
        for j in H_idx:
            H_conc += gaussian_at_r(r2 = cart_squared_distance(pos[i], pos[j]),
                                    sig = 1.5)
    H_r.append(H_conc)
    return H_r



if __name__ == "__main__":
    atoms = read(sys.argv[1],sys.argv[2])
    Ni_idx = np.array([atom.index for atom in atoms if atom.symbol=='Ni'])
    
    H_r = H_r(Ni_idx = Ni_idx,
              H_idx = np.array([atom.index for atom in atoms if atom.symbol=='H']),
              pos = atoms.get_positions()
        )
    np.savetxt('H_r.csv', 
               np.vstack([Ni_idx, H_r]).T, 
               fmt=['%d', '%.4f'])