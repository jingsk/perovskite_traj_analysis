from ase.io import read
import numpy as np
from util.octahedron import octahedral_volume1
from ase.neighborlist import neighbor_list
import sys
from tqdm import tqdm

def trim(src, 
         dst, 
         vec, 
         species= 'O', 
         minimum = 6):
    mask = np.ones(len(dst), dtype=bool)
    chem_sym = np.array(atoms[dst].get_chemical_symbols())
    #if np.any(chem_sym=='H'):
        #print(src)
    chem_mask = (chem_sym==species)
    mask = (mask) & chem_mask
    
    dis_mask = np.zeros(len(vec[:,0]),dtype='bool')
    argsort = np.argsort(np.linalg.norm(vec, axis=1))
    #print(argsort)
    while np.sum(dis_mask) <minimum: 
        if (argsort.size == 0) | (chem_mask.size==0):
            return np.zeros(len(vec[:,0]),dtype='bool')
        dis_mask[argsort[0]]= (True) & (chem_mask[argsort[0]])
        argsort = np.delete(argsort, 0)
    mask = (mask) & dis_mask
    return mask
    


#atoms = read('./01_SCF_111k_400eV/POSCAR')
#atoms = read('./pristine/01_SCF_111k_400eV/POSCAR')
#atoms = read('/lcrc/globalscratch/jingt/perovskite/Nd/relax/04_OPT_sukriti/CONTCAR')
atoms = read(sys.argv[1],sys.argv[2])
#atoms = read('supercell_18_2.poscar')
#atoms = read('supercell_18.poscar')

Ni_idx = np.array([atom.index for atom in atoms if atom.symbol=='Ni'])
#print(Ni_idx) 
r_max = 7 #/ (6) * (atoms.get_volume()) ** (1/3)
edge_src, edge_dst, edge_vector = neighbor_list("ijD",
                                                a=atoms,
                                                cutoff=r_max)
edge_src2, edge_dst2, edge_vector2 = neighbor_list("ijD",
                                                a=atoms,
                                                cutoff=7)
#flag for incomplete octahedron [typical defects - tetradedral defect, vacancies]
oct_vol = []
oct_dst = []
oct_mask = []
oct_points = []
H_dist = []
for src in tqdm(Ni_idx):
    #print(src)
    dst = edge_dst[edge_src==src]
    vec = edge_vector[edge_src==src]
    #print(dst)
    #print(vec)
    mask = trim(src, dst, vec)
    points = vec[mask]
    oct_dst.append(dst[mask])
    oct_points.append(points)
    oct_mask.append(mask)
    mask = trim(src = src, 
                dst = edge_dst2[edge_src2==src],
                vec = edge_vector2[edge_src2==src], 
                species='H',
                minimum=1
               )
    try:
        oct_vol.append(octahedral_volume1(*points))
    except TypeError:
        #raise Exception("missing one octahedron here")
        #print("missing one octahedron here")
        #print(points)
        oct_vol.append(-1)
    try: 
        H_dist.append(np.linalg.norm(edge_vector2[edge_src2==src][mask]))
    except IndexError:
        H_dist.append(-1)

oct_vol =np.array(oct_vol)

np.savetxt('vol.csv', 
           np.vstack([Ni_idx, oct_vol]).T, 
           fmt=['%d', '%.4f'])

