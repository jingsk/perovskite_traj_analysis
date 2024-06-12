from ase.io import read
from ase.geometry.analysis import Analysis
from ase.visualize import view
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def get_rdf(atoms):
    ana = Analysis(atoms)
    [(rdf, r)] = ana.get_rdf(rmax=4,
                 nbins=50, 
                 elements=['O', 'Ni'],
                 #elements=['O'],
                 return_dists=True,)
    return rdf, r


def plot_rdf(r_s, rdf_s, labels, save=False):
    #for i, ax in enumerate(axs.ravel()[0,-1]):
    for r, rdf, label in zip(r_s, rdf_s, labels):
        ax.plot(r, rdf, label=label)
        ax.set_xlabel(r'r ($\AA{}$)')
        ax.set_ylabel('g(r)')
        #ax.set_yscale()
        ax.set_ylim([-0.5,5])
    ax.legend(labels)

    if save:
        fig.tight_layout()
        fig.savefig('rdf_start_stop.png', dpi=300)


if __name__ == "__main__":
    traj = read(sys.argv[1], ':')
    timestep = 1 #ps
    time = timestep * np.arange(len(traj))
    slc = [0, 100, 200]
    traj = [traj[s] for s in slc]
    time = [time[s] for s in slc]
    rdf_s = []
    r_s =[]
    
    for atoms in tqdm(traj):
        rdf, r = get_rdf(atoms)
        rdf_s.append(rdf)
        r_s.append(r)
    
    labels = [f'{t:02f} ps' for t in time]
    fig, ax = plt.subplots(figsize = (5, 3))
    plot_rdf(r_s, rdf_s, labels, save=True)
    