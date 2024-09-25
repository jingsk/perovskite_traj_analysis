import numpy as np

def gaussian(x, mu, sig):
    return 1/sig/np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/(2*sig**2))

def linear_fit(t,msd):
    [m,b], cov = np.polyfit(t, msd, 1, cov=True)
    x_grid = np.linspace(t[0], t[-1], 5*t.size, endpoint=True)
    return x_grid, x_grid * m + b

def d_to_acf(r_t0_t_it0_j0, r_cut = 5.0,grid_size=201):
    r_grid = np.linspace(0,r_cut, grid_size)
    acf_t = np.zeros([r_t0_t_it0_j0.shape[0],grid_size])
    for t, r_it0_j0 in enumerate(r_t0_t_it0_j0):
        mask1 = r_it0_j0<r_cut
        mask2 = r_it0_j0>0.
        hist, bin_edges = np.histogram(r_it0_j0[np.logical_and(mask1,mask2)], r_grid)
        bin_center = (bin_edges[:-1] + bin_edges[1:])/2
        for mu, count in zip(bin_center,hist):
             acf_t[t] += 1/mu**2 *count * gaussian(r_grid, mu, sig=0.1)

    acf_t /= (r_t0_t_it0_j0.shape[1])*(r_t0_t_it0_j0.shape[1]-1)
    acf_t = acf_t.sum(axis=0)
    return r_grid, acf_t