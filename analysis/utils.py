import numpy as np

def gaussian(x, mu, sig):
    return 1/sig/np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/(2*sig**2))