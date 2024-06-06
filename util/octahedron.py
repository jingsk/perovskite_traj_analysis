import numpy as np
#import matplotlib.pyplot as plt
from itertools import combinations

def tetrahedral_volume(p1, p2, p3, p4):
    u = p2 - p1
    v = p3 - p1
    w = p4 - p1
    volume = 1/6 * np.absolute((u @ np.cross(v,w)))
    return volume

def octahedral_volume1(p1, p2, p3, p4, p5, p6):
    points = np.vstack((p1, p2, p3, p4, p5, p6))
    top = points[np.argmax(points[:,2])]
    bottom = points[np.argmin(points[:,2])]
    
    plane = np.delete(points,
                      axis=0,
                      obj=[np.argmax(points[:,2]), 
                           np.argmin(points[:,2])])
    
    vol = []
    for triplet in combinations(plane, 3):
        vol.append(tetrahedral_volume(top,*triplet))
        vol.append(tetrahedral_volume(bottom,*triplet))
    #for 4C3 = 4 combinations and one top, one bottom:
    #there 8 tetrahedrons meaning volume is double-counted
    return 1/2 * np.sum(vol)

def octahedral_volume2(p1, p2, p3, p4, p5, p6):
    points = np.vstack((p1, p2, p3, p4, p5, p6))
    vol = []
    for quads in combinations(points, 4):
        vol.append(tetrahedral_volume(*quads))
    
    #6C4 = 15 combinations containing 12 tetrahedron, 3 planes:
    #after removing 3 planes, 12 tetrahedrons - volume is triple-counted
    return 1/3 * np.sum(np.array(sorted(vol))[3:])

def octahedral_volume_ref(b, h):
    return 2/3 * b**2 *h

