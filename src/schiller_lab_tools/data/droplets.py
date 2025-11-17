from skimage.draw import ellipsoid
import numpy as np
from scipy.ndimage import center_of_mass
from particles import fibonacci_sphere

def draw_ellipsoid_in_box(ellipseDims, boxDims, npart = 0, seedval = None):
    a,b,c = ellipseDims
    volume_field = ellipsoid(a, b, c)
    diffs = boxDims - volume_field.shape
    pad_width = [[0,0], [0,0], [0,0]]
    for i in range(diffs.size):
        pad_width[i][0] = diffs[i]//2
        pad_width[i][1] = diffs[i]//2+diffs[i]%2
    volume_field = np.pad(volume_field, pad_width)

    particle_positions = fibonacci_sphere(npart, R=ellipseDims)
    particle_positions += np.array(center_of_mass(volume_field))[np.newaxis, :]

    rng = np.random.default_rng(seedval)
    particle_orientations = rng.random(size = (npart, 3))
    particle_orientations /= np.linalg.norm(particle_orientations, axis = 1)[:, np.newaxis]

    return volume_field, particle_positions, particle_orientations