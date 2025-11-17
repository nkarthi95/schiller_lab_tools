from skimage.draw import ellipsoid
import numpy as np
from scipy.ndimage import center_of_mass
from particles import fibonacci_sphere

def draw_ellipsoid_in_box(ellipseDims, boxDims, npart = 0, seedval = None):
    """
    Generate a voxelized ellipsoid centered within a 3D box and sample
    particle positions/orientations on or within the ellipsoid.

    Parameters
    ----------
    ellipseDims : array_like of shape (3,)
        Semi-axes of the ellipsoid ``(a, b, c)`` in voxel units.
    boxDims : array_like of shape (3,)
        Dimensions of the containing box. Must satisfy
        ``boxDims[i] >= 2*ellipseDims[i]`` to avoid negative padding.
    npart : int, optional
        Number of particle samples. Positions are generated from a
        Fibonacci-sphere distribution scaled by ``ellipseDims``.
        Default is 0.
    seedval : int or None, optional
        Seed for the random number generator used to create
        particle orientations. Default is None.

    Returns
    -------
    volume_field : ndarray of shape ``boxDims``
        Binary voxel array representing the ellipsoid centered in the box.
        Voxels inside the ellipsoid are 1, outside are 0.
    particle_positions : ndarray of shape (npart, 3)
        Cartesian positions of sampled particles, shifted so that the
        ellipsoid is centered within ``volume_field``.
    particle_orientations : ndarray of shape (npart, 3)
        Random unit vectors representing particle orientations.

    Notes
    -----
    The ellipsoid is generated at unit resolution by calling
    ``ellipsoid(a, b, c)``, then symmetrically padded to match
    ``boxDims``. Particle positions are assigned before orientation
    normalization. Particle orientations are uniform on the sphere.
    """
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