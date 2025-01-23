#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
import freud
from scipy.optimize import curve_fit
from skimage import transform, measure


# In[2]:


def calculate_average_cos_interface_normal(phi, positions, orientations, step_size=1, cutoff=7.9):
    """
    Calculates the angle between particle orientations and the interface normal for particles near the interface.

    This function uses the marching cubes algorithm to identify the interface in a 3D field represented by `phi` 
    and calculates the angle between the particle orientation and the normal to the interface for particles that 
    are within a specified distance (`cutoff`) from the interface. It returns the angles for particles near the 
    interface and the number of particles that are not within the cutoff distance.

    Parameters
    ----------
    phi : numpy.ndarray
        A 3D numpy array representing the binary density field of two phases (fluid or otherwise). The marching 
        cubes algorithm is applied to identify the interface between the phases.

    positions : numpy.ndarray
        A 2D numpy array of shape (n, D), where `n` is the number of particles and `D` is the number of dimensions 
        of the system. Each row represents the position of a particle in the system.

    orientations : numpy.ndarray
        A 2D numpy array of shape (n, D), where `n` is the number of particles and `D` is the number of dimensions. 
        Each row represents the orientation vector of a particle.

    step_size : int, optional
        The grid size for the marching cubes algorithm. A smaller value will produce more accurate results, but may 
        increase computation time. Default is 1.

    cutoff : float, optional
        The maximum distance a particle can be from the interface to be considered for angle calculation. Particles 
        farther than this distance from the interface are excluded. Default is 7.9.

    Returns
    -------
    theta : numpy.ndarray
        A 1D numpy array of shape (m,) containing the angles (in degrees) between the particle orientations and the 
        normal to the interface for all particles that are within the specified `cutoff` distance from the interface. 
        Particles further than the `cutoff` are excluded.

    mask : numpy.ndarray
        A 1D numpy array containing the indices of the particles that are not within the `cutoff` distance from the 
        interface. This array provides the indices of particles that were excluded from the angle calculation.

    Notes
    -----
    - The marching cubes algorithm is used to extract the interface (isosurface) from the `phi` field.
    - The angle between each particle's orientation and the normal to the interface is calculated using the dot product, 
      and the result is converted from radians to degrees.
    - Particles with a center-to-interface distance greater than `cutoff` are excluded from the angle calculation.
    - The output `theta` contains the angles in degrees, and the returned `mask` indicates which particles were excluded 
      based on their distance from the interface.

    Examples
    --------
    >>> phi = np.random.randn(100, 100, 100)  # Example binary field
    >>> positions = np.random.rand(10, 3) * 100  # Random positions for 10 particles
    >>> orientations = np.random.rand(10, 3)  # Random orientations for 10 particles
    >>> theta, mask = calculate_average_cos_interface_normal(phi, positions, orientations)
    >>> print(theta)  # Angles of particles near the interface
    >>> print(mask)   # Indices of particles not near the interface
    """
    v, f, n, vals = measure.marching_cubes(phi, 0, step_size=step_size)  # verts, faces, normals, values

    distances = np.zeros(orientations.shape[0])
    theta = np.zeros(orientations.shape[0])

    for i in range(orientations.shape[0]):
        curr_pos = positions[i]
        part_norm = orientations[i]

        part_to_int_distance = np.linalg.norm(curr_pos - v, axis=-1)
        idx_norm = np.argsort(part_to_int_distance)[0]
        distances[i] = part_to_int_distance[idx_norm]
        int_norm = n[idx_norm]

        angle = np.dot(part_norm, int_norm) / (np.linalg.norm(part_norm) * np.linalg.norm(int_norm))
        angle = np.arccos(angle) * 180 / np.pi
        theta[i] = 180 - angle if angle > 90 else angle

    mask = np.where(distances >= cutoff)
    theta = np.delete(theta, mask)

    return theta, mask[0]


# In[3]:


def calculate_rdf(boxDims, positions):
    """
    Calculate the radial distribution function (RDF) for a system of particles.

    This function computes the radial distribution function (g(r)) for a system of particles, 
    using the freud library's RDF implementation. The RDF is calculated by determining the 
    density of particles as a function of distance from a reference particle. The function 
    returns the radii `r_s` and the normalized densities `g_r` for the system.

    Parameters
    ----------
    boxDims : numpy.ndarray
        A 1D numpy array of length D representing the dimensions of the simulation box. The values 
        in the array correspond to the size of the box along each dimension (e.g., [Lx, Ly, Lz] for a 
        3D system).

    positions : numpy.ndarray
        A 2D numpy array of shape (N, D) representing the positions of the particles in the system, 
        where N is the number of particles and D is the number of dimensions (e.g., 3 for a 3D system).

    Returns
    -------
    r_s : numpy.ndarray
        A 1D numpy array of the radii (bin edges) used in the RDF calculation. These represent 
        the radial distances from a reference particle.

    g_r : numpy.ndarray
        A 1D numpy array of the normalized radial distribution function, which represents the 
        density of particles as a function of distance from a reference particle.

    Notes
    -----
    - The `r_max` parameter is calculated as the ceiling of `np.min(boxDims) * np.sqrt(3) / 4`, 
      which provides an estimate for the maximum radial distance used in the RDF calculation.
    - The function uses the `freud.density.RDF` class from the freud library to perform the RDF computation.
    - The radial distribution function is normalized such that `g_r = 1` for an ideal gas.
    - The output `r_s` represents the bin edges, and `g_r` represents the density at each corresponding 
      radial distance.

    Examples
    --------
    >>> boxDims = np.array([10.0, 10.0, 10.0])  # Simulation box dimensions in 3D
    >>> positions = np.random.rand(1000, 3) * boxDims  # Random particle positions in 3D
    >>> r_s, g_r = calculate_rdf(boxDims, positions)
    >>> print(r_s)  # Radii used in the RDF calculation
    >>> print(g_r)  # Normalized radial distribution function values
    """
    L = np.amin(boxDims)
    r_max = int(np.ceil(np.min(boxDims) * np.sqrt(3) / 4))
    rdf = freud.density.RDF(bins=r_max, r_max=r_max)
    rdf.compute(system=(boxDims, positions), reset=False)
    r_s = rdf.bin_edges[:-1]
    g_r = rdf.rdf
    return r_s, g_r


# In[4]:


def calculate_nematic_order(orientations, director=[0, 0, 1]):
    """
    Calculate the nematic order parameter for a system of particles.

    This function computes the nematic order parameter, which measures the degree of alignment 
    of the particles in a given direction, referred to as the director. The director is typically 
    chosen to be a vector that represents the preferred direction of alignment in the system, 
    and the nematic order quantifies how well the particle orientations are aligned with this 
    direction.

    Parameters
    ----------
    orientations : numpy.ndarray
        A 2D numpy array of shape (N, D) representing the orientations of the particles, where 
        N is the number of particles and D is the number of dimensions. Each row contains the 
        orientation vector of a single particle in the system.

    director : list or numpy.ndarray, optional, default=[0, 0, 1]
        A 1D array or list of length D representing the director vector along which the nematic 
        order is computed. This vector specifies the preferred direction of alignment in the system. 
        If not provided, the default is the unit vector along the z-axis, i.e., [0, 0, 1].

    Returns
    -------
    nematic_order : float
        The nematic order parameter of the system, which quantifies the alignment of the particle 
        orientations with the director. Values of 1, 0 and -0.5 indicate perfect alignment, no alignment and opposite
        orthogonal alignment respectively.

    Notes
    -----
    - The nematic order parameter is computed using the `freud.order.Nematic` class from the freud 
      library, which uses the orientation of the particles relative to the provided director.
    - This function assumes that the director is a vector that represents the axis of preferred alignment. 
      It is commonly used in systems such as liquid crystals or systems of elongated particles.
    - A positive nematic order indicates a preference for alignment along the director, while a negative 
      value would indicate an opposite alignment.

    Examples
    --------
    >>> orientations = np.random.rand(100, 3)  # 100 random particle orientations in 3D
    >>> director = [0, 0, 1]  # Director along the z-axis
    >>> nematic_order = calculate_nematic_order(orientations, director)
    >>> print(nematic_order)  # Nematic order parameter of the system
    """
    if not isinstance(director, np.ndarray):
        director = np.array(director)
    nematic = freud.order.Nematic(director)
    nematic.compute(orientations)
    return nematic.order


# In[5]:


def calculate_minkowski_q(boxDims, positions, L=6):
    """
    Calculate the Minkowski structure metric of order L for each particle in the system.

    This function computes the Minkowski structure metric (also known as the Steinhardt order parameter)
    of order L for each particle in the system. It uses the Voronoi tessellation of the system to determine 
    the local environment of each particle, and then calculates the Steinhardt order parameter to describe 
    the local symmetry of the particle arrangements.

    Parameters
    ----------
    boxDims : list or numpy.ndarray
        A 1D array or list of length D representing the dimensions of the simulation box, where D is the 
        number of dimensions. The array should contain the box lengths along each dimension (e.g., 
        [Lx, Ly, Lz] for a 3D system).
    
    positions : numpy.ndarray
        A 2D numpy array of shape (N, D) where N is the number of particles, and D is the number of dimensions.
        Each row represents the position of a single particle in the system.

    L : int, optional, default=6
        The Steinhardt order parameter to compute. The order parameter quantifies the local symmetry of 
        the particle arrangement, and higher orders (e.g., L=6) correspond to more detailed descriptions 
        of the local symmetry. The default is L=6, which corresponds to the typical hexagonal or crystalline 
        symmetry for many systems.

    Returns
    -------
    ql_sc : numpy.ndarray
        A 1D numpy array of size N, where each element represents the Minkowski structure metric (Steinhardt 
        order parameter) of order L for the corresponding particle in the system.

    Notes
    -----
    - The function uses the `freud.locality.Voronoi` class to compute the Voronoi tessellation and the 
      `freud.order.Steinhardt` class to compute the Steinhardt order parameter.
    - The Voronoi tessellation is used to identify the local neighborhood of each particle, and the Steinhardt 
      order parameter describes how ordered that neighborhood is.
    - The returned array contains the Steinhardt order parameter for each particle, quantifying the local 
      symmetry around that particle. A higher value indicates a more ordered arrangement in the local environment.
    
    Examples
    --------
    >>> boxDims = [10, 10, 10]  # Box dimensions for a 3D system
    >>> positions = np.random.rand(100, 3) * boxDims  # 100 random particle positions in 3D
    >>> L = 6  # Steinhardt order parameter to calculate
    >>> ql_sc = calculate_minkowski_q(boxDims, positions, L)
    >>> print(ql_sc)  # Minkowski structure metric for each particle
    """
    if not isinstance(boxDims, np.ndarray):
        boxDims = np.array(boxDims)
    box = freud.box.Box(*boxDims)
    voro = freud.locality.Voronoi()
    ql = freud.order.Steinhardt(L, weighted=True)
    sc_system = (box, positions)
    voronoi_cells = voro.compute((box, positions))
    ql_sc = ql.compute(sc_system, neighbors=voronoi_cells.nlist).particle_order
    return ql_sc

