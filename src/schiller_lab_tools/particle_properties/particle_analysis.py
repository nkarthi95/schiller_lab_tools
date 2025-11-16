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

    :param phi: 
        A 3D numpy array representing the binary density field of two phases (fluid or otherwise). The marching 
        cubes algorithm is applied to identify the interface between the phases.
    :type phi: numpy.ndarray
    :param positions: 
        A 2D numpy array of shape (n, D), where `n` is the number of particles and `D` is the number of dimensions 
        of the system. Each row represents the position of a particle in the system.
    :type positions: numpy.ndarray
    :param orientations: 
        A 2D numpy array of shape (n, D), where `n` is the number of particles and `D` is the number of dimensions. 
        Each row represents the orientation vector of a particle.
    :type orientations: numpy.ndarray
    :param step_size: 
        The grid size for the marching cubes algorithm. A smaller value will produce more accurate results, but may 
        increase computation time. Default is 1.
    :type step_size: int, optional
    :param cutoff: 
        The maximum distance a particle can be from the interface to be considered for angle calculation. Particles 
        farther than this distance from the interface are excluded. Default is 7.9.
    :type cutoff: float, optional

    :return: 
        A tuple containing:
        - `theta`: A 1D numpy array of shape (m,) containing the angles (in degrees) between the particle orientations 
          and the normal to the interface for all particles that are within the specified `cutoff` distance from the interface.
        - `mask`: A 1D numpy array containing the indices of the particles that are not within the `cutoff` distance 
          from the interface.
    :rtype: tuple (numpy.ndarray, numpy.ndarray)

    :notes: 
        - The marching cubes algorithm is used to extract the interface (isosurface) from the `phi` field.
        - The angle between each particle's orientation and the normal to the interface is calculated using the dot product, 
          and the result is converted from radians to degrees.
        - Particles with a center-to-interface distance greater than `cutoff` are excluded from the angle calculation.
        - The output `theta` contains the angles in degrees, and the returned `mask` indicates which particles were excluded 
          based on their distance from the interface.

    :examples: 
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

    :param boxDims: 
        A 1D numpy array of length D representing the dimensions of the simulation box. The values 
        in the array correspond to the size of the box along each dimension (e.g., [Lx, Ly, Lz] for a 
        3D system).
    :type boxDims: numpy.ndarray
    :param positions: 
        A 2D numpy array of shape (N, D) representing the positions of the particles in the system, 
        where N is the number of particles and D is the number of dimensions (e.g., 3 for a 3D system).
    :type positions: numpy.ndarray

    :return: 
        A tuple containing:
        - `r_s`: A 1D numpy array of the radii (bin edges) used in the RDF calculation. These represent 
        the radial distances from a reference particle.
        - `g_r`: A 1D numpy array of the normalized radial distribution function, which represents the 
        density of particles as a function of distance from a reference particle.
    :return type: tuple (numpy.ndarray, numpy.ndarray)

    :notes: 
        - The `r_max` parameter is calculated as the ceiling of `np.min(boxDims) * np.sqrt(3) / 4`, 
        which provides an estimate for the maximum radial distance used in the RDF calculation.
        - The function uses the `freud.density.RDF` class from the freud library to perform the RDF computation.
        - The radial distribution function is normalized such that `g_r = 1` for an ideal gas.
        - The output `r_s` represents the bin edges, and `g_r` represents the density at each corresponding 
        radial distance.

    :examples: 
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

  :param orientations: 
      A 2D numpy array of shape (N, D) representing the orientations of the particles, where 
      N is the number of particles and D is the number of dimensions. Each row contains the 
      orientation vector of a single particle in the system.
  :type orientations: numpy.ndarray
  :param director: 
      A 1D array or list of length D representing the director vector along which the nematic 
      order is computed. This vector specifies the preferred direction of alignment in the system. 
      If not provided, the default is the unit vector along the z-axis, i.e., [0, 0, 1].
      :default: [0, 0, 1]
  :type director: list or numpy.ndarray, optional

  :return: 
      The nematic order parameter of the system, which quantifies the alignment of the particle 
      orientations with the director. Values of 1, 0, and -0.5 indicate perfect alignment, no alignment, 
      and opposite orthogonal alignment respectively.
  :return type: float

  :notes: 
      - The nematic order parameter is computed using the `freud.order.Nematic` class from the freud 
        library, which uses the orientation of the particles relative to the provided director.
      - This function assumes that the director is a vector that represents the axis of preferred alignment. 
        It is commonly used in systems such as liquid crystals or systems of elongated particles.
      - A positive nematic order indicates a preference for alignment along the director, while a negative 
        value would indicate an opposite alignment.

  :examples: 
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


def calculate_ql(boxDims, positions, L=6, voronoi = True, average = False, weighted = False):
    """
    .. function:: calculate_ql(boxDims, positions, L=6, voronoi=True, average=False, weighted=False)

      Calculate the Steinhardt order parameter :math:`q_l` for a given set of particles in a simulation box.

      :param boxDims: Dimensions of the simulation box. A list or NumPy array of floats with length 3, where each entry defines the size of the box along the x, y, and z axes.
      :type boxDims: list[float] or numpy.ndarray
      :param positions: Positions of the particle centers in the simulation box. A NumPy array of shape :math:`(N, 3)` where :math:`N` is the number of particles.
      :type positions: numpy.ndarray
      :param L: The order of the Steinhardt spherical harmonics to compute. Default is 6.
      :type L: int, optional
      :param voronoi: Whether to use the Voronoi neighborhood to define neighbor lists. If True, `average` and `weighted` are automatically set to False and True, respectively. Default is True.
      :type voronoi: bool, optional
      :param average: Whether to include contributions from the second neighbor shell in the calculation. Only applicable when `voronoi` is False. Default is False.
      :type average: bool, optional
      :param weighted: Whether to weight the order parameter calculation by the distance from each particle. Only applicable when `voronoi` is False. Default is False.
      :type weighted: bool, optional
      :returns: The Steinhardt order parameter :math:`q_l` for each particle.
      :rtype: numpy.ndarray

      .. note::
        - If `voronoi` is True:
          - The Voronoi neighborhood is used to define the neighbor list.
          - `average` is ignored, and `weighted` is set to True.
        - If `voronoi` is False:
          - The neighbor list is defined based on distance-based metrics using the `L` parameter.
          - The behavior of `average` and `weighted` depends on their respective input values.

      **Examples**

      Compute the order parameter with Voronoi neighbors:

      .. code-block:: python

        import numpy as np
        boxDims = [10.0, 10.0, 10.0]
        positions = np.random.rand(100, 3) * 10.0
        calculate_ql(boxDims, positions, L=6, voronoi=True)

      Compute the order parameter with distance-based neighbors:

      .. code-block:: python

        calculate_ql(boxDims, positions, L=6, voronoi=False, average=True, weighted=False)

    """  

    if not isinstance(boxDims, np.ndarray):
        boxDims = np.array(boxDims)
    box = freud.box.Box(*boxDims)
    sc_system = (box, positions)

    if voronoi == True:
      ql = freud.order.Steinhardt(L, weighted=True)
      voro = freud.locality.Voronoi()
      voronoi_cells = voro.compute((box, positions))
      neighbor_list = voronoi_cells.nlist
    else:
      ql = freud.order.Steinhardt(L, weighted=weighted, average=average)
      neighbor_list = {"num_neighbors": L}

    ql_sc = ql.compute(sc_system, neighbors=neighbor_list).particle_order

    return ql_sc


# In[ ]:


def calculate_wl(boxDims, positions, L=6, normalize=False):
    """
    Calculate the Steinhardt order parameter :math:`w_l` for each particle in a simulation box.

    This function computes the :math:`w_l` parameter, a variation of the Steinhardt order parameter 
    that incorporates Wigner 3j symbols for describing the local symmetry of particle arrangements. 
    The calculation can optionally include normalization.

    :param boxDims: Dimensions of the simulation box. A list or NumPy array of floats with length 3, 
                    where each value represents the size of the box along the x, y, and z axes.
    :type boxDims: list[float] or numpy.ndarray
    :param positions: Positions of the particle centers in the simulation box. A NumPy array of shape 
                      :math:`(N, 3)` where :math:`N` is the number of particles.
    :type positions: numpy.ndarray
    :param L: The order of the Steinhardt spherical harmonics to compute. This defines the number of 
              neighbors to consider in the computation. Default is 6.
    :type L: int, optional
    :param normalize: Whether to normalize the :math:`w_l` values. If True, the order parameter values 
                      are normalized. Default is False.
    :type normalize: bool, optional

    :returns: A 1D NumPy array of size :math:`N`, where each element represents the :math:`w_l` parameter 
              for the corresponding particle.
    :rtype: numpy.ndarray

    :note:
        - The :math:`w_l` parameter differs from :math:`q_l` as it uses Wigner 3j symbols, providing a 
          different characterization of local particle symmetry.
        - The neighbors used in the calculation are determined based on the number of neighbors specified 
          by the :math:`L` parameter.

    :example:
        >>> import numpy as np
        >>> boxDims = [10.0, 10.0, 10.0]  # Box dimensions
        >>> positions = np.random.rand(100, 3) * 10.0  # Random positions for 100 particles
        >>> L = 6  # Number of neighbors to use
        >>> wl_values = calculate_wl(boxDims, positions, L=L, normalize=True)
        >>> print(wl_values)  # Output the w_l values for each particle
    """

    if not isinstance(boxDims, np.ndarray):
        boxDims = np.array(boxDims)
    box = freud.box.Box(*boxDims)
    wl = freud.order.Steinhardt(L, wl = True, wl_normalize = normalize)
    sc_system = (box, positions)
    wl_sc = wl.compute(sc_system, neighbors={"num_neighbors": L}).particle_order
    return wl_sc

