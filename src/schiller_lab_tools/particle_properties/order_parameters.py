import numpy as np
import freud
from scipy.optimize import brute, fmin

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

def calculate_smectic_order(particle_positions, nematic_director, layer_thickness_range):
    def calc_smectic(d, director, pos):
        return -(
            np.absolute(np.sum(np.exp(np.dot(director, pos.T) * 2 * np.pi * 1j / d)))
        ) / len(pos)
    
    director = nematic_director/np.linalg.norm(nematic_director)
    
    maximal_d = brute(
        calc_smectic,  # function to optimize
        ranges=(
            layer_thickness_range,
        ),  # range of values for optimization, these depend on the size of the particles in the direction of orientation
        args=(director, particle_positions),  # arguments to pass to calc_smectic
        finish=fmin,  # use Nelder-Mead to refine the brute force result
    )[0]

    smec = -calc_smectic(maximal_d, director, particle_positions)

    return smec, maximal_d