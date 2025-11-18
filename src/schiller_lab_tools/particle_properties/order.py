import numpy as np
import freud
from scipy.optimize import brute, fmin

def calculate_nematic_order(orientations, director=None):
  """
  Compute the nematic order parameter for a system of particle orientations.

  Parameters
  ----------
  orientations : ndarray of shape (N, D)
      Orientation vectors for ``N`` particles in ``D`` dimensions.
  director : array_like of shape (D,), optional
      Reference direction used to evaluate alignment. Defaults to the
      unit vector along the z-axis ``[0, 0, 1]``.

  Returns
  -------
  float
      Nematic order parameter. Values near 1 indicate strong alignment
      with the director; 0 corresponds to no net alignment; -0.5
      corresponds to orthogonal anti-alignment.

  Notes
  -----
  The computation uses ``freud.order.Nematic`` to evaluate alignment of
  particle orientations relative to the supplied director. The director
  is interpreted as the preferred axis of orientation, as in liquid-crystal
  or elongated-particle systems. Positive values indicate alignment along
  the director, negative values indicate alignment opposite or orthogonal
  to it.

  Examples
  --------
  >>> orientations = np.random.rand(100, 3)
  >>> director = [0, 0, 1]
  >>> calculate_nematic_order(orientations, director)
  """

  if not isinstance(director, np.ndarray):
      director = np.array(director)
  nematic = freud.order.Nematic(director)
  nematic.compute(orientations)
  return nematic.order

def calculate_ql(boxDims, positions, L=6, voronoi=True, average=False, weighted=False):
    """
    Compute the Steinhardt order parameter ``q_l`` for a system of particles.

    Parameters
    ----------
    boxDims : array_like of shape (3,)
        Dimensions of the simulation box along x, y, and z.
    positions : ndarray of shape (N, 3)
        Particle coordinates. ``N`` is the number of particles.
    L : int, optional
        Spherical harmonic order ``l`` used in the Steinhardt definition.
        Default is 6.
    voronoi : bool, optional
        If True, use a Voronoi-based neighborhood. In this mode,
        ``average`` is ignored and ``weighted`` is forced to True.
        Default is True.
    average : bool, optional
        Include second-shell contributions (Steinhardt ``\bar{q}_l`` variant).
        Only applied when ``voronoi`` is False. Default is False.
    weighted : bool, optional
        Weight contributions by neighbor distances. Only applied when
        ``voronoi`` is False. Default is False.

    Returns
    -------
    ndarray of shape (N,)
        The Steinhardt order parameter ``q_l`` for each particle.

    Notes
    -----
    When ``voronoi`` is True, neighbor lists are constructed from
    Voronoi tessellation. ``weighted`` is implicitly set to True and
    ``average`` is unused. When ``voronoi`` is False, neighbor lists are
    distance-based and the behaviors of ``average`` and ``weighted`` follow
    their explicit input values.

    Examples
    --------
    Compute ``q_l`` using Voronoi neighborhoods:

    >>> boxDims = [10.0, 10.0, 10.0]
    >>> positions = np.random.rand(100, 3) * 10.0
    >>> calculate_ql(boxDims, positions, L=6, voronoi=True)

    Compute ``q_l`` using distance-based neighbors:

    >>> calculate_ql(boxDims, positions, L=6,
    ...              voronoi=False, average=True, weighted=False)
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
    Compute the Steinhardt ``w_l`` order parameter for a system of particles.

    Parameters
    ----------
    boxDims : array_like of shape (3,)
        Dimensions of the simulation box along x, y, and z.
    positions : ndarray of shape (N, 3)
        Particle coordinates. ``N`` is the number of particles.
    L : int, optional
        Spherical harmonic order used in the ``w_l`` definition. Also
        determines the neighbor count for the computation. Default is 6.
    normalize : bool, optional
        If True, normalize the ``w_l`` values. Default is False.

    Returns
    -------
    ndarray of shape (N,)
        The Steinhardt ``w_l`` order parameter for each particle.

    Notes
    -----
    ``w_l`` differs from the corresponding ``q_l`` parameter in that it
    incorporates Wigner 3j symbols, providing a distinct measure of local
    symmetry. Neighbor identification is based on the number of neighbors
    implied by the chosen ``L`` value.

    Examples
    --------
    >>> boxDims = [10.0, 10.0, 10.0]
    >>> positions = np.random.rand(100, 3) * 10.0
    >>> calculate_wl(boxDims, positions, L=6, normalize=True)
    """

    if not isinstance(boxDims, np.ndarray):
        boxDims = np.array(boxDims)
    box = freud.box.Box(*boxDims)
    wl = freud.order.Steinhardt(L, wl = True, wl_normalize = normalize)
    sc_system = (box, positions)
    wl_sc = wl.compute(sc_system, neighbors={"num_neighbors": L}).particle_order
    return wl_sc

def calculate_smectic_order(particle_positions, nematic_director, layer_thickness_range):
    """
    Compute the smectic order parameter and optimal layer spacing for a system of particles.

    Parameters
    ----------
    particle_positions : ndarray of shape (N, 3)
        Positions of the particles. ``N`` is the number of particles.
    nematic_director : array_like of shape (3,)
        Nematic director used to project particle positions onto the layer-normal direction.
        The vector is normalized internally.
    layer_thickness_range : tuple or list
        Search interval for the candidate smectic layer thickness used in the brute-force
        optimization.

    Returns
    -------
    smectic_order : float
        Smectic order parameter evaluated at the optimal layer spacing. Values approach 1
        for strong layering and 0 for no layering.
    optimal_layer_thickness : float
        Value of the layer spacing ``d`` that maximizes the smectic ordering measure.

    Notes
    -----
    The smectic order parameter is computed by maximizing

        S(d) = | Σ exp( i * 2π * (director · r) / d ) | / N

    via a two-stage procedure:
    a brute-force search over ``layer_thickness_range`` followed by refinement with
    ``fmin`` (Nelder–Mead). The director is normalized before use.

    Examples
    --------
    >>> positions = np.random.rand(500, 3)
    >>> director = np.array([0, 0, 1])
    >>> calculate_smectic_order(positions, director, (1.0, 10.0))
    """
  
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