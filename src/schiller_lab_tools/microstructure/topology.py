import numpy as np
from numba import njit
from skimage.measure import label, euler_number
from skimage.measure import label, euler_number, marching_cubes, mesh_surface_area
from .interface import get_mesh, boxcar_avg
from scipy.ndimage import convolve
from pyvista import PolyData

# -----------------------------------------------------------
# Helpers: Void calculation through a periodic boundary
# -----------------------------------------------------------

def _calculate_enclosed_voids(labelled_array, nvoids_total):
    """
    Count the number of fully enclosed voids in a labeled 3D array.

    This function identifies and counts void regions that do not touch the boundary
    of the input volume. It assumes that void regions have been previously labeled
    using a connected-component labeling algorithm.

    Parameters
    ----------
    labelled_array : ndarray
        A 3D NumPy array where each connected void region is labeled with a unique
        integer (typically from `scipy.ndimage.label` or `skimage.measure.label`).
    nvoids_total : int
        The total number of labeled void regions in the array.

    Returns
    -------
    num_internal_voids : int
        The number of void regions that are completely enclosed within the volume
        (i.e., they do not touch any of the array boundaries).

    Notes
    -----
    - Background should be labeled as 0 in the `labelled_array`.
    - This function assumes 6-connectivity unless the labeling function was called with different connectivity.
    - A void is considered "enclosed" only if none of its voxels touch any of the six faces of the volume.
    """
    border_labels = set()
    border_labels.update(np.unique(labelled_array[0, :, :]))
    border_labels.update(np.unique(labelled_array[-1, :, :]))
    border_labels.update(np.unique(labelled_array[:, 0, :]))
    border_labels.update(np.unique(labelled_array[:, -1, :]))
    border_labels.update(np.unique(labelled_array[:, :, 0]))
    border_labels.update(np.unique(labelled_array[:, :, -1]))

    all_labels = set(range(1, nvoids_total + 1))
    internal_voids = all_labels - border_labels
    num_internal_voids = len(internal_voids)
    return num_internal_voids

def _calculate_voids(box, level = 0):
    array_bin = label(np.where(box > level, 1, 0))
    box_inv = ~array_bin
    regions, r = label(box_inv, return_num=True, connectivity=1)
    voids = _calculate_enclosed_voids(regions, r)

    return voids

# -------------------------------------------------------------------
# Helpers: Calculating genus, number of channels, voids, surface area
# -------------------------------------------------------------------

def _from_mesh(array, level, decimate_prop, epsilon = 0.5):
    """
    Estimate topological and geometric properties from an isosurface mesh using PyVista.

    This function extracts a surface mesh from a 3D scalar field using the marching cubes
    algorithm and calculates the genus, number of handles, number of enclosed voids, and 
    surface area. It uses mesh-based estimates of Euler characteristic and optionally 
    applies mesh decimation to reduce complexity.

    Parameters
    ----------
    array : ndarray
        A 3D NumPy array representing the scalar field or volume data.
    level : float
        The isosurface level (threshold) used to extract the surface mesh.
    decimate_prop : float, optional
        Proportion of the mesh to reduce during decimation (default is 0.8). This value 
        should be between 0 (no reduction) and 1 (maximum reduction).
    epsilon: float, optional
        Sets the boundary padding layer value. Constant value per level set to create a smooth
        transition to the edge of the box. Defaults to 0.5

    Returns
    -------
    g : float
        The genus of the surface, estimated from a mesh-based Euler-Poincaré formula.
    handles : float
        The number of handles in the structure, calculated as genus + number of enclosed voids.
    voids : int
        The number of fully enclosed voids (regions that do not touch the array boundary).
    surface_area : float
        The surface area of the extracted and decimated isosurface mesh.

    Notes
    -----
    - The mesh is generated using PyVista’s `ImageData.contour` method with marching cubes.
    - The function assumes unit-valued data where the isosurface separates material from void.
    - `calculate_enclosed_voids` is a required helper function that detects non-boundary-connected voids.
    - Edge count is estimated assuming triangular faces (`e = 1.5 * f`), which holds for manifold triangle meshes.

    """
    array_pad = np.pad(array, pad_width = 1, mode = 'constant', constant_values = level - epsilon)

    pmesh = get_mesh(array_pad, array_pad.shape, level = level)
    pmesh.decimate(decimate_prop, inplace = True)
    pmesh.triangulate(inplace = True)

    v = pmesh.n_points
    f = pmesh.n_faces_strict
    e = 1.5*f

    EP = v + f - e
    g = 1 - EP/2
    surface_area = pmesh.area

    voids = _calculate_voids(array, level = level)
    handles = g + voids

    return g, handles, voids, surface_area

def _from_region_props(array, level):
    """
    Compute topological and geometric properties of a level set surface from a 3D scalar field.

    This function calculates the genus, number of handles, number of enclosed voids,
    and surface area of the isosurface extracted from a 3D array at a given isovalue level.
    It uses region properties for Euler number estimation and marching cubes for surface extraction.

    Parameters
    ----------
    array : ndarray
        A 3D NumPy array representing the scalar field or volume data.
    level : float
        The isosurface level (threshold) used to extract the surface and compute region properties.
    step_size : int
        Step size used in the marching cubes algorithm. Smaller values yield higher resolution.

    Returns
    -------
    g : float
        The genus of the structure, calculated from the Euler-Poincare characteristic.
    handles : float
        The number of handles in the structure, calculated as genus + number of enclosed voids.
    voids : int
        The number of fully enclosed void regions that do not touch the boundary.
    surface_area : float
        The surface area of the extracted isosurface.

    Notes
    -----
    - The function assumes unit voxel spacing in all dimensions.
    - The input array should represent a scalar field where isosurfaces define material boundaries.
    - A helper function `calculate_enclosed_voids` is required to compute the number of enclosed voids.
    """

    array_bin = label(np.where(array > level, 1, 0))
    box = ~array_bin

    _, c = label(~box, return_num=True, connectivity=3)
    EP = euler_number(~box, connectivity=3)
    g = c - EP

    regions, r = label(box, return_num=True, connectivity=1)
    v = _calculate_voids(regions, r)

    h = g + v

    verts, faces, _, _ = marching_cubes(array, level = level)
    surface_area = mesh_surface_area(verts, faces)

    return g, h, v, surface_area

# --------------------------------------------------------------------------
# Helpers: Genus, channels, voids, surface area calculation for single level
# --------------------------------------------------------------------------

def calculate_genus_handles_surface_area(array, level = 0, method = 'region_props', decimate_prop = 0.95):
    """
    Computes the mathematical genus, number of handles/channels, and surface area 
    of an isosurface within a 3D NumPy array.

    The function applies the marching cubes algorithm to extract the surface at 
    the given isosurface level and computes:
    
    - **Genus (g)**: A topological invariant representing the number of holes in the surface.
    - **Handles (h)**: The number of independent loops or channels within the structure.
    - **Surface Area (A)**: The total surface area of the extracted isosurface.

    Parameters
    ----------
    array : numpy.ndarray
        A 3D NumPy array representing the input volumetric data.
    level : float, optional
        The isosurface level to extract from the array (default is 0).
    step_size : int, optional
        The step size for the marching cubes algorithm, controlling the resolution 
        of the extracted surface (default is 1).
    method: str, optional
        Specifying the method used to calculate the output parameters. Options are
        'region_props'(default) and 'mesh'.
    Returns
    -------
    tuple
        A tuple containing:
        - **g (float)**: The genus of the surface.
        - **handles (float)**: The number of handles or channels in the surface.
        - **surface_area (float)**: The computed surface area.

    Notes
    -----
    - The function uses `measure.marching_cubes` from `skimage` to extract the surface mesh.
    - `pv.PolyData` (PyVista) is used to analyze the mesh and compute edges.
    - The genus is calculated using Euler-Poincaré formula: `EP = V - E + F`, where
      `V` is the number of vertices, `E` is the number of edges, and `F` is the number of faces.
    - The number of voids (separate enclosed regions) is determined using `measure.label`.
    """
    g = 0
    handles = 0
    surface_area = 0

    if method == 'mesh':
        g, handles, voids, surface_area = _from_mesh(array, level, decimate_prop)
    elif method == 'region_props':
        g, handles, voids, surface_area = _from_region_props(array, level)
    else:
        raise ValueError(f"{method} is invalid. Please use either mesh or region_props options")

    return g, handles, voids, surface_area

# -------------------------------------------------------------------
# Main: CSD calculation over distance from interface
# -------------------------------------------------------------------

def calc_csd(phi_edt, l = 1, nbins = None, method = "region_props", decimate_prop = 0.7):
    """
    Computes the channel size distribution (CSD) in a binary fluid system.

    This function calculates the CSD by determining the genus and number of handles 
    of the structure across different normalized radii, following the method described 
    in the paper: https://doi.org/10.1016/j.actamat.2011.12.042.

    Parameters
    ----------
    phi_edt : numpy.ndarray
        A 3D NumPy array representing the distance transform of an order parameter of a binary fluid system.
    l : int, optional
        The number of grid points to average over when computing the distance transform. (Default is 3)
    nbins : int, optional
        Number of bins used to calculate topology of distance from the interface (default is None which sets bin width to 1).
    step_size : int, optional
        The step size used in the marching cubes algorithm to extract surface information (default is 4).
    method: str, optional
        Specifies which method to use for the genus calculation algorithm. Options are 'region_props'(default) and 'mesh'.
    Returns
    -------
    tuple
        A tuple containing:
        - **radii_norm (numpy.ndarray)**: Normalized radii values ranging from 0 to 1.
        - **gv (numpy.ndarray)**: Genus density values as a function of the normalized radii.
        - **hv (numpy.ndarray)**: Handle density values as a function of the normalized radii.
        - **voids (numpy.ndarray)**: Number of voids in the system as a function of the normalized radii.

    Notes
    -----
    - The genus (g) and number of handles (h) are calculated at different radii using 
      `calculate_genus_handles_surface_area()`, with normalized distances.
    - A boxcar averaging filter is applied based on a convolution.
    - The genus and handle densities are normalized by the system volume and surface area density.
    - The final values are computed symmetrically for positive values only. Structure is assumed to be symmetric.
    """

    if l is not None:
        if l == 1:
            edt_in = boxcar_avg(phi_edt, l = 1)
        else:
            kernel = np.ones((2*l + 1, 2*l + 1, 2*l + 1))
            edt_in = convolve(phi_edt, kernel, mode = "wrap")/np.prod(kernel.shape)
    else:
        edt_in = phi_edt

    V = np.prod(edt_in.shape)
    _, _, _, A = calculate_genus_handles_surface_area(edt_in, level = 0, method = "region_props")
    # print(A)
    sigma = A/V
    factor = V*np.power(sigma, 3)

    # lmax = 1/(sigma)
    # print(lmax)
    lmax = np.amin(edt_in.shape)//2
    if nbins is None:
        distance = np.arange(0, np.ceil(lmax), 1, dtype = float)
    else:
        distance = np.linspace(0, lmax, nbins, dtype = float)

    gv = np.zeros_like(distance)
    hv = np.zeros_like(distance)
    voids = np.zeros_like(distance)

    for i, c in enumerate(distance):
        if c <= edt_in.max():
            # print(c)
            g, h, v, A =calculate_genus_handles_surface_area(edt_in, c, method = method, decimate_prop=decimate_prop)
            # if g > 0:
            gv[i] += g#/factor
            hv[i] += h#/factor
            voids[i] = v
        else:
            continue

    return distance, gv, hv, voids

def curvature(phi, limit=(50, 1), step_size = 1):
    """
    Compute the Gaussian and mean curvatures, as well as the interface area, for a given order parameter.

    This function uses a marching cubes algorithm to extract an isosurface mesh from the input order parameter 
    `phi`. It then calculates the Gaussian and mean curvatures of the mesh and filters them based on specified 
    limits. The function returns the filtered curvatures and the total area of the interface.

    :param phi:
        A 2D or 3D array representing the order parameter of the system, typically used to describe a density 
        field or phase. The function generates a mesh of the isosurface of `phi` using a marching cubes algorithm.
    :type phi: numpy.ndarray

    :param limit:
        A tuple `(K_max, H_max)` where `K_max` and `H_max` specify the maximum absolute values for filtering the 
        Gaussian and mean curvatures, respectively. Only curvatures with absolute values less than these limits 
        will be returned. Default is `(50, 1)`.
    :type limit: tuple of float, optional

    :param step_size:
        An integer that specifies how coarse the squares to calculate the marching squared grid is.
    :type step_size: integer, optional

    :return:
        A tuple containing:
        - **K** (*numpy.ndarray*): The Gaussian curvature at each vertex of the isosurface, filtered based on the specified limit.
        - **H** (*numpy.ndarray*): The mean curvature at each vertex of the isosurface, filtered based on the specified limit.
        - **A** (*float*): The total area of the interface of the isosurface.
    :rtype: tuple

    :note:
        - The marching cubes algorithm is used to extract an isosurface from the input `phi` field, and the curvatures 
          are computed on the resulting mesh.
        - The curvatures are filtered by comparing their absolute values to the provided `limit`. 
        - The area of the interface is computed using the `pv.PolyData.area` method from the `pyvista` library.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10, 10))
        >>> K, H, A = curvature(phi, limit=(30, 0.5))
        >>> print(K.shape, H.shape, A)
    """
    # Use marching cubes to obtain isosurface mesh
    verts, faces, normals, values = marching_cubes(phi, 0, step_size = step_size)
    pmesh = PolyData(verts, np.insert(faces, 0, 3, axis=1))
    
    K = pmesh.curvature(curv_type="gaussian")
    H = pmesh.curvature(curv_type="mean")
    A = pmesh.area
    
    # Apply filtering based on the curvature limits
    filt_K = np.abs(K) < limit[0]
    filt_H = np.abs(H) < limit[1]

    return K[filt_K], H[filt_H], A