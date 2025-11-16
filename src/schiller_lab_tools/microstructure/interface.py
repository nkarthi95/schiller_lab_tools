import numpy as np
from numba import njit
from pyvista import PolyData, ImageData
from skimage.measure import marching_cubes, label
from skimage.transform import downscale_local_mean
from _csd_helper_calculation import calculate_genus_handles_surface_area
from _csd_helper_smoothing import boxcar_avg
from scipy.ndimage import convolve

def label_regions_hk(phi, filter=None):
    """
    Binarize the input field `phi` and label the largest connected region.

    This function processes the input array `phi`, typically representing the difference between two density 
    fields, and returns a binarized version where the largest connected region is labeled with 1 and all other 
    regions are labeled with 0. An optional filter function can be applied to `phi` before binarization. 
    The largest region is determined based on the size of the connected components.

    :param phi:
        A numpy array representing the difference between two density fields. The array can have any dimensionality 
        and typically contains scalar values encoding density differences.
    :type phi: numpy.ndarray

    :param filter:
        An optional function to filter the input data before binarization. The function should accept a numpy array 
        and return a binary mask of the same shape. If not provided, positive values in `phi` are treated as part 
        of the region of interest (labeled as 1), and non-positive values are labeled as 0.
    :type filter: function, optional

    :return:
        A binary numpy array of the same shape as `phi`, where the largest connected region (after filtering, if 
        provided) is labeled with 1, and all other regions are labeled with 0.
    :rtype: numpy.ndarray

    :note:
        - The function uses `morphology.label` from the `skimage` library to perform connected component labeling.
        - Connectivity is determined by the number of dimensions in `phi`.
        - The largest connected component is identified as the region with the highest count of connected points.

    :example:
        >>> import numpy as np
        >>> from skimage import morphology
        >>> phi = np.array([[0.1, -0.3, 0.4], [0.2, 0.5, -0.2], [-0.1, 0.3, 0.6]])
        >>> labeled_regions = label_regions_hk(phi)
        >>> print(labeled_regions)

        >>> filter_func = lambda x: np.where(x > 0.2, 1, 0)
        >>> labeled_regions_filtered = label_regions_hk(phi, filter=filter_func)
        >>> print(labeled_regions_filtered)
    """

    dims = len(phi.shape)
    phi_filter = np.zeros_like(phi)
    
    if filter is None:
        phi_filter = np.where(phi > 0, 1, 0)
    else:
        phi_filter = filter(phi)
        
    phi_label = label(phi_filter, connectivity=dims)

    labels, counts = np.unique(phi_label, return_counts=True)
    l = labels[1:][np.argsort(counts[1:])][-1]
    
    phi_label = np.where(phi_label == l, 1, 0)
    return phi_label

@njit()
def fill(field):
    """
    Fill zero values in a distribution using an averaging algorithm.

    This function iteratively fills zero values in a 3D density field by averaging the values from 
    neighboring cells. The process propagates from non-zero neighbors to zero values, using a weighted 
    average of the nearest and next-nearest neighbors. Periodic boundary conditions are applied to handle 
    edges of the field. The input array is not modified, and a new array with filled values is returned.

    :param field:
        A 3D array representing a distribution of densities (e.g., fluid densities). The array may contain 
        zero values that need to be filled using neighboring values. The original array is not modified.
    :type field: numpy.ndarray

    :return:
        A 3D array of the same shape as `field`, with zero values filled using the averaging algorithm.
    :rtype: numpy.ndarray

    :note:
        - Periodic boundary conditions are used, meaning the edges of the field wrap around to connect 
          with the opposite edges.
        - The averaging uses both nearest and next-nearest neighbors, applying a weight factor to the 
          next-nearest neighbors.
        - The filling process continues until all zero values are filled.
        - A copy of the input `field` is created for the filling process to ensure the original array remains unaltered.

    :example:
        >>> import numpy as np
        >>> field = np.random.random((5, 5, 5))
        >>> field[1, 1, 1] = 0  # introduce a zero value
        >>> filled_field = fill(field)
        >>> print(filled_field)
    """

    while (field.size - np.count_nonzero(field)):

        new = field.copy()

        for x, y, z in zip(*np.nonzero(field == 0)):

            # Neighbor indices with periodic boundary conditions
            xp = (x + 1) % field.shape[0]
            yp = (y + 1) % field.shape[1]
            zp = (z + 1) % field.shape[2]
            xm = (x - 1) % field.shape[0]
            ym = (y - 1) % field.shape[1]
            zm = (z - 1) % field.shape[2]

            # Values at nearest neighbors
            nn = [field[xp, y, z], field[xm, y, z], field[x, yp, z], 
                  field[x, ym, z], field[x, y, zp], field[x, y, zm]]
            nn = np.array(nn)

            # Values at next nearest neighbors
            nnn = [field[xp, yp, z], field[xm, ym, z], field[xp, ym, z], 
                   field[xm, yp, z], field[x, yp, zp], field[x, ym, zm], 
                   field[x, yp, zm], field[x, ym, zp], field[xp, y, zp], 
                   field[xm, y, zm], field[xm, y, zp], field[xp, y, zm]]
            nnn = np.array(nnn)

            cn = np.count_nonzero(nn)  # Number of filled nearest neighbors
            cnn = np.count_nonzero(nnn)  # Number of filled next nearest neighbors

            f = 1. / np.sqrt(2)  # Weight factor for next nearest neighbors
            w = 1. / (cn + f * cnn) if (cn + cnn) > 0 else 0

            # Calculate average density, zero if no filled neighbors
            new[x, y, z] = w * (np.sum(nn) + f * np.sum(nnn)) if (cn + cnn) > 0 else 0

        field = new.copy()

    return field

def interface_order(phi):
    """
    Compute the maximum eigenvalue of the orientation tensor for the system.

    This function calculates the orientation tensor based on the gradient of the order parameter 
    `phi` and returns the maximum eigenvalue of the tensor. The eigenvalue characterizes the order 
    of the interface in the system, providing a measure of the system's anisotropy and interface 
    alignment.

    :param phi:
        A 2D or 3D array representing the order parameter of the system. The field `phi` is typically 
        used to describe some phase or interface in the system whose orientation is being analyzed.
    :type phi: numpy.ndarray

    :return:
        The maximum eigenvalue of the orientation tensor, representing the degree of order or alignment 
        of the interface in the system.
    :rtype: float

    :note:
        - The function computes the gradient of `phi` and uses this to form the orientation tensor, 
          which is averaged over the spatial dimensions.
        - The orientation tensor is diagonalized, and the eigenvalues are computed using `np.linalg.eig`.
        - The maximum eigenvalue is used as a measure of the order of the interface, with higher values 
          indicating stronger alignment of the interface.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> order = interface_order(phi)
        >>> print(order)
    """
    dphi = np.gradient(phi)
    A = np.einsum('ixyz,jxyz->ijxyz', *[dphi]*2)
    traceA = np.einsum('iixyz->xyz', A)
    Q = np.average(A - traceA / 3 * np.expand_dims(np.eye(3), axis=(-3, -2, -1)), axis=(-3, -2, -1)) / np.average(traceA)
    ev, evec = np.linalg.eig(Q)
    return max(ev)

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

def get_mesh(analyte, boxDims, level = 0, 
             origin = (0,0,0), spacing = (1,1,1), 
             downscale = (1,1,1), method = "marching_cubes",
             order = "F"):

    test = downscale_local_mean(analyte, downscale)
    dimensions = tuple([boxDims[i]//downscale[i] for i in range(len(boxDims))])
    grid = ImageData(
        dimensions = dimensions,
        spacing = spacing,
        origin = origin,
    )

    mesh = grid.contour([level], test.flatten(order = order), method=method)
    return mesh

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