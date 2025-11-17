import numpy as np
from numba import njit
from pyvista import ImageData
from skimage.measure import marching_cubes, label
from skimage.transform import downscale_local_mean
from scipy.ndimage import distance_transform_edt

# -----------------------------------------------------------
# Helpers: Binarizing a bolume field
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# Helpers: Distance transforms
# -----------------------------------------------------------

@njit
def _gradient_upwind_3d(phi, indicator, dx):
    nx, ny, nz = phi.shape
    grad = np.empty((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = i%nx
                y = j%ny
                z = k%nz

                # Forward/backward differences in x
                x_forward = (i+1)%nx
                x_backward = (i-1)%nx
                phi_x_forward = (phi[x_forward,y,z] - phi[x,y,z])/dx
                phi_x_backward = (phi[x,y,z] - phi[x_backward,y,z])/dx
                phi_x_p = np.maximum(phi_x_backward, 0)**2 + np.minimum(phi_x_forward, 0)**2
                phi_x_m = np.minimum(phi_x_backward, 0)**2 + np.maximum(phi_x_forward, 0)**2

                # Forward/backward differences in y
                y_forward = (j+1)%ny
                y_backward = (j-1)%ny
                phi_y_forward = (phi[x, y_forward,z] - phi[x,y,z])/dx
                phi_y_backward = (phi[x,y,z] - phi[x, y_backward,z])/dx
                phi_y_p = np.maximum(phi_y_backward, 0)**2 + np.minimum(phi_y_forward, 0)**2
                phi_y_m = np.minimum(phi_y_backward, 0)**2 + np.maximum(phi_y_forward, 0)**2

                # Forward/backward differences in z
                z_forward = (k+1)%nz
                z_backward = (k-1)%nz
                phi_z_forward = (phi[x,y,z_forward] - phi[x,y,z])/dx
                phi_z_backward = (phi[x,y,z] - phi[x,y,z_backward])/dx
                phi_z_p = np.maximum(phi_z_backward, 0)**2 + np.minimum(phi_z_forward, 0)**2
                phi_z_m = np.minimum(phi_z_backward, 0)**2 + np.maximum(phi_z_forward, 0)**2

                # Magnitude of gradient
                if indicator[x,y,z] >= 0:
                    grad[x,y,z] = np.sqrt(phi_x_p + phi_y_p + phi_z_p)
                else:
                    grad[x,y,z] = np.sqrt(phi_x_m + phi_y_m + phi_z_m)
    
    return grad

def reinitialize_3d(phi0, dx=1.0, dt=0.1, iterations=1000, max_error = 1e-8, epsilon = 1e-6):
    phi = phi0.copy()
    sign_phi0 = phi0/np.sqrt(phi0**2 + epsilon**2)
    errors = []
    
    curr_ite = 0
    curr_err = 1

    while curr_ite <= iterations:
        grad = _gradient_upwind_3d(phi, sign_phi0, dx)
        phi_new = phi - dt*sign_phi0*(grad - 1)

        curr_err = np.mean(np.abs(phi_new - phi))
        errors.append(curr_err)
        phi = phi_new
        
        if curr_err < max_error:
            break

        curr_ite += 1
    
    return phi, errors

def create_euclidean_distance_transform(phi):
    """
    Computes the Euclidean distance transform of a 3D emulsion surface.

    The function calculates the signed Euclidean distance of each point in the 
    input array `phi` to the interface between phases. Positive distances 
    correspond to points in the phase where `phi >= 0`, while negative distances 
    correspond to points where `phi <= 0`.

    Parameters
    ----------
    phi : numpy.ndarray
        A 3D NumPy array representing the emulsion surface.

    Returns
    -------
    numpy.ndarray
        A 3D NumPy array of the same shape as `phi`, where each element contains 
        the Euclidean distance to the nearest interface. Positive values indicate 
        distances inside the `phi >= 0` region, and negative values indicate 
        distances inside the `phi <= 0` region.

    Notes
    -----
    - The function uses `distance_transform_edt` from `scipy.ndimage` to compute 
      the Euclidean distance transform.
    - The interface is determined by binarizing the input array into two regions:
      one where `phi >= 0` and another where `phi <= 0`.
    - Signed distances are assigned accordingly, with positive distances inside 
      the `phi >= 0` region and negative distances inside the `phi <= 0` region.
    """
    phi_edt = np.empty(phi.shape)

    # phi_bin_pos = label_regions_hk(phi, lambda x: np.where(x >= 0, 1, 0))
    # phi_bin_neg = label_regions_hk(phi, lambda x: np.where(x <= 0, 1, 0))
    phi_bin_pos = np.where(phi >= 0, 1, 0)
    phi_bin_neg = np.where(phi < 0, 1, 0)

    phi_edt_pos = distance_transform_edt(phi_bin_pos)
    phi_edt_neg = distance_transform_edt(phi_bin_neg)

    idxs = np.where(phi >= 0)
    phi_edt[idxs] = phi_edt_pos[idxs]
    idxs = np.where(phi < 0)
    phi_edt[idxs] = -phi_edt_neg[idxs]

    return phi_edt

# -------------------------------------------------------------
# Helpers: Volumetric data based interface smoothing algorithms
# -------------------------------------------------------------

@njit
def boxcar_avg(data, l = 1):
    """
    Performs a boxcar averaging on a 3D NumPy array.

    The function computes the average value of neighboring cells within a given 
    window size `l` for each element in the input array. It handles three cases:
    interior portions, edges, and corners, using the modulo operator for periodic 
    boundary conditions.

    Parameters
    ----------
    data : numpy.ndarray
        A 3D NumPy array representing the input data to be averaged.
    l : int, optional
        The number of cells to average over in each direction (default is 1).

    Returns
    -------
    numpy.ndarray
        A 3D NumPy array with the averaged values.

    Notes
    -----
    - The function uses periodic boundary conditions to handle edges and corners.
    - The function does not modify the input array.
    - It utilizes a simple brute-force approach for averaging.
    """
    nx, ny, nz = data.shape
    smoothed = np.empty(data.shape)
    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):      
                sum_data = 0.0

                sum_data += data[i,j,k]
                # nearest neighbor x
                sum_data += data[(i-1)%nx, j, k]
                sum_data += data[(i+1)%nx, j, k]
                # nearest neighbor y
                sum_data += data[i, (j-1)%ny, k]
                sum_data += data[i, (j+1)%ny, k]
                # nearest neighbor z
                sum_data += data[i, j, (k-1)%nz]
                sum_data += data[i, j, (k+1)%nz]
                
                smoothed[i, j, k] = sum_data/7
    
    return smoothed

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

# -------------------------------------------------------------
# Helpers: Order parameter of interface
# -------------------------------------------------------------

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

# -------------------------------------------------------------
# Helpers: Meshing the interface at a level
# -------------------------------------------------------------

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

# ---------------------------------------------------------------------
# Helpers: Calculating the angle of a set of particles to the interface
# ---------------------------------------------------------------------

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

    v, f, n, vals = marching_cubes(phi, 0, step_size=step_size)  # verts, faces, normals, values

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