import numpy as np
from numba import njit
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

def calculate_enclosed_voids(labelled_array, nvoids_total):
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

def calculate_voids(box, level = 0):
    array_bin = label(np.where(box > level, 1, 0))
    box_inv = ~array_bin
    regions, r = label(box_inv, return_num=True, connectivity=1)
    voids = calculate_enclosed_voids(regions, r)

    return voids

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

@njit
def gradient_upwind_3d(phi, indicator, dx):
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
        grad = gradient_upwind_3d(phi, sign_phi0, dx)
        phi_new = phi - dt*sign_phi0*(grad - 1)

        curr_err = np.mean(np.abs(phi_new - phi))
        errors.append(curr_err)
        phi = phi_new
        
        if curr_err < max_error:
            break

        curr_ite += 1
    
    return phi, errors

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