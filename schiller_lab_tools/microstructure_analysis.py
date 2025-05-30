#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.fftpack as fft
from scipy.ndimage import distance_transform_edt, convolve

from skimage import measure, morphology, transform

from numba import njit

import pyvista as pv
import openpnm as op
import porespy as ps
import torch
import taufactor as tau


# In[2]:


def structure_factor(phi):
    """
    Compute the structure factor of a given field using its Fourier transform.

    This function calculates the structure factor of a 2D or 3D field by performing 
    a Fourier transform and computing the intensity at each wavevector `k`. The 
    structure factor provides insight into the spatial frequency content of the field.

    :param phi:
        A 2D or 3D array representing the field whose structure factor is to be computed. 
        The input array should be of real or complex values.
    :type phi: numpy.ndarray

    :return:
        A tuple containing:
        - **k** (*numpy.ndarray*): The wavevectors corresponding to the Fourier transform 
          frequencies. This is a meshgrid of wavevectors along each dimension of the input field.
        - **S** (*numpy.ndarray*): The structure factor, which is the intensity of the 
          Fourier transform at each wavevector `k`. This is calculated as the square of the 
          magnitude of the Fourier transform, normalized by the number of elements in the input array.
    :rtype: tuple

    :note:
        - The Fourier transform is computed using `fft.fftn`, and the field is mean-centered 
          before the transform to remove any zero-frequency component.
        - The structure factor is computed as `S(k) = (1/N) * |F(phi)(k)|^2`, where `N` is 
          the number of elements in the field and `F(phi)(k)` is the Fourier transform of the field.
        - The result is the distribution of intensities across different spatial frequencies.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> k, S = structure_factor(phi)
        >>> print(k.shape, S.shape)
    """
    k = np.stack(np.meshgrid(*tuple([2*np.pi*fft.fftfreq(L) for L in phi.shape]), indexing='ij'), axis=-1)
    N = np.prod(phi.shape)
    phi_k = fft.fftn(phi - np.mean(phi))
    S = 1 / N * np.abs(phi_k) ** 2
    return k, S


# In[3]:


def spherically_averaged_structure_factor(phi, binned=True):
    """
    Compute the spherically averaged structure factor of a field.

    This function calculates the structure factor of a given field using its Fourier transform 
    and averages the intensities over spherical shells in wavevector space. It provides a way 
    to analyze the frequency distribution of the field's spatial structure. The function can 
    return either a binned or unbinned version of the structure factor.

    :param phi:
        A 2D or 3D array representing the field whose structure factor is to be computed. 
        The input array should be of real or complex values.
    :type phi: numpy.ndarray

    :param binned:
        A boolean flag indicating whether to return the binned version of the structure factor. 
        If `True`, the result will be averaged over spherical bins. If `False`, the unbinned 
        version is returned. Default is `True`.
    :type binned: bool, optional

    :return:
        A tuple containing:
        - **k** (*numpy.ndarray*): The wavevectors corresponding to the Fourier transform 
          frequencies, binned or unbinned, depending on the `binned` flag.
        - **S** (*numpy.ndarray*): The structure factor values at each wavevector `k`, either 
          binned or unbinned.
    :rtype: tuple

    :note:
        - The function first computes the structure factor using `structure_factor`, then computes 
          the spherical radial distances from the wavevector grid.
        - If `binned` is `True`, the structure factor is averaged over spherical shells of 
          wavevectors, with each bin corresponding to a range of wavevector magnitudes.
        - If `binned` is `False`, the function returns the structure factor values for each 
          individual wavevector.
        - The radial distance from the origin in wavevector space is calculated using the 
          Euclidean norm of the wavevector `k`.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> k, S = spherically_averaged_structure_factor(phi, binned=True)
        >>> print(k.shape, S.shape)

        >>> k, S = spherically_averaged_structure_factor(phi, binned=False)
        >>> print(k.shape, S.shape)
    """
    L = np.min(np.shape(phi))
    k, S = structure_factor(phi)
    k1 = np.linalg.norm(k, axis=-1).flatten()
    S1 = S.flatten()
    kmin = 2*np.pi/L  # sampling frequency
    where = np.s_[:]  # Select all k values (no upper bound)
    
    if not binned:
        ku, inv, counts = np.unique(k1[where], return_inverse=True, return_counts=True)
        sums = np.zeros(len(ku), dtype=S1.dtype)
        np.add.at(sums, inv, S1[where])
        return ku, sums/counts
    else:
        bins = np.arange(L//2+1) * kmin  # Define bin edges
        shells = np.histogram(k1[where], bins, weights=S1[where])[0]
        counts = np.histogram(k1[where], bins)[0]
        return (bins[:-1] + bins[1:]) / 2, shells / counts


# In[4]:


def spherical_first_moment(phi, binned=True):
    """
    Compute the first moment of the spherically averaged structure factor.

    This function calculates the first moment of the structure factor, which is used to 
    determine the characteristic length scale of the input order parameter `phi`. The 
    structure factor is first computed and optionally binned in spherical shells. The first 
    moment is then computed as the weighted average of the wavevector magnitudes.

    :param phi:
        A 2D or 3D array representing the order parameter of the field. This array typically 
        represents some physical data whose spatial structure is being analyzed.
    :type phi: numpy.ndarray

    :param binned:
        A boolean flag indicating whether the structure factor should be averaged over spherical 
        shells. If `True`, the structure factor is binned and averaged. If `False`, the unbinned 
        structure factor is used. Default is `True`.
    :type binned: bool, optional

    :return:
        The first moment of the spherically averaged structure factor, which is used to estimate 
        the characteristic length scale of the field represented by `phi`.
    :rtype: float

    :note:
        - The first moment is calculated as the weighted average of the wavevector magnitudes, with 
          the structure factor values serving as weights.
        - The result of this function can be interpreted as a characteristic length scale of the 
          spatial structure in `phi`.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> k1 = spherical_first_moment(phi, binned=True)
        >>> print(k1)

        >>> k1 = spherical_first_moment(phi, binned=False)
        >>> print(k1)
    """
    k, S = spherically_averaged_structure_factor(phi, binned=binned)
    k1 = np.sum(k * S) / np.sum(S)
    return k1


# In[5]:


def second_moment(phi):
    """
    Compute the second moment of the structure factor.

    This function calculates the second moment of the structure factor, which is a measure of 
    the spread or variance of the wavevector distribution. The second moment quantifies the 
    spatial distribution of the field's spatial frequencies and returns a 3-element array 
    representing the variance in each spatial direction ([x, y, z]).

    :param phi:
        A 2D or 3D array representing the order parameter of the field. This array represents the 
        spatial configuration of the system whose structure factor is to be analyzed.
    :type phi: numpy.ndarray

    :return:
        A 1D array of floats representing the second moment of the structure factor in the 
        [x, y, z] directions. This represents the variance of the spatial frequencies in each 
        direction.
    :rtype: numpy.ndarray

    :note:
        - The second moment is calculated as a weighted average of the squared wavevectors, with 
          the structure factor values serving as weights.
        - This function uses the output from the `structure_factor` function to calculate the second 
          moment.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> k2 = second_moment(phi)
        >>> print(k2)
    """
    k, S = structure_factor(phi)
    k2 = np.sum(k * k * S[..., np.newaxis], axis=(0, 1, 2)) / np.sum(S)
    return k2


# In[6]:


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


# In[7]:


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
    verts, faces, normals, values = measure.marching_cubes(phi, 0, step_size = step_size)
    pmesh = pv.PolyData(verts, np.insert(faces, 0, 3, axis=1))
    
    K = pmesh.curvature(curv_type="gaussian")
    H = pmesh.curvature(curv_type="mean")
    A = pmesh.area
    
    # Apply filtering based on the curvature limits
    filt_K = np.abs(K) < limit[0]
    filt_H = np.abs(H) < limit[1]

    return K[filt_K], H[filt_H], A


# In[8]:


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


# In[ ]:


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
        
    phi_label = morphology.label(phi_filter, connectivity=dims)

    labels, counts = np.unique(phi_label, return_counts=True)
    l = labels[1:][np.argsort(counts[1:])][-1]
    
    phi_label = np.where(phi_label == l, 1, 0)
    return phi_label


# In[10]:


def get_pn(phi, filter=None, sigma=0.4, r_max=8, voxel_size=1, boundary_width=3, parallel={'cores': 8, 'divs': [2, 2, 2], 'overlap': 8}):
    """
    Constructs a pore network model from a binary density field using the Sub-Network of an Oversegmented 
    Watershed (SNOW) method.

    This function uses the SNOW algorithm to identify the pore network from a given binary field `phi` 
    that encodes the difference between two density fields. The function first binarizes the field using 
    `label_regions_hk` and then applies the SNOW algorithm to segment the structure into a pore network. 
    The method includes several parameters that control the accuracy, size of pores, and parallelization 
    strategy for the computation. The output is a pore network model that can be further analyzed.

    :param phi: 
        A binary numpy array representing the difference between two density fields. The regions of interest
        in `phi` are labeled, and the network model is constructed from the binary regions.
    :type phi: numpy.ndarray
    :param filter: 
        A lambda or function to filter the input data before binarization. The function should accept a numpy array
        and return a binary mask of the same shape. If not provided, the function will binarize `phi` by setting
        all positive values to 1 and all non-positive values to 0.
    :type filter: function, optional
    :param sigma: 
        A smoothing parameter for the SNOW algorithm. Default is 0.4. The higher the value, the more the image is 
        smoothed before segmentation.
    :type sigma: float, optional
    :param r_max: 
        Maximum pore radius for the pore network model. Default is 8. It defines the largest pore size that will
        be considered in the pore network.
    :type r_max: int, optional
    :param voxel_size: 
        The size of a single voxel in the 3D grid. Default is 1. The voxel size should be set based on the physical
        dimensions of the input `phi` field.
    :type voxel_size: float, optional
    :param boundary_width: 
        The width of the boundary layer for segmentation. Default is 3. This helps define the region near the boundary 
        of the system where the model may be less accurate.
    :type boundary_width: int, optional
    :param parallel: 
        A dictionary specifying the parallelization strategy. It includes:
            - 'cores': int, the number of CPU cores to use (default is 8).
            - 'divs': list of 3 ints, the number of divisions along each spatial axis (default is [2, 2, 2]).
            - 'overlap': int, the overlap between subdomains during parallel computation (default is 8).
        The parallelization settings should be tuned for the system and often work best for square box decompositions.
    :type parallel: dict, optional

    :return: 
        A pore network model constructed from the labeled regions in `phi`. The model represents the connectivity 
        of pores and throats, which can be analyzed for transport and structural properties.
    :rtype: openpnm.Network

    :notes: 
        - This function is designed for 3D systems, where the smallest defined volume has a radius of 8.
        - The parallelization strategy should be adjusted depending on the system's size and the number of available CPU cores.
        - The SNOW algorithm is an effective method for segmenting complex porous structures, and the pore network model 
          can be used for simulations of fluid flow and other processes within the material.

    :examples:
        >>> import numpy as np
        >>> phi = np.random.randn(100, 100, 100)  # Example input data
        >>> pn = get_pn(phi)
        >>> print(pn)  # Output will be a pore network model
    """

    phi_label = label_regions_hk(phi, filter)
    snow_output = ps.networks.snow2(phi_label, phase_alias={1: 'blue', 0: 'red'}, 
                                    boundary_width=boundary_width, accuracy='standard', 
                                    voxel_size=voxel_size, sigma=sigma, r_max=r_max,
                                    parallelization=parallel)
    pn = op.io.network_from_porespy(snow_output.network)
    return pn


# In[11]:


def taufactor_tortuosity(phi, filter=None, device = "cpu", convergence_criteria = 0.05):
    """
    Calculates the tortuosity of a system along each axis and the average tortuosity across all axes.

    This function computes the tortuosity factor for the binary regions in the input array `phi` along each 
    spatial axis ([x, y, z]) individually. It first binarizes `phi` using the `label_regions_hk` function (with 
    an optional filter) and then calculates the tortuosity along each axis. The tortuosity along each axis is 
    calculated using the `tau.PeriodicSolver` solver, and the average tortuosity across all axes is also returned.

    :param phi: 
        A numpy array representing the difference between two density fields, typically encoding scalar values
        that describe the structure of a material or system.
    :type phi: numpy.ndarray
    :param filter: 
        A lambda or function to filter the input data before binarization. The function should accept a numpy array
        and return a binary mask of the same shape. If not provided, the function will binarize `phi` by setting
        all positive values to 1 and all non-positive values to 0.
    :type filter: function, optional
    :param device:
        Tells pytorch whether to use a cpu or gpu implementation. Default is cpu
    :type device: str, optional
    :param convergence_criteria:
        Tells the solver when to stop computation. Not sure how this value is used in the code, but the higher it is the
        faster convergence takes place.
    :type device: int, optional

    :return: 
        A numpy array of length 4 containing the tortuosity values along each axis and the average tortuosity, 
        [tx, ty, tz, ta].
    :rtype: numpy.ndarray

    :notes: 
        - Tortuosity is a measure of the complexity of a path or structure, and is often used to quantify the 
          degree of "twistiness" of interfaces or material structures.
        - The function computes tortuosity along the x, y, and z axes individually and averages the results to provide
          a global tortuosity factor.
        - The `tau.PeriodicSolver` is used to compute the tortuosity factor for each axis, and the periodic boundary 
          conditions are assumed by the solver.

    :examples:
        >>> import numpy as np
        >>> phi = np.random.randn(10, 10, 10)  # Example input data
        >>> result = taufactor_tortuosity(phi)
        >>> print(result)  # Output will be an array with tortuosity values for each axis and the average
    """

    axes = np.arange(0, len(phi.shape), 1, dtype=int)
    phi_bin = label_regions_hk(phi, filter)
    out = []
    profile_in = phi_bin.copy()
    for i in range(len(phi_bin.shape)):
        s = tau.PeriodicSolver(profile_in, device=torch.device(device))
        s.solve(conv_crit=convergence_criteria)
        out.append(s.tau.item())
        axes = np.roll(axes, shift=1)
        profile_in = np.transpose(phi_bin, axes)
    
    out.append((out[0] + out[1] + out[2]) / 3)
    return np.array(out)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def calculate_voids(box, level = 0):
    array_bin = measure.label(np.where(box > level, 1, 0))
    box_inv = ~array_bin
    regions, r = measure.label(box_inv, return_num=True, connectivity=1)
    voids = calculate_enclosed_voids(regions, r)

    return voids


# In[ ]:


def meshify(volume_field, level, decimate_prop = 0.7, method = 'marching_cubes'):
    grid = pv.ImageData(dimensions = volume_field.shape)
    pmesh = grid.contour([level], volume_field.flatten(order = "F"), method = method)
    pmesh.decimate(decimate_prop, inplace = True)
    pmesh.triangulate(inplace = True)
    return pmesh


# In[ ]:


def from_mesh(array, level, decimate_prop, epsilon = 0.5):
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
    pmesh = meshify(array_pad, level, decimate_prop = decimate_prop, method = 'marching_cubes')

    v = pmesh.n_points
    f = pmesh.n_faces_strict
    e = 1.5*f

    EP = v + f - e
    g = 1 - EP/2
    surface_area = pmesh.area

    voids = calculate_voids(array, level = level)
    handles = g + voids

    return g, handles, voids, surface_area


# In[ ]:


def from_region_props(array, level):
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

    array_bin = measure.label(np.where(array > level, 1, 0))
    box = ~array_bin

    _, c = measure.label(~box, return_num=True, connectivity=3)
    EP = measure.euler_number(~box, connectivity=3)
    g = c - EP

    regions, r = measure.label(box, return_num=True, connectivity=1)
    v = calculate_voids(regions, r)

    h = g + v

    verts, faces, _, _ = measure.marching_cubes(array, level = level)
    surface_area = measure.mesh_surface_area(verts, faces)

    return g, h, v, surface_area


# In[ ]:


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
        g, handles, voids, surface_area = from_mesh(array, level, decimate_prop)
    elif method == 'region_props':
        g, handles, voids, surface_area = from_region_props(array, level)
    else:
        raise ValueError(f"{method} is invalid. Please use either mesh or region_props options")

    return g, handles, voids, surface_area


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

