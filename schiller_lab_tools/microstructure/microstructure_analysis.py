#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import scipy.fftpack as fft

from skimage import measure, morphology
from skimage import measure

from numba import njit

import pyvista as pv
import openpnm as op
import porespy as ps
# import taufactor as tau


# In[3]:


def structure_factor(phi):
    """
    Computes the structure factor of a given field using its Fourier transform.

    This function calculates the structure factor of a 2D or 3D field by performing 
    a Fourier transform and computing the intensity at each wavevector `k`. The 
    structure factor provides insight into the spatial frequency content of the field.

    Parameters
    ----------
    phi : numpy.ndarray
        A 2D or 3D array representing the field whose structure factor is to be computed. 
        The input array should be of real or complex values.

    Returns
    -------
    tuple
        A tuple containing:
        - k : numpy.ndarray
            The wavevectors corresponding to the Fourier transform frequencies. 
            This is a meshgrid of wavevectors along each dimension of the input field.
        - S : numpy.ndarray
            The structure factor, which is the intensity of the Fourier transform at each 
            wavevector `k`. This is calculated as the square of the magnitude of the Fourier 
            transform, normalized by the number of elements in the input array.

    Notes
    -----
    - The Fourier transform is computed using `fft.fftn`, and the field is mean-centered 
      before the transform to remove any zero-frequency component.
    - The structure factor is computed as `S(k) = (1/N) * |F(phi)(k)|^2`, where `N` is 
      the number of elements in the field and `F(phi)(k)` is the Fourier transform of the field.
    - The result is the distribution of intensities across different spatial frequencies.

    Examples
    --------
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


# In[4]:


def spherically_averaged_structure_factor(phi, binned=True):
    """
    Computes the spherically averaged structure factor of a field.

    This function calculates the structure factor of a given field using its Fourier transform 
    and averages the intensities over spherical shells in the wavevector space. It provides a 
    way to analyze the frequency distribution of the field's spatial structure. The function 
    can return either a binned or unbinned version of the structure factor.

    Parameters
    ----------
    phi : numpy.ndarray
        A 2D or 3D array representing the field whose structure factor is to be computed. 
        The input array should be of real or complex values.

    binned : bool, optional
        A boolean flag indicating whether to return the binned version of the structure factor. 
        If `True`, the result will be averaged over spherical bins. If `False`, the unbinned 
        version is returned. Default is `True`.

    Returns
    -------
    tuple
        A tuple containing:
        - k : numpy.ndarray
            The wavevectors corresponding to the Fourier transform frequencies, binned or 
            unbinned, depending on the `binned` flag.
        - S : numpy.ndarray
            The structure factor values at each wavevector `k`, either binned or unbinned.

    Notes
    -----
    - The function first computes the structure factor using `structure_factor`, then 
      computes the spherical radial distances from the wavevector grid.
    - If `binned` is `True`, the structure factor is averaged over spherical shells of 
      wavevectors, with each bin corresponding to a range of wavevector magnitudes.
    - If `binned` is `False`, the function returns the structure factor values for each 
      individual wavevector.
    - The radial distance from the origin in wavevector space is calculated using the 
      Euclidean norm of the wavevector `k`.

    Examples
    --------
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


# In[5]:


def spherical_first_moment(phi, binned=True):
    """
    Computes the first moment of the spherically averaged structure factor.

    This function calculates the first moment of the structure factor, which is used to 
    determine the characteristic length scale of the input order parameter `phi`. The 
    structure factor is first computed and optionally binned in spherical shells. The first 
    moment is then computed as the weighted average of the wavevector magnitudes.

    Parameters
    ----------
    phi : numpy.ndarray
        A 2D or 3D array representing the order parameter of the field. This array typically 
        represents some physical data whose spatial structure is being analyzed.

    binned : bool, optional
        A boolean flag indicating whether the structure factor should be averaged over spherical 
        shells. If `True`, the structure factor is binned and averaged. If `False`, the unbinned 
        structure factor is used. Default is `True`.

    Returns
    -------
    float
        The first moment of the spherically averaged structure factor, which is used to estimate 
        the characteristic length scale of the field represented by `phi`.

    Notes
    -----
    - The first moment is calculated as the weighted average of the wavevector magnitudes, with 
      the structure factor values serving as weights.
    - The result of this function can be interpreted as a characteristic length scale of the 
      spatial structure in `phi`.

    Examples
    --------
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


# In[6]:


def second_moment(phi):
    """
    Computes the second moment of the structure factor.

    This function calculates the second moment of the structure factor, which is a measure of 
    the spread or variance of the wavevector distribution. The second moment is computed using 
    the wavevectors and the structure factor, and it quantifies the spatial distribution of the 
    field's spatial frequencies. The result is a 3-element array representing the variance in 
    each spatial direction ([x, y, z]).

    Parameters
    ----------
    phi : numpy.ndarray
        A 2D or 3D array representing the order parameter of the field. This array represents the 
        spatial configuration of the system whose structure factor is to be analyzed.

    Returns
    -------
    numpy.ndarray
        A 1D array of floats representing the second moment of the structure factor in the 
        [x, y, z] directions. This represents the variance of the spatial frequencies in each 
        direction.

    Notes
    -----
    - The second moment is calculated as a weighted average of the squared wavevectors, with 
      the structure factor values serving as weights.
    - This function uses the output from the `structure_factor` function to calculate the second 
      moment.

    Examples
    --------
    >>> import numpy as np
    >>> phi = np.random.random((10, 10))
    >>> k2 = second_moment(phi)
    >>> print(k2)

    """
    k, S = structure_factor(phi)
    k2 = np.sum(k * k * S[..., np.newaxis], axis=(0, 1, 2)) / np.sum(S)
    return k2


# In[7]:


def interface_order(phi):
    """
    Computes the maximum eigenvalue of the orientation tensor for the system.

    This function calculates the orientation tensor based on the gradient of the order parameter 
    `phi` and returns the maximum eigenvalue of the tensor. The eigenvalue characterizes the order 
    of the interface in the system, providing a measure of the system's anisotropy and interface 
    alignment.

    Parameters
    ----------
    phi : numpy.ndarray
        A 2D or 3D array representing the order parameter of the system. The field `phi` is typically 
        used to describe some phase or interface in the system whose orientation is being analyzed.

    Returns
    -------
    float
        The maximum eigenvalue of the orientation tensor, representing the degree of order or alignment 
        of the interface in the system.

    Notes
    -----
    - The function computes the gradient of `phi` and uses this to form the orientation tensor, which 
      is averaged over the spatial dimensions.
    - The orientation tensor is diagonalized, and the eigenvalues are computed using `np.linalg.eig`.
    - The maximum eigenvalue is used as a measure of the order of the interface, with higher values indicating 
      stronger alignment of the interface.

    Examples
    --------
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


# In[8]:


def curvature(phi, limit=(50, 1)):
    """
    Computes the Gaussian and mean curvatures, as well as the interface area, for a given order parameter.

    This function uses a marching cubes algorithm to extract an isosurface mesh from the input order parameter 
    `phi`. It then computes the Gaussian and mean curvatures of the mesh and filters the curvatures based on 
    specified limits. The function returns the filtered curvatures and the area of the interface.

    Parameters
    ----------
    phi : numpy.ndarray
        A 2D or 3D array representing the order parameter of the system, typically used to describe a density 
        field or phase. The function generates a mesh of the isosurface of `phi` using a marching cubes algorithm.

    limit : tuple of two floats, optional
        A tuple `(K_max, H_max)` where `K_max` and `H_max` specify the maximum absolute values for filtering the 
        Gaussian and mean curvatures, respectively. Only curvatures with absolute values less than these limits 
        will be returned. The default value `(50, 1)` means that curvatures are filtered using these thresholds, 
        and all curvature values are considered within the specified ranges.

    Returns
    -------
    tuple
        A tuple containing:
        - K : numpy.ndarray
            The Gaussian curvature at each vertex of the isosurface, filtered based on the specified limit.
        - H : numpy.ndarray
            The mean curvature at each vertex of the isosurface, filtered based on the specified limit.
        - A : float
            The area of the interface of the isosurface.

    Notes
    -----
    - The marching cubes algorithm is used to extract an isosurface from the input `phi` field, and the curvatures 
      are computed on the resulting mesh.
    - The curvatures are filtered by comparing their absolute values to the provided `limit`. 
    - The area of the interface is computed using the `pv.PolyData.area` method from the `pyvista` library.

    Examples
    --------
    >>> import numpy as np
    >>> phi = np.random.random((10, 10, 10))
    >>> K, H, A = curvature(phi, limit=(30, 0.5))
    >>> print(K.shape, H.shape, A)

    """
    # Use marching cubes to obtain isosurface mesh
    verts, faces, normals, values = measure.marching_cubes(phi, 0)
    pmesh = pv.PolyData(verts, np.insert(faces, 0, 3, axis=1))
    
    K = pmesh.curvature(curv_type="gaussian")
    H = pmesh.curvature(curv_type="mean")
    A = pmesh.area
    
    # Apply filtering based on the curvature limits
    filt_K = np.abs(K) < limit[0]
    filt_H = np.abs(H) < limit[1]

    return K[filt_K], H[filt_H], A


# In[ ]:


@njit()
def fill(field):
    """
    Fills the zero values in a distribution using an averaging algorithm.

    This function iteratively fills in zero values in a 3D density field by averaging the values from 
    neighboring cells. The filling process propagates from the non-zero neighbors towards the zero values, 
    using a weighted average of the nearest and next nearest neighbors. The function uses periodic boundary 
    conditions to handle the edges of the field, and it continues until all zero values are filled. 
    The original data in `field` is not overwritten, and a new array with the filled values is returned.

    Parameters
    ----------
    field : numpy.ndarray
        A 3D array representing a distribution of densities (e.g., fluid densities). The array may contain 
        zero values that need to be filled using neighboring values. The input array remains unchanged by 
        this function.

    Returns
    -------
    numpy.ndarray
        A 3D array of the same shape as `field`, where the zero values have been filled using the 
        averaging algorithm. The filled array is returned without modifying the original input.

    Notes
    -----
    - The algorithm uses periodic boundary conditions, meaning that the edges of the field wrap around 
      to connect with the opposite edges.
    - The averaging is done using both nearest and next nearest neighbors, with a weight factor applied 
      to the next nearest neighbors.
    - The function stops once all zero values are filled.
    - The function does not modify the original `field`, instead creating a copy of it for the filling process.
    
    Examples
    --------
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


# In[11]:


def label_regions_hk(phi, filter=None):
    """
    Binarizes the input field `phi` and labels the largest connected region.

    This function processes the input array `phi`, which generally encodes the difference between two density 
    fields, and returns a binarized version where the largest connected region is labeled with 1 and all other 
    regions are labeled with 0. The binarization can be customized using an optional filter function that is 
    applied to `phi` before the labeling. The largest region is determined based on the size of the connected components.

    Parameters
    ----------
    phi : numpy.ndarray
        A numpy array representing the difference between two density fields. This array can be of any dimensionality 
        and typically encodes scalar values that represent density differences.

    filter : function, optional
        A lambda or function to filter the input data before binarization. The function should accept a numpy array 
        and return a binary mask of the same shape. If not provided, the function will simply binarize `phi` by 
        setting all positive values to 1 and all non-positive values to 0.

    Returns
    -------
    numpy.ndarray
        A binary numpy array where the largest connected region (after applying the filter if provided) is labeled 
        with 1 and all other regions are labeled with 0.

    Notes
    -----
    - The function uses `morphology.label` from the `skimage` library to perform connected component labeling, 
      where the connectivity is determined by the number of dimensions in `phi`.
    - The largest connected component is determined by the size of the connected regions, and the region with 
      the highest count is labeled as 1.
    - If no filter is provided, the function treats positive values in `phi` as part of the region of interest 
      (labeled as 1) and non-positive values as background (labeled as 0).

    Examples
    --------
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


# In[ ]:


def get_pn(phi, filter=None, sigma=0.4, r_max=8, voxel_size=1, boundary_width=3, parallel={'cores':8,'divs':[2, 2, 2],'overlap':8}):
    """
    Constructs a pore network model from a binary density field using the Sub-Network of an Oversegmented 
    Watershed (SNOW) method.

    This function uses the SNOW algorithm to identify the pore network from a given binary field `phi` 
    that encodes the difference between two density fields. The function first binarizes the field using 
    `label_regions_hk` and then applies the SNOW algorithm to segment the structure into a pore network. 
    The method includes several parameters that control the accuracy, size of pores, and parallelization 
    strategy for the computation. The output is a pore network model that can be further analyzed.

    Parameters
    ----------
    phi : numpy.ndarray
        A binary numpy array representing the difference between two density fields. The regions of interest
        in `phi` are labeled, and the network model is constructed from the binary regions.

    filter : function, optional
        A lambda or function to filter the input data before binarization. The function should accept a numpy array
        and return a binary mask of the same shape. If not provided, the function will binarize `phi` by setting
        all positive values to 1 and all non-positive values to 0.

    sigma : float, optional
        A smoothing parameter for the SNOW algorithm. Default is 0.4. The higher the value, the more the image is 
        smoothed before segmentation.

    r_max : int, optional
        Maximum pore radius for the pore network model. Default is 8. It defines the largest pore size that will
        be considered in the pore network.

    voxel_size : float, optional
        The size of a single voxel in the 3D grid. Default is 1. The voxel size should be set based on the physical
        dimensions of the input `phi` field.

    boundary_width : int, optional
        The width of the boundary layer for segmentation. Default is 3. This helps define the region near the boundary 
        of the system where the model may be less accurate.

    parallel : dict, optional
        A dictionary specifying the parallelization strategy. It includes:
            - 'cores' : int, the number of CPU cores to use (default is 8).
            - 'divs' : list of 3 ints, the number of divisions along each spatial axis (default is [2, 2, 2]).
            - 'overlap' : int, the overlap between subdomains during parallel computation (default is 8).
        The parallelization settings should be tuned for the system and often work best for square box decompositions.

    Returns
    -------
    pn : openpnm.Network
        A pore network model constructed from the labeled regions in `phi`. The model represents the connectivity 
        of pores and throats, which can be analyzed for transport and structural properties.

    Notes
    -----
    - This function is designed for 3D systems, where the smallest defined volume has a radius of 8.
    - The parallelization strategy should be adjusted depending on the system's size and the number of available CPU cores.
    - The SNOW algorithm is an effective method for segmenting complex porous structures, and the pore network model 
      can be used for simulations of fluid flow and other processes within the material.

    Examples
    --------
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


# In[ ]:


def taufactor_tortuosity(phi, filter=None):
    """
    Calculates the tortuosity of a system along each axis and the average tortuosity across all axes.

    This function computes the tortuosity factor for the binary regions in the input array `phi` along each 
    spatial axis ([x, y, z]) individually. It first binarizes `phi` using the `label_regions_hk` function (with 
    an optional filter) and then calculates the tortuosity along each axis. The tortuosity along each axis is 
    calculated using the `tau.PeriodicSolver` solver, and the average tortuosity across all axes is also returned.

    Parameters
    ----------
    phi : numpy.ndarray
        A numpy array representing the difference between two density fields, typically encoding scalar values
        that describe the structure of a material or system.

    filter : function, optional
        A lambda or function to filter the input data before binarization. The function should accept a numpy array
        and return a binary mask of the same shape. If not provided, the function will binarize `phi` by setting
        all positive values to 1 and all non-positive values to 0.

    Returns
    -------
    numpy.ndarray
        A numpy array of length 4 containing the tortuosity values along each axis and the average tortuosity, [tx, ty, tz, ta].

    Notes
    -----
    - Tortuosity is a measure of the complexity of a path or structure, and is often used to quantify the 
      degree of "twistiness" of interfaces or material structures.
    - The function computes tortuosity along the x, y, and z axes individually and averages the results to provide
      a global tortuosity factor.
    - The `tau.PeriodicSolver` is used to compute the tortuosity factor for each axis, and the periodic boundary 
      conditions are assumed by the solver.

    Examples
    --------
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
        s = tau.PeriodicSolver(profile_in)
        s.solve()
        out.append(s.tau.item())
        axes = np.roll(axes, shift=1)
        profile_in = np.transpose(phi_bin, axes)
    
    out.append((out[0] + out[1] + out[2]) / 3)
    return np.array(out)

