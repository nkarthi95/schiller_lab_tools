import numpy as np
from interface import label_regions_hk
import openpnm as op
import porespy as ps
import torch
import taufactor as tau


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