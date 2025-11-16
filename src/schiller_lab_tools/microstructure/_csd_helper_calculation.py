from interface import get_mesh
import numpy as np
from numba import njit
from skimage.measure import label, euler_number, marching_cubes, mesh_surface_area
from _csd_helper_smoothing import calculate_voids

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

    pmesh = get_mesh(array_pad, array_pad.shape, level = level)
    pmesh.decimate(decimate_prop, inplace = True)
    pmesh.triangulate(inplace = True)

    v = pmesh.n_points
    f = pmesh.n_faces_strict
    e = 1.5*f

    EP = v + f - e
    g = 1 - EP/2
    surface_area = pmesh.area

    voids = calculate_voids(array, level = level)
    handles = g + voids

    return g, handles, voids, surface_area

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

    array_bin = label(np.where(array > level, 1, 0))
    box = ~array_bin

    _, c = label(~box, return_num=True, connectivity=3)
    EP = euler_number(~box, connectivity=3)
    g = c - EP

    regions, r = label(box, return_num=True, connectivity=1)
    v = calculate_voids(regions, r)

    h = g + v

    verts, faces, _, _ = marching_cubes(array, level = level)
    surface_area = mesh_surface_area(verts, faces)

    return g, h, v, surface_area

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