import numpy as np
from freud.density import RDF

def calculate_rdf(boxDims, positions):
    """
    Compute the radial distribution function (RDF) for a particle system.

    Parameters
    ----------
    boxDims : ndarray of shape (D,)
        Simulation box dimensions along each axis.
    positions : ndarray of shape (N, D)
        Particle coordinates. ``N`` is the number of particles and ``D`` is
        the spatial dimension.

    Returns
    -------
    r_s : ndarray
        Radii (bin edges) used in the RDF calculation.
    g_r : ndarray
        Normalized radial distribution function evaluated at the radii.
        ``g_r = 1`` corresponds to an ideal gas.

    Notes
    -----
    The maximum radius is computed as
    ``r_max = ceil(min(boxDims) * sqrt(3) / 4)``, used to define the RDF
    cutoff. The computation uses ``freud.density.RDF`` to evaluate pair
    statistics and return normalized densities. ``r_s`` corresponds to the
    bin edges and ``g_r`` to the RDF values for each bin.

    Examples
    --------
    >>> boxDims = np.array([10.0, 10.0, 10.0])
    >>> positions = np.random.rand(1000, 3) * boxDims
    >>> r_s, g_r = calculate_rdf(boxDims, positions)
    >>> r_s[:5]
    >>> g_r[:5]
    """


    L = np.amin(boxDims)
    r_max = int(np.ceil(np.min(boxDims) * np.sqrt(3) / 4))
    rdf = RDF(bins=r_max, r_max=r_max)
    rdf.compute(system=(boxDims, positions), reset=False)
    r_s = rdf.bin_edges[:-1]
    g_r = rdf.rdf
    return r_s, g_r

def calculate_geodesic_pdf(mesh, particle_positions, boxDims):
    npart, _ = particle_positions.shape

    geodesic_distances = np.zeros((npart, npart))
    mesh_locations = np.zeros((npart), dtype = int)

    for i, part in enumerate(particle_positions):
        mesh_locations[i] = mesh.find_closest_point(part)

    for i in range(npart):
        part1_pos = mesh_locations[i]
        for j in range(i, npart):
            if i == j:
                geodesic_distances[i,j] = 0.0
            else:
                part2_pos = mesh_locations[j]
                geo_dist = mesh.geodesic_distance(part1_pos, part2_pos)
                geodesic_distances[i, j] = geo_dist
                geodesic_distances[j, i] = geo_dist
    
    r_max = geodesic_distances.max()*2
    bins = np.arange(1, 256, 1)
    counts, bin_edges = np.histogram(geodesic_distances, bins, range=(0, r_max))
    volume = 4/3*np.pi*(bin_edges[1:]**3-bin_edges[:-1]**3)
    rdf = counts/npart/volume*np.prod(boxDims)/npart
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2

    return bin_centers, rdf