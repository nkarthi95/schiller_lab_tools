import numpy as np
from frued.density import RDF

def calculate_rdf(boxDims, positions):
    """
    Calculate the radial distribution function (RDF) for a system of particles.

    This function computes the radial distribution function (g(r)) for a system of particles, 
    using the freud library's RDF implementation. The RDF is calculated by determining the 
    density of particles as a function of distance from a reference particle. The function 
    returns the radii `r_s` and the normalized densities `g_r` for the system.

    :param boxDims: 
        A 1D numpy array of length D representing the dimensions of the simulation box. The values 
        in the array correspond to the size of the box along each dimension (e.g., [Lx, Ly, Lz] for a 
        3D system).
    :type boxDims: numpy.ndarray
    :param positions: 
        A 2D numpy array of shape (N, D) representing the positions of the particles in the system, 
        where N is the number of particles and D is the number of dimensions (e.g., 3 for a 3D system).
    :type positions: numpy.ndarray

    :return: 
        A tuple containing:
        - `r_s`: A 1D numpy array of the radii (bin edges) used in the RDF calculation. These represent 
        the radial distances from a reference particle.
        - `g_r`: A 1D numpy array of the normalized radial distribution function, which represents the 
        density of particles as a function of distance from a reference particle.
    :return type: tuple (numpy.ndarray, numpy.ndarray)

    :notes: 
        - The `r_max` parameter is calculated as the ceiling of `np.min(boxDims) * np.sqrt(3) / 4`, 
        which provides an estimate for the maximum radial distance used in the RDF calculation.
        - The function uses the `freud.density.RDF` class from the freud library to perform the RDF computation.
        - The radial distribution function is normalized such that `g_r = 1` for an ideal gas.
        - The output `r_s` represents the bin edges, and `g_r` represents the density at each corresponding 
        radial distance.

    :examples: 
        >>> boxDims = np.array([10.0, 10.0, 10.0])  # Simulation box dimensions in 3D
        >>> positions = np.random.rand(1000, 3) * boxDims  # Random particle positions in 3D
        >>> r_s, g_r = calculate_rdf(boxDims, positions)
        >>> print(r_s)  # Radii used in the RDF calculation
        >>> print(g_r)  # Normalized radial distribution function values
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