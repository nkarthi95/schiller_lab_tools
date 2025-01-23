"""
This package details and defines the functions used for research activities 
performed by the Schiller research group
"""

__version__ = "0.1.0"  # Define the package version here.
__author__ = "Nikhil Karthikeyan"
__license__ = "MIT"  # Replace with your license type if needed

# Import core functionality from your package
from .lb3d import read_hdf5, read_asc, rewrite_asc_file
from .visualization import write_vti, animate_colormap, animate_plot
from .droplet_analysis import droplet_radius, pressure_jump, inertia_tensor, gyration_tensor, deformation1, inclination_angle
from .microstructure_analysis import structure_factor, spherically_averaged_structure_factor, spherical_first_moment, second_moment, interface_order, curvature, fill, label_regions_hk, get_pn, taufactor_tortuosity
from .particle_analysis import calculate_average_cos_interface_normal, calculate_rdf, calculate_nematic_order, calculate_ql, calculate_wl

# from .input_output import lb3d, visualization
# from .microstructure import droplet_analysis, microstructure_analysis
# from .particle import particle_analysis