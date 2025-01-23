"""
This module enables reading and writing of data created by various software packages
"""

__version__ = "0.1.0"  # Define the package version here.
__author__ = "Nikhil Karthikeyan"
__license__ = "MIT"  # Replace with your license type if needed

# Import core functionality from your package
from .lb3d import read_hdf5, read_asc, rewrite_asc_file
from .visualization import write_vti, animate_colormap, animate_plot

__all__ = [read_hdf5, read_asc, rewrite_asc_file,
           write_vti, animate_colormap, animate_plot]