"""
This module enables particle property analysis for structures. Instructions are not provided. Use as relevant.
"""

__version__ = "0.1.0"  # Define the package version here.
__author__ = "Nikhil Karthikeyan"
__license__ = "MIT"  # Replace with your license type if needed

# Import core functionality from your package
from .particle_analysis import calculate_average_cos_interface_normal, calculate_rdf, calculate_nematic_order, calculate_minkowski_q

__all__ = [calculate_average_cos_interface_normal, 
           calculate_rdf, calculate_nematic_order, 
           calculate_minkowski_q]