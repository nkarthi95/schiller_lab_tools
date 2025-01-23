import os
import sys
import schiller_lab_tools
sys.path.insert(0, os.path.abspath('../../schiller_lab_tools'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'schiller_lab_tools'
copyright = '2025, Nikhil Karthikeyan'
author = 'Nikhil Karthikeyan'
release = schiller_lab_tools.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'numpydoc',
]

templates_path = ['_templates']
master_doc = 'index'
language = "en"
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
