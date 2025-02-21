# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../metocean_ml'))
sys.path.insert(0, os.path.abspath('../..'))
#sys.path.append(os.path.abspath('../..'))

project = 'metocean-ml'
copyright = '2025, MET Norway'
author = 'MET Norway'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon', # For numpy and google style docstrings
    'sphinx.ext.autodoc',  # For automatic documentation generation
#    'sphinx.ext.autosummary',  # Create summary tables
]
#autosummary_generate = True  # Turn on sphinx.ext.autosummary
napoleon_numpy_docstring = True
napoleon_google_docstring = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
