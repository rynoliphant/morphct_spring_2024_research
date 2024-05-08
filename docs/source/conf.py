import pathlib
import sys
import os
sys.path.append('/../..')
sys.path.insert(0, os.path.abspath("../.."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MorphCT'
copyright = '2024, Ryn Oliphant'
author = 'Ryn Oliphant'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ["sphinx.ext.autodoc","sphinx.ext.autosummary", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
autodoc_mock_imports = [
    "blas"
    "ele",
    "freud",
    "grits",
    "gsd",
    "h5py"
    "mbuild",
    "matplotlib"
    "numpy",
    "openbabel",
    "py3Dmol",
    "scipy",
    "signac",
    "jupyterlab",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
source_suffix = [".rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
