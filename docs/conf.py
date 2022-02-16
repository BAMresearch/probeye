# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute.

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../probeye"))
sys.path.insert(0, os.path.abspath("../"))
from probeye import __version__

now = datetime.datetime.now()


# -- Project information -----------------------------------------------------

project = "probeye"
copyright = f"{now.year}, BAM"
author = "Alexander Klawonn"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinxcontrib.bibtex",
    "myst_parser",
    "sphinx.ext.imgmath",
]

autodoc_typehints = "description"

# Add external package dependencies to be mocked (this way they do not have to be
# installed for the doc build) TODO: this list could be built automatically
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "matplotlib",
    "emcee",
    "tabulate",
    "torch",
    "pyro",
    "arviz",
    "loguru",
    "dynesty",
    "tri-py",
    "numba",
]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# bibliography settings
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"


# Sphinx gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "examples",  # path to examples for the gallery
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "download_all_examples": False,
    "show_signature": False,
    "remove_config_comments": True,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_title = "probeye"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# this adds a custom javascript file that
# * opens all external link in a new tab
html_js_files = ["js/custom.js"]
