
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
add_path = os.path.abspath("../..")
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath('..'))





# -- Project information -----------------------------------------------------

project = 'NLSQ'
copyright = '2022-2025, Lucas Hofer'
author = 'Lucas Hofer'

# Get version dynamically
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

try:
    from nlsq import __version__
    release = __version__
    version = '.'.join(__version__.split('.')[:2])  # short version
except ImportError:
    release = 'unknown'
    version = 'unknown'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
]
    
suppress_warnings = [
    'ref.citation',  # Many duplicated citations in numpy/scipy docstrings.
    'ref.footnote',  # Many unreferenced footnotes in numpy/scipy docstrings
]

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autosummary_generate = True

# Napoleon configuration for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Notebooks are not included in the documentation build
# Example notebooks are available in the examples/ directory

# MyST configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Source file types
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_title = f"{project} v{release}"
html_short_title = project

html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_context = {
    "display_github": True,
    "github_user": "Dipolar-Quantum-Gases",
    "github_repo": "nlsq",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ['_static']
html_css_files = []

# Additional HTML options
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_last_updated_fmt = '%b %d, %Y'

# Logo and favicon
html_logo = 'images/NLSQ_logo.png'
html_favicon = None
