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
import warnings

# Filter out specific warnings that we can't easily suppress
warnings.filterwarnings("ignore", category=UserWarning, module="sphinx")

add_path = os.path.abspath("../..")
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "NLSQ"
copyright = (
    "2024-2025, Wei Chen (Argonne National Laboratory) | 2022, Original JAXFit Authors"
)
author = "Wei Chen"

# Get version dynamically
# (imports already done above)

try:
    from nlsq import __version__

    release = __version__
    version = ".".join(__version__.split(".")[:2])  # short version
except ImportError:
    release = "unknown"
    version = "unknown"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "myst_parser",  # Enabled for developer documentation in Markdown
]

suppress_warnings = [
    "ref.citation",  # Many duplicated citations in numpy/scipy docstrings.
    "ref.footnote",  # Many unreferenced footnotes in numpy/scipy docstrings
    "autosummary",  # Suppress autosummary warnings
    "autodoc.import_object",  # Suppress missing import warnings for experimental features
    "app.add_node",  # Suppress node warnings
    "app.add_directive",  # Suppress directive warnings
    "app.add_role",  # Suppress role warnings
]

# Additional Sphinx configuration to handle duplicate warnings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Configure autodoc to not warn about duplicate descriptions
autodoc_warningiserror = False

# Handle duplicate object warnings by making Sphinx less strict
keep_warnings = False

# Nitpick configuration to ignore specific warnings
nitpicky = False

# Custom event handler removed - caused TypeError

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autosummary_generate = True
autodoc_mock_imports = []
autodoc_typehints_format = "short"

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
    # "linkify",  # Disabled due to missing dependency
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Source file types - RST for main docs, Markdown for developer docs
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "_reorganization",
    "Thumbs.db",
    ".DS_Store",
    "user_guides",  # Old directory, content moved to guides/
    "tutorials",  # Old directory, content moved to getting_started/ and guides/
    "autodoc",  # Old directory, renamed to api/
    "development",  # Old directory, consolidated to history/
    "archive",  # Old directory, moved to history/archived_reports/
    "FINAL_DOCUMENTATION_REPORT.md",  # Standalone report file
    "history/archived_reports/TEST_GENERATION_PHASE2_REPORT.md",  # Archived report
    "history/archived_reports/sprint_1_2_completion_report.md",  # Archived report
    "history/archived_reports/sprint_3_completion_report.md",  # Archived report
]
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"{project} v{release}"
html_short_title = project

html_theme_options = {
    "analytics_id": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_context = {
    "display_github": True,
    "github_user": "imewei",
    "github_repo": "nlsq",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]
html_css_files = []

# Additional HTML options
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_last_updated_fmt = "%b %d, %Y"

# Logo and favicon
html_logo = "images/NLSQ_logo.png"
html_favicon = None
