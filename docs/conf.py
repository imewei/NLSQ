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
import logging
import os
import sys
import warnings

# Suppress Streamlit logging before any imports
# This prevents "missing ScriptRunContext" warnings during docs build
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)

# Filter out specific warnings that we can't easily suppress
warnings.filterwarnings("ignore", category=UserWarning, module="sphinx")
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

add_path = os.path.abspath("../..")
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath(".."))

# Module aliases for backward compatibility with documentation
# The package was reorganized in v0.4.2 but docs still reference old paths
import importlib

_MODULE_ALIASES = {
    "nlsq.adaptive_hybrid_streaming": "nlsq.streaming.adaptive_hybrid",
    "nlsq.fallback": "nlsq.stability.fallback",
    "nlsq.bound_inference": "nlsq.precision.bound_inference",
    "nlsq.streaming_optimizer": "nlsq.streaming.optimizer",
    "nlsq.least_squares": "nlsq.core.least_squares",
    "nlsq.async_logger": "nlsq.utils.async_logger",
    "nlsq.svd_fallback": "nlsq.stability.svd_fallback",
    "nlsq.loss_functions": "nlsq.core.loss_functions",
    "nlsq.functions": "nlsq.core.functions",
    "nlsq.sparse_jacobian": "nlsq.core.sparse_jacobian",
    "nlsq.memory_manager": "nlsq.caching.memory_manager",
    "nlsq.error_messages": "nlsq.utils.error_messages",
    "nlsq.algorithm_selector": "nlsq.precision.algorithm_selector",
    "nlsq.memory_pool": "nlsq.caching.memory_pool",
    "nlsq.profiling": "nlsq.utils.profiling",
    "nlsq.diagnostics": "nlsq.utils.diagnostics",
    "nlsq.hybrid_streaming_config": "nlsq.streaming.hybrid_config",
    "nlsq.large_dataset": "nlsq.streaming.large_dataset",
    "nlsq.optimizer_base": "nlsq.core.optimizer_base",
    "nlsq.logging": "nlsq.utils.logging",
    "nlsq.streaming_config": "nlsq.streaming.config",
    "nlsq.parameter_estimation": "nlsq.precision.parameter_estimation",
    "nlsq.workflow": "nlsq.core.workflow",
    "nlsq.recovery": "nlsq.stability.recovery",
    "nlsq.unified_cache": "nlsq.caching.unified_cache",
    "nlsq.smart_cache": "nlsq.caching.smart_cache",
    "nlsq.profiler": "nlsq.utils.profiler",
    "nlsq.minpack": "nlsq.core.minpack",
    "nlsq.parameter_normalizer": "nlsq.precision.parameter_normalizer",
    "nlsq.compilation_cache": "nlsq.caching.compilation_cache",
    "nlsq.trf": "nlsq.core.trf",
    "nlsq.robust_decomposition": "nlsq.stability.robust_decomposition",
    "nlsq.mixed_precision": "nlsq.precision.mixed_precision",
    "nlsq.profiler_visualization": "nlsq.utils.profiler_visualization",
    "nlsq.validators": "nlsq.utils.validators",
    "nlsq.stability": "nlsq.stability.guard",
}

for alias, real_module in _MODULE_ALIASES.items():
    try:
        mod = importlib.import_module(real_module)
        sys.modules[alias] = mod
    except ImportError:
        pass  # Module not available, skip alias


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
    "sphinx_design",  # For grid cards and tabs in documentation
]

suppress_warnings = [
    "ref.citation",  # Many duplicated citations in numpy/scipy docstrings.
    "ref.footnote",  # Many unreferenced footnotes in numpy/scipy docstrings
    "ref.python",  # Suppress ambiguous cross-reference warnings for classes in multiple modules
    "toc.excluded",  # Suppress toctree warnings for documents in multiple toctrees
    "toc.not_readable",  # Suppress toctree readability warnings
    "toc.not_included",  # Suppress warnings for autosummary-generated files not in explicit toctree
    "autosummary",  # Suppress autosummary warnings
    "autodoc",  # Suppress autodoc warnings including duplicate object descriptions
    "autodoc.import_object",  # Suppress missing import warnings for experimental features
    "app.add_node",  # Suppress node warnings
    "app.add_directive",  # Suppress directive warnings
    "app.add_role",  # Suppress role warnings
    "duplicate",  # Suppress duplicate object description warnings
    "py:duplicate",  # Suppress Python domain duplicate warnings
    "object",  # Suppress object-related warnings
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

# When nitpicky mode is enabled via -n flag, ignore common type description patterns
# These are informal type descriptions in docstrings, not actual class references
nitpick_ignore = [
    ("py:class", "callable"),
    ("py:class", "optional"),
    ("py:class", "array_like"),
    ("py:class", "arrays"),
    ("py:class", "function"),
    ("py:class", "default True"),
    ("py:class", "default False"),
    ("py:class", "np.ndarray"),
    # Suppress ambiguous cross-reference warnings for classes defined in multiple modules
    ("py:class", "OptimizeResult"),
    ("py:class", "PerformanceProfiler"),
    ("py:class", "StreamingConfig"),
    # JAX array types (not in intersphinx)
    ("py:class", "jnp.ndarray"),
    ("py:class", "jax.Array"),
    # Python stdlib types sometimes not resolved
    ("py:class", "Path"),
    ("py:class", "pathlib.Path"),
    # Internal class references in docstrings
    ("py:class", "auto"),
    ("py:class", "ClusterInfo"),
    ("py:class", "WorkflowConfig"),
    ("py:class", "WorkflowTier"),
    ("py:class", "OptimizationGoal"),
    ("py:class", "DatasetSizeTier"),
    ("py:class", "MemoryTier"),
    ("py:class", "LDMemoryConfig"),
    ("py:class", "MultiGPUConfig"),
    ("py:class", "MixedPrecisionConfig"),
    ("py:class", "UnifiedCache"),
    ("py:class", "CheckpointInfo"),
    ("py:class", "AggregateStats"),
    ("py:class", "CommonError"),
    # Default value patterns in signatures
    ("py:class", "default=128"),
    ("py:class", "default=True"),
    ("py:class", "default=False"),
    ("py:class", "default="),
    # Base class references
    ("py:class", "nlsq.optimizer_base.TrustRegionOptimizerBase"),
    ("py:class", "nlsq.workflow.WorkflowTier"),
    ("py:class", "nlsq.workflow.OptimizationGoal"),
    ("py:class", "nlsq.types.CheckpointInfo"),
    ("py:class", "nlsq.types.AggregateStats"),
    ("py:class", "nlsq.types.CommonError"),
    # Common docstring type patterns
    ("py:class", "ndarray"),
    ("py:class", "n"),
    ("py:class", "shape"),
    ("py:class", "array-like"),
    ("py:class", "csr_matrix"),
    ("py:class", "Figure"),
    ("py:class", "Logger"),
    ("py:class", "jnp.dtype"),
    # Internal config/state classes
    ("py:class", "OptimizationState"),
    ("py:class", "MemoryConfig"),
    ("py:class", "LargeDatasetConfig"),
    ("py:class", "DatasetStats"),
    ("py:class", "nlsq.callbacks.CallbackBase"),
    # More default value patterns
    ("py:class", "default=100"),
    ("py:class", "default=10"),
    ("py:class", "default='auto'"),
    ("py:class", "default=1.0"),
    ("py:class", "default=0.0"),
    ("py:class", "default=None"),
]

# Ignore py:obj references that Sphinx can't resolve
nitpick_ignore_regex = [
    # Function/class references
    (r"py:obj", r".*curve_fit.*"),
    (r"py:obj", r".*Config.*"),
    (r"py:obj", r".*Normalizer.*"),
    (r"py:obj", r".*Selector.*"),
    (r"py:obj", r".*Upgrader.*"),
    (r"py:obj", r".*Monitor.*"),
    (r"py:obj", r".*Fitter.*"),
    (r"py:obj", r".*Tracker.*"),
    (r"py:obj", r".*Orchestrator.*"),
    (r"py:obj", r"nlsq\..*"),  # All nlsq module references
    (r"py:obj", r"estimate_p0.*"),
    (r"py:obj", r"detect_function.*"),
    (r"py:obj", r"fit$"),
    (r"py:obj", r"_save_checkpoint"),
    (r"py:obj", r"_load_checkpoint"),
    (r"py:obj", r"_process_batch.*"),
    # Default value patterns
    (r"py:class", r"default.*"),
    (r"py:class", r"default \d+.*"),
    # String literal types from docstrings
    (r"py:class", r"'.*'"),
    (r"py:class", r"\{.*"),
    # Descriptive type patterns
    (r"py:class", r"ndarray.*"),
    (r"py:class", r"sparse.*"),
    (r"py:class", r"various.*"),
    (r"py:class", r"\d+"),  # Numeric literals
]

# Additional specific ignores for internal classes
nitpick_ignore += [
    # Internal optimizer classes
    ("py:class", "AutoDiffJacobian"),
    ("py:class", "ConvergenceMetrics"),
    ("py:class", "MixedPrecisionManager"),
    ("py:class", "LinearOperator"),
    ("py:class", "LogLevel"),
    # Single-letter type hints from math notation
    ("py:class", "k"),
    ("py:class", "m"),
    ("py:class", "p"),
    ("py:class", "x"),
    # Internal NLSQ classes
    ("py:class", "TrustRegionReflective"),
    ("py:class", "PrecisionUpgrader"),
    ("py:class", "PrecisionState"),
    ("py:class", "NLSQLogger"),
    ("py:class", "LossFunctionsJIT"),
    ("py:class", "JITCompilationCache"),
    ("py:class", "CurveFitResult"),
    ("py:class", "BestParameterTracker"),
    ("py:class", "nlsq.result.CurveFitResult"),
    ("py:class", "nlsq.mixed_precision.OptimizationState"),
    ("py:class", "nlsq.mixed_precision.ConvergenceMonitor"),
    # Descriptive types
    ("py:class", "file-like object"),
    ("py:class", "2-tuple"),
    # py:obj references
    ("py:obj", "AdaptiveHybridStreamingOptimizer"),
    ("py:obj", "StreamingOptimizer"),
    ("py:obj", "PrecisionState"),
    ("py:obj", "OptimizationState"),
    ("py:obj", "ConvergenceMetrics"),
    ("py:obj", "WORKFLOW_PRESETS"),
    ("py:obj", "format_error_message"),
    ("py:obj", "estimate_initial_parameters"),
    ("py:obj", "device_put"),
    ("py:obj", "auto_select_workflow"),
    ("py:obj", "apply_automatic_fixes"),
    ("py:obj", "analyze_failure"),
    # py:mod references
    ("py:mod", "notebook_utils"),
    ("py:mod", "nlsq.profiling"),
]

# Regex patterns for closing braces in dict types
nitpick_ignore_regex += [
    (r"py:class", r".*\}$"),  # Catch dict closing braces like 'halton'}
    (r"py:class", r"False\}"),
    (r"py:class", r".*with shape.*"),  # "int with shape", "ndarray with shape"
    (r"py:class", r".*object$"),  # "config object", "file-like object"
]

# Final specific ignores
nitpick_ignore += [
    # Module references
    ("py:mod", "nlsq.diagnostics"),
    # More internal classes
    ("py:class", "result"),
    ("py:class", "OptimizationError"),
    ("py:class", "nlsq.trf.TrustRegionJITFunctions"),
    ("py:class", "nlsq.ParameterNormalizer"),
    ("py:class", "nlsq.optimizer_base.OptimizerBase"),
    ("py:class", "nlsq.large_dataset.LDMemoryConfig"),
    ("py:class", "nlsq.LargeDatasetHandler"),
    ("py:class", "nlsq.large_dataset.DataChunker"),
    ("py:class", "matplotlib.axes.Axes"),
    ("py:class", "generator"),
    ("py:class", "CallbackBase"),
    # Plotly and GUI classes
    ("py:class", "go.Figure"),
    ("py:class", "FitConfig"),
    ("py:class", "plotly.graph_objects.Figure"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "ProgressCallback"),
    ("py:class", "ValidationResult"),
    ("py:class", "DataLoader"),
    ("py:class", "ModelRegistry"),
    ("py:class", "ResultExporter"),
    ("py:class", "ModelFunction"),
    # Bound inference types (variable names parsed as types)
    ("py:class", "x_min"),
    ("py:class", "y_min"),
    ("py:class", "x_range"),
]

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
napoleon_use_ivar = True  # Use :ivar: for attributes to avoid duplicate with autodoc
napoleon_attr_annotations = True  # Parse type annotations for attributes

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
    "jax": ("https://docs.jax.dev/en/latest/", None),  # Updated from jax.readthedocs.io
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
    "navigation_depth": 3,  # Reduced from 4 for shallower sidebar navigation
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

# -- Linkcheck configuration -------------------------------------------------

# Ignore known broken or problematic links
linkcheck_ignore = [
    # Files that may not exist in remote repo yet
    r"https://github\.com/imewei/nlsq/blob/main/CODE_OF_CONDUCT\.md",
    r"https://github\.com/imewei/NLSQ/blob/main/.*",  # Case-sensitive URLs
    # Rate-limited or auth-required URLs
    r"https://github\.com/orgs/community/.*",
]

# Handle known redirects gracefully
linkcheck_allowed_redirects = {
    r"https://doi\.org/.*": r"https://arxiv\.org/.*",
    r"https://jax\.readthedocs\.io/.*": r"https://docs\.jax\.dev/.*",
    r"https://nlsq\.readthedocs\.io/?$": r"https://nlsq\.readthedocs\.io/en/latest/",
    r"https://numpydoc\.readthedocs\.io/?$": r"https://numpydoc\.readthedocs\.io/en/latest/",
    r"https://www\.sphinx-doc\.org/?$": r"https://www\.sphinx-doc\.org/en/master/",
    r"https://codeql\.github\.com/.*": r"https://docs\.github\.com/.*",
    r"https://support\.github\.com/?$": r"https://support\.github\.com/request/landing",
}

# Linkcheck settings
linkcheck_timeout = 30
linkcheck_retries = 2
linkcheck_workers = 5
