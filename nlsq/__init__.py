"""
NLSQ: JAX-accelerated nonlinear least squares curve fitting.

GPU/TPU-accelerated curve fitting with automatic differentiation.
Provides drop-in SciPy compatibility with curve_fit function.
Supports large datasets through automatic chunking and streaming optimization.

Key Features
------------
- Drop-in replacement for scipy.optimize.curve_fit
- GPU/TPU acceleration via JAX
- Automatic memory management for datasets up to 100M+ points
- Streaming optimization for unlimited data
- Smart algorithm selection and numerical stability
- Unified fit() entry point with automatic workflow selection

Examples
--------
>>> import jax.numpy as jnp
>>> from nlsq import curve_fit, fit
>>> def model(x, a, b): return a * jnp.exp(-b * x)
>>> popt, pcov = curve_fit(model, xdata, ydata)
>>> # Or use unified fit() with automatic workflow selection:
>>> popt, pcov = fit(model, xdata, ydata, workflow="auto", goal="quality")

"""

import faulthandler
import os
import sys
import threading

# Enable faulthandler FIRST so any SIGBUS/SIGSEGV during import prints a
# traceback to stderr instead of dying silently.  This is deliberately before
# the environment lock and all library imports below.
faulthandler.enable()

_NLSQ_DEBUG = bool(os.environ.get("NLSQ_DEBUG"))


def _dbg(msg: str) -> None:
    if _NLSQ_DEBUG:
        sys.stderr.write(f"[nlsq] {msg}\n")
        sys.stderr.flush()


_dbg("faulthandler enabled")

# Suppress noisy XLA/PJRT C++ warnings (e.g. "Assume version compatibility.
# PjRt-IFRT does not track XLA executable versions.") that are harmless on
# JAX >=0.8 and clutter stderr.  Level 2 = ERROR-only (suppress INFO+WARNING).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =============================================================================
# PLATFORM-SPECIFIC ENVIRONMENT LOCK
# GPU acceleration (CUDA) is supported ONLY on Linux.
# macOS and Windows enforce CPU-only JAX backend.
# These MUST be set before any JAX, Matplotlib, or Qt modules are initialized.
# =============================================================================
if sys.platform != "linux":
    # Non-Linux CPU enforcement (macOS + Windows)
    os.environ["NLSQ_FORCE_CPU"] = "1"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"

if sys.platform == "darwin":
    # macOS-specific: Prevent SIGBUS from Metal/OpenGL/XLA conflicts
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    if "QT_MAC_WANTS_LAYER" not in os.environ:
        os.environ["QT_MAC_WANTS_LAYER"] = "1"
    os.environ["QT_API"] = "pyside6"
    os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
    # Force software OpenGL — prevents Metal/OpenGL context conflicts that
    # cause SIGBUS on macOS 26+ Apple Silicon with PySide6 / Qt 6.10.
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

if sys.platform == "linux" and not os.environ.get("NLSQ_FORCE_CPU"):
    # Configure XLA GPU memory behaviour BEFORE JAX initializes its backend.
    # The PJRT GPU client reads these at backend-init (first device use), so
    # they only take effect if set before any `import jax` triggers init — which
    # is why they live here rather than in a helper called after JAX is already
    # imported. Disabling preallocation avoids cuSolver fragmentation/OOM during
    # the SVD fallback path; the mem fraction caps how much device memory JAX
    # may grab. NOTE: the *correct* XLA variable names are
    # XLA_PYTHON_CLIENT_PREALLOCATE / XLA_PYTHON_CLIENT_MEM_FRACTION — earlier
    # code wrote non-existent JAX_PREALLOCATE_GPU_MEMORY / JAX_GPU_MEMORY_FRACTION
    # names (and did so after JAX import), so the settings were silently ignored.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

_dbg("env lock applied")

import numpy as np

_dbg("numpy imported")

# =============================================================================
# CORE IMPORTS (Always Eager - Required for basic curve_fit usage)
# =============================================================================

# Version information
try:
    from nlsq._version import __version__  # type: ignore[import-not-found]
except ImportError:
    __version__ = "0.0.0+unknown"

# Standard library imports needed at module level
from typing import TYPE_CHECKING, Any, Literal

from nlsq.callbacks import (
    CallbackBase,
    CallbackChain,
    EarlyStopping,
    IterationLogger,
    ProgressBar,
    StopOptimization,
)

_dbg("callbacks imported")

from nlsq.core import functions
from nlsq.core.least_squares import LeastSquares
from nlsq.core.minpack import CurveFit, curve_fit

_dbg("core imported")

from nlsq.result import OptimizeResult, OptimizeWarning
from nlsq.types import ArrayLike, BoundsTuple, MethodLiteral, ModelFunction

_dbg("types imported")

# =============================================================================
# LAZY IMPORT SYSTEM
# Specialty modules are loaded on-demand to reduce import time.
# This achieves 50%+ reduction in cold import time (SC-001).
# =============================================================================

# Mapping of lazy export names to their source modules
_LAZY_MODULES: dict[str, str] = {
    # Streaming & Large Dataset (h5py dependency, specialty use case)
    "AdaptiveHybridStreamingOptimizer": "nlsq.streaming.adaptive_hybrid",
    "DefenseLayerTelemetry": "nlsq.streaming.adaptive_hybrid",
    "get_defense_telemetry": "nlsq.streaming.adaptive_hybrid",
    "reset_defense_telemetry": "nlsq.streaming.adaptive_hybrid",
    "HybridStreamingConfig": "nlsq.streaming.hybrid_config",
    "LargeDatasetFitter": "nlsq.streaming.large_dataset",
    "LDMemoryConfig": "nlsq.streaming.large_dataset",
    "estimate_memory_requirements": "nlsq.streaming.large_dataset",
    "fit_large_dataset": "nlsq.streaming.large_dataset",
    # Global Optimization
    "GlobalOptimizationConfig": "nlsq.global_optimization",
    "MultiStartOrchestrator": "nlsq.global_optimization",
    "TournamentSelector": "nlsq.global_optimization",
    # Profiling & Visualization
    "PerformanceProfiler": "nlsq.utils.profiler",
    "ProfileMetrics": "nlsq.utils.profiler",
    "clear_profiling_data": "nlsq.utils.profiler",
    "get_global_profiler": "nlsq.utils.profiler",
    "ProfilerVisualization": "nlsq.utils.profiler_visualization",
    "ProfilingDashboard": "nlsq.utils.profiler_visualization",
    # Diagnostics
    "ConvergenceMonitor": "nlsq.utils.diagnostics",
    "OptimizationDiagnostics": "nlsq.utils.diagnostics",
    # Memory Management
    "MemoryManager": "nlsq.caching.memory_manager",
    "clear_memory_pool": "nlsq.caching.memory_manager",
    "get_memory_manager": "nlsq.caching.memory_manager",
    "get_memory_stats": "nlsq.caching.memory_manager",
    # Fallback & Recovery
    "FallbackOrchestrator": "nlsq.stability.fallback",
    "FallbackResult": "nlsq.stability.fallback",
    "FallbackStrategy": "nlsq.stability.fallback",
    "OptimizationRecovery": "nlsq.stability.recovery",
    # Sparse Jacobian
    "SparseJacobianComputer": "nlsq.core.sparse_jacobian",
    "SparseOptimizer": "nlsq.core.sparse_jacobian",
    "detect_jacobian_sparsity": "nlsq.core.sparse_jacobian",
    # Workflow System (014-unified-memory-strategy)
    "MemoryBudget": "nlsq.core.workflow",
    "MemoryBudgetSelector": "nlsq.core.workflow",
    "OptimizationGoal": "nlsq.core.workflow",
    # Algorithm Selection
    "AlgorithmSelector": "nlsq.precision.algorithm_selector",
    "auto_select_algorithm": "nlsq.precision.algorithm_selector",
    # Bounds Inference
    "BoundsInference": "nlsq.precision.bound_inference",
    "infer_bounds": "nlsq.precision.bound_inference",
    "merge_bounds": "nlsq.precision.bound_inference",
    # Parameter Normalizer
    "ParameterNormalizer": "nlsq.precision.parameter_normalizer",
    # Stability
    "NumericalStabilityGuard": "nlsq.stability.guard",
    "apply_automatic_fixes": "nlsq.stability.guard",
    "check_problem_stability": "nlsq.stability.guard",
    "detect_collinearity": "nlsq.stability.guard",
    "detect_parameter_scale_mismatch": "nlsq.stability.guard",
    "estimate_condition_number": "nlsq.stability.guard",
    # Configuration
    "LargeDatasetConfig": "nlsq.config",
    "MemoryConfig": "nlsq.config",
    "configure_for_large_datasets": "nlsq.config",
    "get_large_dataset_config": "nlsq.config",
    "get_memory_config": "nlsq.config",
    "large_dataset_context": "nlsq.config",
    "memory_context": "nlsq.config",
    "set_memory_limits": "nlsq.config",
    # Caching
    "cached_jit": "nlsq.caching.unified_cache",
    "clear_cache": "nlsq.caching.unified_cache",
    "get_cache_stats": "nlsq.caching.unified_cache",
    "get_global_cache": "nlsq.caching.unified_cache",
    "UnifiedCache": "nlsq.caching.unified_cache",
    # Validators
    "InputValidator": "nlsq.utils.validators",
    # Model Health Diagnostics
    "HealthStatus": "nlsq.diagnostics.types",
    "IssueSeverity": "nlsq.diagnostics.types",
    "IssueCategory": "nlsq.diagnostics.types",
    "DiagnosticLevel": "nlsq.diagnostics.types",
    "ModelHealthIssue": "nlsq.diagnostics.types",
    "IdentifiabilityReport": "nlsq.diagnostics.types",
    "GradientHealthReport": "nlsq.diagnostics.types",
    "ParameterSensitivityReport": "nlsq.diagnostics.types",
    "ModelHealthReport": "nlsq.diagnostics.types",
    "DiagnosticsConfig": "nlsq.diagnostics.types",
    "IdentifiabilityAnalyzer": "nlsq.diagnostics.identifiability",
    "GradientMonitor": "nlsq.diagnostics.gradient_health",
    "ParameterSensitivityAnalyzer": "nlsq.diagnostics.parameter_sensitivity",
    "DiagnosticPlugin": "nlsq.diagnostics.plugin",
    "PluginRegistry": "nlsq.diagnostics.plugin",
    "run_plugins": "nlsq.diagnostics.plugin",
    "create_health_report": "nlsq.diagnostics.health_report",
}

# Cache for lazily-loaded attributes
_LAZY_CACHE: dict[str, Any] = {}

# Top-level exports that still resolve but warn once per name. Import them from
# their defining modules (nlsq.utils.profiler / nlsq.utils.profiler_visualization).
_DEPRECATED_EXPORTS: dict[str, str] = {
    name: f"nlsq.{name} is deprecated and will be removed in the next minor release"
    for name in (
        "PerformanceProfiler",
        "ProfileMetrics",
        "clear_profiling_data",
        "get_global_profiler",
        "ProfilerVisualization",
        "ProfilingDashboard",
    )
}
_deprecation_seen: set[str] = set()
_deprecation_lock = threading.Lock()


def __getattr__(name: str) -> Any:
    """Lazily load specialty modules on first access.

    This enables faster import time by deferring loading of specialty
    modules until they are actually needed.
    """
    if name in _DEPRECATED_EXPORTS and name not in _deprecation_seen:
        import warnings

        # The lock makes check-warn-add atomic so concurrent first accesses warn
        # once. Mark seen only after warn: under -W error the raise must repeat.
        with _deprecation_lock:
            if name not in _deprecation_seen:
                warnings.warn(
                    _DEPRECATED_EXPORTS[name], DeprecationWarning, stacklevel=2
                )
                _deprecation_seen.add(name)

    # Check if it's a lazy module
    if name in _LAZY_MODULES:
        # Return cached value if already loaded
        if name in _LAZY_CACHE:
            return _LAZY_CACHE[name]

        # Import the module and get the attribute
        module_path = _LAZY_MODULES[name]
        try:
            import importlib

            module = importlib.import_module(module_path)
            attr = getattr(module, name)
            # Cache for future access (setdefault is atomic under GIL)
            return _LAZY_CACHE.setdefault(name, attr)
        except ImportError as e:
            # Handle missing optional dependencies gracefully
            raise ImportError(
                f"Cannot import '{name}' from '{module_path}'. "
                f"This may require an optional dependency. Error: {e}",
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Module '{module_path}' does not have attribute '{name}'",
            ) from e

    # Not found - raise standard AttributeError
    raise AttributeError(f"module 'nlsq' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Return list of module attributes including lazy exports.

    This allows tools like IPython and IDEs to see all available exports
    even before they are loaded.
    """
    # Get standard module attributes
    module_attrs = list(globals().keys())
    # Add all lazy module exports
    lazy_attrs = list(_LAZY_MODULES.keys())
    # Combine and deduplicate
    return sorted(set(module_attrs + lazy_attrs))


# Public API - only expose main user-facing functions
__all__ = [
    "AdaptiveHybridStreamingOptimizer",
    "AlgorithmSelector",
    "BoundsInference",
    "CallbackBase",
    "CallbackChain",
    "ConvergenceMonitor",
    "CurveFit",
    "DefenseLayerTelemetry",
    "EarlyStopping",
    "FallbackOrchestrator",
    "FallbackResult",
    "FallbackStrategy",
    "GlobalOptimizationConfig",
    "HybridStreamingConfig",
    "InputValidator",
    "IterationLogger",
    "LDMemoryConfig",
    "LargeDatasetConfig",
    "LargeDatasetFitter",
    "LeastSquares",
    "MemoryBudget",
    "MemoryBudgetSelector",
    "MemoryConfig",
    "MemoryManager",
    "MultiStartOrchestrator",
    "NumericalStabilityGuard",
    "OptimizationDiagnostics",
    "OptimizationGoal",
    "OptimizationRecovery",
    "OptimizeResult",
    "OptimizeWarning",
    "ParameterNormalizer",
    "ProgressBar",
    "SparseJacobianComputer",
    "SparseOptimizer",
    "StopOptimization",
    "TournamentSelector",
    "UnifiedCache",
    "__version__",
    "apply_automatic_fixes",
    "auto_select_algorithm",
    "cached_jit",
    "check_problem_stability",
    "clear_cache",
    "clear_memory_pool",
    "configure_for_large_datasets",
    "curve_fit",
    "curve_fit_large",
    "detect_collinearity",
    "detect_jacobian_sparsity",
    "detect_parameter_scale_mismatch",
    "estimate_condition_number",
    "estimate_memory_requirements",
    "fit",
    "fit_large_dataset",
    "functions",
    "get_cache_stats",
    "get_defense_telemetry",
    "get_global_cache",
    "get_large_dataset_config",
    "get_memory_config",
    "get_memory_manager",
    "get_memory_stats",
    "infer_bounds",
    "large_dataset_context",
    "memory_context",
    "merge_bounds",
    "reset_defense_telemetry",
    "set_memory_limits",
]

# Preset configurations for the fit() function
_FIT_PRESETS = {
    "fast": {
        "n_starts": 0,
        "multistart": False,
        "description": "Single-start optimization for maximum speed",
    },
    "robust": {
        "n_starts": 5,
        "multistart": True,
        "description": "Multi-start with 5 starts for robustness",
    },
    "global": {
        "n_starts": 20,
        "multistart": True,
        "description": "Thorough global search with 20 starts",
    },
    "streaming": {
        "n_starts": 10,
        "multistart": True,
        "use_streaming": True,
        "description": "Streaming optimization for large datasets with multi-start",
    },
    "large": {
        "n_starts": 5,
        "multistart": True,
        "use_large_dataset": True,
        "description": "Auto-detect dataset size and use appropriate strategy",
    },
}


def fit(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    check_finite: bool = True,
    bounds: BoundsTuple | tuple[float, float] = (-float("inf"), float("inf")),
    method: MethodLiteral | None = None,
    # NEW: Workflow parameter (v0.6.3)
    workflow: str | None = None,
    # Legacy: preset parameter (deprecated)
    preset: Literal["fast", "robust", "global", "streaming", "large"] | None = None,
    # Multi-start parameters (can override preset)
    multistart: bool | None = None,
    n_starts: int | None = None,
    sampler: Literal["lhs", "sobol", "halton"] = "lhs",
    center_on_p0: bool = True,
    scale_factor: float = 1.0,
    # Large dataset parameters
    memory_limit_gb: float | None = None,
    size_threshold: int = 1_000_000,
    show_progress: bool = False,
    chunk_size: int | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray] | OptimizeResult:
    """Unified curve fitting function with preset-based configuration.

    This function provides a simplified API for curve fitting with sensible
    defaults based on preset configurations. It automatically selects the
    appropriate backend (curve_fit, curve_fit_large, or streaming) based on
    the preset and dataset characteristics.

    Parameters
    ----------
    f : callable
        Model function f(x, \\*params) -> y. Must use jax.numpy operations.
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    p0 : array_like, optional
        Initial parameter guess.
    sigma : array_like, optional
        Uncertainties in ydata for weighted fitting.
    absolute_sigma : bool, optional
        Whether sigma represents absolute uncertainties.
    check_finite : bool, optional
        Check for finite input values.
    bounds : tuple, optional
        Parameter bounds as (lower, upper).
    method : str, optional
        Optimization algorithm ('trf', 'lm', or None for auto).
    preset : {'fast', 'robust', 'global', 'streaming', 'large'}, optional
        Preset configuration to use:

        - 'fast': Single-start optimization for maximum speed (n_starts=0)
        - 'robust': Multi-start with 5 starts for robustness
        - 'global': Thorough global search with 20 starts
        - 'streaming': Streaming optimization for large datasets with multi-start
        - 'large': Auto-detect dataset size and use appropriate strategy

        If None, defaults to 'fast' for small datasets or 'large' for datasets
        exceeding size_threshold.
    multistart : bool, optional
        Override preset's multi-start setting.
    n_starts : int, optional
        Override preset's n_starts setting.
    sampler : {'lhs', 'sobol', 'halton'}, optional
        Sampling strategy for multi-start. Default: 'lhs'.
    center_on_p0 : bool, optional
        Center multi-start samples around p0. Default: True.
    scale_factor : float, optional
        Scale factor for exploration region. Default: 1.0.
    memory_limit_gb : float, optional
        Maximum memory usage in GB for large datasets.
    size_threshold : int, optional
        Point threshold for large dataset processing (default: 1M).
    show_progress : bool, optional
        Display progress bar for long operations.
    chunk_size : int, optional
        Override automatic chunk size calculation.
    **kwargs
        Additional optimization parameters (ftol, xtol, gtol, max_nfev, loss).

    Returns
    -------
    result : CurveFitResult or tuple
        Optimization result. Contains popt, pcov, and multistart_diagnostics.
        Supports tuple unpacking: popt, pcov = fit(...)

    Examples
    --------
    Basic usage with default preset:

    >>> popt, pcov = fit(model_func, xdata, ydata, p0=[1, 2, 3])

    Using 'robust' preset for multi-start:

    >>> result = fit(model_func, xdata, ydata, p0=[1, 2, 3],
    ...              bounds=([0, 0, 0], [10, 10, 10]), preset='robust')

    Using 'global' preset for thorough search:

    >>> result = fit(model_func, xdata, ydata, p0=[1, 2, 3],
    ...              bounds=([0, 0, 0], [10, 10, 10]), preset='global')

    Large dataset with auto-detection:

    >>> result = fit(model_func, big_xdata, big_ydata,
    ...              preset='large', show_progress=True)

    See Also
    --------
    curve_fit : Lower-level API with full control
    curve_fit_large : Specialized API for large datasets
    MemoryBudgetSelector : Memory-based optimizer selection
    """
    if method == "exact_streaming":  # not in MethodLiteral, but reachable untyped
        raise ValueError(
            "method='exact_streaming' is available only through curve_fit_large(); "
            "fit() does not dispatch to it at any dataset size"
        )

    # NEW WORKFLOW SYSTEM (v0.6.3)
    # If workflow parameter is specified, delegate to minpack.fit()
    if workflow is not None:
        from nlsq.core.minpack import fit as minpack_fit

        # Forward to the new workflow-based fit
        return minpack_fit(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            method=method,
            workflow=workflow,
            n_starts=n_starts,
            memory_limit_gb=memory_limit_gb,
            **kwargs,
        )

    # LEGACY PATH: preset-based configuration (for backwards compatibility)
    # Emit deprecation warning if preset is explicitly provided
    if preset is not None:
        import warnings

        warnings.warn(
            "The 'preset' parameter is deprecated and will be removed in v0.8.0. "
            "Use the 'workflow' parameter instead: "
            "workflow='auto' (default), 'auto_global' (robust), or 'hpc' (streaming). "
            "See migration guide: https://nlsq.readthedocs.io/en/latest/migration.html",
            FutureWarning,
            stacklevel=2,
        )

    # Input validation
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    # scipy convention: 2-D xdata is (k, M), point count is the last axis
    n_points = xdata.shape[-1] if xdata.ndim > 1 else len(xdata)

    # Auto-select preset if not provided
    if preset is None:
        if n_points >= size_threshold:
            preset = "large"
        else:
            preset = "fast"

    # Get preset configuration
    preset_config = _FIT_PRESETS.get(preset, _FIT_PRESETS["fast"])

    # Apply preset defaults, allowing overrides
    effective_multistart: bool = (
        multistart
        if multistart is not None
        else bool(preset_config.get("multistart", False))
    )
    _n_starts_default: int = preset_config.get("n_starts", 0)  # type: ignore[assignment]
    effective_n_starts: int = n_starts if n_starts is not None else _n_starts_default
    use_streaming = preset_config.get("use_streaming", False)
    use_large_dataset = preset_config.get("use_large_dataset", False)

    # Determine which backend to use
    if use_streaming or preset == "streaming":
        # Use AdaptiveHybridStreaming for streaming preset
        from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        # Prepare p0
        if p0 is None:
            from inspect import signature

            sig = signature(f)
            args = sig.parameters
            if len(args) < 2:
                raise ValueError("Unable to determine number of fit parameters.")
            n_params = len(args) - 1
            p0 = np.ones(n_params)
        p0 = np.atleast_1d(p0)

        # Prepare bounds
        from nlsq.core.least_squares import prepare_bounds

        lb, ub = prepare_bounds(bounds, len(p0))
        bounds_tuple = (
            (lb, ub)
            if not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))
            else None
        )

        # Create config with multi-start settings
        # Use the correct parameter names from HybridStreamingConfig
        config = HybridStreamingConfig(
            enable_multistart=effective_multistart and effective_n_starts > 0,
            n_starts=effective_n_starts if effective_multistart else 10,
            multistart_sampler=sampler,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config=config)

        result_dict = optimizer.fit(
            data_source=(xdata, ydata),
            func=f,
            p0=p0,  # type: ignore[arg-type]
            bounds=bounds_tuple,  # type: ignore[arg-type]
            sigma=sigma,  # type: ignore[arg-type]
            absolute_sigma=absolute_sigma,
            callback=kwargs.get("callback"),
            verbose=kwargs.get("verbose", 1),
        )

        # Convert to standard result format
        from nlsq.result import CurveFitResult

        result = CurveFitResult(result_dict)
        result["pcov"] = result_dict.get("pcov", np.full((len(p0), len(p0)), np.inf))
        result["multistart_diagnostics"] = {
            "n_starts_configured": effective_n_starts,
            "bypassed": not effective_multistart or effective_n_starts == 0,
            "preset": preset,
        }
        return result

    if use_large_dataset or n_points >= size_threshold:
        # Use curve_fit_large for large datasets
        return curve_fit_large(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            method=method,
            memory_limit_gb=memory_limit_gb,
            size_threshold=size_threshold,
            show_progress=show_progress,
            chunk_size=chunk_size,
            multistart=effective_multistart,
            n_starts=effective_n_starts,
            sampler=sampler,
            center_on_p0=center_on_p0,
            scale_factor=scale_factor,
            **kwargs,
        )

    # Use standard curve_fit
    return curve_fit(
        f,
        xdata,
        ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        check_finite=check_finite,
        bounds=bounds,
        method=method,
        multistart=effective_multistart,
        n_starts=effective_n_starts,
        sampler=sampler,
        center_on_p0=center_on_p0,
        scale_factor=scale_factor,
        **kwargs,
    )


def _default_p0(f: ModelFunction) -> np.ndarray:
    """Ones for every parameter after ``x`` in ``f``'s signature (curve_fit's p0=None)."""
    from inspect import signature

    args = signature(f).parameters
    if len(args) < 2:
        raise ValueError("Unable to determine number of fit parameters.")
    return np.ones(len(args) - 1)


_EXACT_DEFAULTS: dict[str, Any] = {
    "stability": False,
    "rescale_data": True,
    "max_jacobian_elements_for_svd": 10_000_000,
    "auto_size_detection": True,
    "size_threshold": 1_000_000,
}
_EXACT_SOLVER_KWARGS = frozenset({"ftol", "xtol", "gtol", "max_nfev", "n_points"})
_NOT_EXACT = "is not supported by method='exact_streaming'"


def _curve_fit_exact_streaming(
    f: ModelFunction,
    xdata: Any,
    ydata: Any,
    p0: ArrayLike | None,
    *,
    sigma: Any,
    absolute_sigma: bool,
    check_finite: bool,
    bounds: Any,
    memory_limit_gb: float | None,
    show_progress: bool,
    chunk_size: int | None,
    multistart: bool,
    unsupported: dict[str, Any],
    solver_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """``curve_fit_large(method="exact_streaming")``: argument policy, then exact_lm_fit.

    Validates from ``shape``/``len`` only, so array-like sources (``np.memmap``,
    ``h5py.Dataset``, ``zarr.Array``, any object with ``shape``, ``len`` and slicing) are
    never read whole here; objects without ``shape`` (lists, tuples) are converted.
    A callable ``xdata`` is a chunk factory: ``ydata`` and ``sigma`` must be None and
    ``n_points`` (accepted only then) is optional; blocks are validated as they arrive.
    """
    from nlsq.core.least_squares import prepare_bounds
    from nlsq.streaming import exact_lm

    extra = sorted(set(solver_kwargs) - _EXACT_SOLVER_KWARGS)
    if extra:
        raise TypeError(
            f"Unrecognized keyword argument(s) for method='exact_streaming': {extra}. "
            f"Accepted: {sorted(_EXACT_SOLVER_KWARGS)}.",
        )
    for name, value in unsupported.items():
        if value != _EXACT_DEFAULTS[name]:
            raise ValueError(f"{name}={value!r} {_NOT_EXACT}; leave it at its default")
    if multistart:
        raise ValueError(f"multistart=True {_NOT_EXACT} (also set by global_search)")

    factory = callable(xdata) and not hasattr(xdata, "shape")
    if factory:
        if ydata is not None:
            raise ValueError(
                "ydata must be None when xdata is a chunk factory (callable): the "
                "factory yields (x, y) or (x, y, sigma) blocks"
            )
        if sigma is not None:
            raise ValueError(
                "sigma must be None with a chunk factory: yield (x, y, sigma) blocks "
                "instead"
            )
    elif solver_kwargs.get("n_points") is not None:
        raise ValueError(
            "n_points is accepted only with a chunk factory (callable xdata); array "
            "sources take their length from their shape"
        )
    else:
        solver_kwargs = {k: v for k, v in solver_kwargs.items() if k != "n_points"}

    shapes = {}
    for name, a in (("xdata", xdata), ("ydata", ydata), ("sigma", sigma)):
        if factory or (name == "sigma" and a is None):  # factory: checked per block
            continue
        shape = a.shape if hasattr(a, "shape") else np.shape(a)
        if len(shape) != 1:
            why = (
                "; multi-feature (2-D) xdata is not supported by exact_streaming"
                if name == "xdata" and len(shape) > 1
                else ""
            )
            raise ValueError(
                f"{name} must be 1-D for method='exact_streaming', got shape {shape}{why}"
            )
        shapes[name] = shape[0]
    if not factory:
        n_x, n_y = shapes["xdata"], shapes["ydata"]
        if n_x != n_y:
            raise ValueError(
                f"`xdata` and `ydata` must have the same length: {n_x} vs {n_y}.",
            )
        if n_x < 2:
            raise ValueError(f"Need at least 2 data points for fitting, got {n_x}.")

    p0_arr = _default_p0(f) if p0 is None else np.atleast_1d(np.asarray(p0, float))
    # (p,) shapes and lb < ub elementwise; prepare_bounds only broadcasts scalars
    lb, ub = exact_lm.check_bounds(prepare_bounds(bounds, len(p0_arr)), len(p0_arr))

    res = exact_lm.exact_lm_fit(
        f,
        xdata,
        ydata,
        p0_arr,
        bounds=(lb, ub),
        chunk_size=chunk_size,
        check_finite=check_finite,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        memory_limit_gb=memory_limit_gb,
        show_progress=show_progress,
        **solver_kwargs,
    )
    if not res["success"]:
        raise exact_lm.ExactSolverError(
            f"exact_streaming fit failed ({res['termination']}): {res['message']}",
            res,
        )
    return res["x"], res["pcov"]


# Convenience function for large dataset curve fitting
def curve_fit_large(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    check_finite: bool = True,
    bounds: BoundsTuple | tuple[float, float] = (-float("inf"), float("inf")),
    method: MethodLiteral | Literal["exact_streaming"] | None = None,
    # Stability parameters
    stability: Literal["auto", "check", False] = False,
    rescale_data: bool = True,
    max_jacobian_elements_for_svd: int = 10_000_000,
    # Large dataset specific parameters
    memory_limit_gb: float | None = None,
    auto_size_detection: bool = True,
    size_threshold: int = 1_000_000,  # 1M points
    show_progress: bool = False,
    chunk_size: int | None = None,
    # Multi-start optimization parameters (Task Group 5)
    multistart: bool = False,
    n_starts: int = 10,
    global_search: bool = False,
    sampler: Literal["lhs", "sobol", "halton"] = "lhs",
    center_on_p0: bool = True,
    scale_factor: float = 1.0,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray] | OptimizeResult:
    """Curve fitting with automatic memory management for large datasets.

    Automatically selects processing strategy based on dataset size:

    - Below ``size_threshold`` (default 1M points): standard ``curve_fit``
    - At or above ``size_threshold``: chunked processing (each chunk is
      fit independently and combined via precision-weighted GLS)

    There is currently no automatic escalation to a separate "streaming"
    algorithm for very large datasets -- chunked processing is used
    regardless of how far above ``size_threshold`` the dataset is. To use
    the full-dataset streaming optimizer (``AdaptiveHybridStreamingOptimizer``,
    which accumulates Gauss-Newton statistics across the whole dataset
    instead of fitting each chunk independently) explicitly, pass
    ``method="hybrid_streaming"``.

    Parameters
    ----------
    f : callable
        Model function f(x, \\*params) -> y. Must use jax.numpy operations.
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    p0 : array_like, optional
        Initial parameter guess.
    sigma : array_like, optional
        Uncertainties in ydata for weighted fitting.
    absolute_sigma : bool, optional
        Whether sigma represents absolute uncertainties.
    check_finite : bool, optional
        Check for finite input values.
    bounds : tuple, optional
        Parameter bounds as (lower, upper).
    method : str, optional
        Optimization algorithm ('trf', 'lm', or None for auto), or
        ``'hybrid_streaming'``, or ``'exact_streaming'`` (opt-in): an exact
        full-data Levenberg-Marquardt fit whose result does not depend on
        chunking, for any dataset size (``size_threshold`` is not consulted).
        It requires a pointwise, deterministic model written with
        ``jax.numpy``, 1-D ``xdata`` (multi-feature 2-D ``xdata`` raises
        ``ValueError``) and ``multistart=False``. ``xdata``, ``ydata`` and
        ``sigma`` may be out-of-memory array-likes (``np.memmap``,
        ``h5py.Dataset``, ``zarr.Array``, any object with ``shape``, ``len``
        and slicing): they are read one chunk at a time on every pass and
        never loaded whole. ``xdata`` may instead be a replayable chunk
        factory, ``factory(chunk_size) -> Iterator[(x, y) | (x, y, sigma)]``
        called once per full-data pass (same data, same order, at most
        ``chunk_size`` points per block), with ``ydata=None``; the optional
        keyword ``n_points`` (factories only) is verified, and the point
        count is checked on every pass. 1-D ``sigma``
        and box ``bounds`` (``lb < ub`` elementwise) are supported; bounds are
        handled by an active set, not Coleman-Li TRF: the returned point is a
        KKT point of the box-constrained problem, reached by a different path
        than ``method='trf'``. ``stability``, ``rescale_data``,
        ``max_jacobian_elements_for_svd``, ``auto_size_detection`` and
        ``size_threshold`` must stay at their defaults (``ValueError``
        otherwise), and the only extra keyword arguments accepted are
        ``ftol``, ``xtol``, ``gtol``, ``max_nfev`` (full-data passes,
        default ``100 * len(p0)``) and ``n_points``. A failed fit (stalled or out of passes)
        raises :class:`nlsq.streaming.ExactSolverError` with the full result
        as ``.result``. ``pcov`` is computed from the normal equations
        ``J^T J``, which squares ``cond(J)``: expect about ``1e-6`` relative
        agreement with ``curve_fit`` up to ``cond(J) ~ 1e4``, ``1e-3`` up to
        ``1e6``, and a ``UserWarning`` beyond ``1e4``.
    memory_limit_gb : float, optional
        Maximum memory usage in GB.
    auto_size_detection : bool, optional
        Auto-detect dataset size for processing strategy.
    size_threshold : int, optional
        Point threshold for large dataset processing (default: 1M).
    show_progress : bool, optional
        Display progress bar for long operations.
    chunk_size : int, optional
        Override automatic chunk size calculation.
    multistart : bool, optional
        Enable multi-start optimization for global search. Default: False.
    n_starts : int, optional
        Number of starting points for multi-start optimization. Default: 10.
    global_search : bool, optional
        Shorthand for multistart=True, n_starts=20. Default: False.
    sampler : {'lhs', 'sobol', 'halton'}, optional
        Sampling strategy for multi-start. Default: 'lhs'.
    center_on_p0 : bool, optional
        Center multi-start samples around p0. Default: True.
    scale_factor : float, optional
        Scale factor for exploration region. Default: 1.0.
    **kwargs
        Additional optimization parameters (ftol, xtol, gtol, max_nfev, loss)

    Returns
    -------
    popt : ndarray
        Fitted parameters.
    pcov : ndarray
        Parameter covariance matrix.

    Notes
    -----
    All large datasets use streaming optimization for 100% data utilization.

    **Important: Model Function Requirements for Chunking**

    When auto_size_detection triggers chunked processing (>1M points), your model
    function MUST respect the size of xdata. Model output shape must match ydata shape.

    INCORRECT - Fixed-size output (causes shape errors):

    >>> def bad_model(xdata, a, b):
    ...     # WRONG: Returns fixed-size array regardless of xdata
    ...     t_full = jnp.arange(10_000_000)
    ...     return a * jnp.exp(-b * t_full)  # Shape mismatch!

    CORRECT - Output matches xdata size:

    >>> def good_model(xdata, a, b):
    ...     # CORRECT: Uses xdata as indices
    ...     indices = xdata.astype(jnp.int32)
    ...     return a * jnp.exp(-b * indices)

    >>> def direct_model(xdata, a, b):
    ...     # CORRECT: Operates on xdata directly
    ...     return a * jnp.exp(-b * xdata)

    Examples
    --------
    Basic usage:

    >>> popt, _pcov = curve_fit_large(model_func, xdata, ydata, p0=[1, 2, 3])

    Large dataset with progress bar:

    >>> popt, _pcov = curve_fit_large(model_func, big_xdata, big_ydata,
    ...                             show_progress=True, memory_limit_gb=8)

    With multi-start optimization:

    >>> popt, _pcov = curve_fit_large(model_func, xdata, ydata,
    ...                             p0=[1, 2, 3], bounds=([0, 0, 0], [10, 10, 10]),
    ...                             multistart=True, n_starts=10)

    Using external logger for diagnostics:

    >>> import logging
    >>> my_logger = logging.getLogger("myapp")
    >>> fitter = LargeDatasetFitter(memory_limit_gb=8, logger=my_logger)
    >>> result = fitter.fit(model_func, xdata, ydata, p0=[1, 2])
    >>> # Chunk failures now appear in myapp's logs
    """
    # Handle global_search shorthand
    if global_search:
        multistart = True
        n_starts = 20

    # Reject removed sampling parameters
    removed_params = {"enable_sampling", "sampling_threshold", "max_sampled_size"}
    for param in removed_params:
        if param in kwargs:
            raise TypeError(
                f"curve_fit_large() got an unexpected keyword argument '{param}'. "
                "This parameter was removed in v0.2.0. Use streaming instead.",
            )

    # Hoisted above np.asarray: a lazy source must not be read whole
    if method == "exact_streaming":
        return _curve_fit_exact_streaming(
            f,
            xdata,
            ydata,
            p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            memory_limit_gb=memory_limit_gb,
            show_progress=show_progress,
            chunk_size=chunk_size,
            multistart=multistart,
            unsupported={
                "stability": stability,
                "rescale_data": rescale_data,
                "max_jacobian_elements_for_svd": max_jacobian_elements_for_svd,
                "auto_size_detection": auto_size_detection,
                "size_threshold": size_threshold,
            },
            solver_kwargs=kwargs,
        )

    # Input validation
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Check for edge cases (scipy convention: 2-D xdata is (k, M), points on last axis)
    n_x = xdata.shape[-1] if xdata.ndim > 1 else len(xdata)
    if n_x == 0:
        raise ValueError("`xdata` cannot be empty.")
    if len(ydata) == 0:
        raise ValueError("`ydata` cannot be empty.")
    if n_x != len(ydata):
        raise ValueError(
            f"`xdata` and `ydata` must have the same length: {n_x} vs {len(ydata)}.",
        )
    if n_x < 2:
        raise ValueError(f"Need at least 2 data points for fitting, got {n_x}.")

    n_points = n_x

    # Handle hybrid_streaming method specially
    if method == "hybrid_streaming":
        from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        # Extract verbosity from kwargs
        verbose = kwargs.pop("verbose", 1)

        # Create configuration (allow kwargs to override defaults)
        config_overrides = {}
        for key in list(kwargs.keys()):
            if hasattr(HybridStreamingConfig, key):
                config_overrides[key] = kwargs.pop(key)

        if kwargs:
            raise TypeError(
                "Unrecognized keyword argument(s) for method="
                f"'hybrid_streaming': {sorted(kwargs)}. These are not "
                "HybridStreamingConfig fields.",
            )

        config = (
            HybridStreamingConfig(**config_overrides)
            if config_overrides
            else HybridStreamingConfig()
        )

        # Prepare p0 and bounds
        if p0 is None:
            p0 = _default_p0(f)

        p0 = np.atleast_1d(p0)
        from nlsq.core.least_squares import prepare_bounds

        lb, ub = prepare_bounds(bounds, len(p0))
        bounds_tuple = (
            (lb, ub)
            if not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))
            else None
        )

        # Create optimizer
        optimizer = AdaptiveHybridStreamingOptimizer(config=config)

        # Run optimization
        result_dict = optimizer.fit(
            data_source=(xdata, ydata),
            func=f,
            p0=p0,  # type: ignore[arg-type]
            bounds=bounds_tuple,  # type: ignore[arg-type]
            sigma=sigma,  # type: ignore[arg-type]
            absolute_sigma=absolute_sigma,
            callback=kwargs.get("callback"),
            verbose=verbose,
        )

        # Convert to tuple format for backward compatibility
        popt = result_dict["x"]
        pcov = result_dict.get("pcov", np.full((len(p0), len(p0)), np.inf))

        return popt, pcov

    # Auto-detect if we should use large dataset processing
    if auto_size_detection and n_points < size_threshold:
        # Use regular curve_fit for small datasets
        # Rebuild kwargs for curve_fit
        fit_kwargs = kwargs.copy()
        if p0 is not None:
            fit_kwargs["p0"] = p0
        if sigma is not None:
            fit_kwargs["sigma"] = sigma
        _lb, _ub = bounds
        _lb = np.atleast_1d(np.asarray(_lb, dtype=float))
        _ub = np.atleast_1d(np.asarray(_ub, dtype=float))
        if not (np.all(np.isneginf(_lb)) and np.all(np.isposinf(_ub))):
            fit_kwargs["bounds"] = bounds
        if method is not None:
            fit_kwargs["method"] = method
        fit_kwargs["absolute_sigma"] = absolute_sigma
        fit_kwargs["check_finite"] = check_finite
        fit_kwargs["stability"] = stability
        fit_kwargs["rescale_data"] = rescale_data
        fit_kwargs["max_jacobian_elements_for_svd"] = max_jacobian_elements_for_svd
        # Add multi-start parameters
        fit_kwargs["multistart"] = multistart
        fit_kwargs["n_starts"] = n_starts
        fit_kwargs["sampler"] = sampler
        fit_kwargs["center_on_p0"] = center_on_p0
        fit_kwargs["scale_factor"] = scale_factor

        return curve_fit(f, xdata, ydata, **fit_kwargs)

    # Use large dataset processing
    # Import lazy modules needed for large dataset processing
    from dataclasses import replace

    from nlsq.config import (
        LargeDatasetConfig,
        MemoryConfig,
        get_memory_config,
        large_dataset_context,
        memory_context,
    )
    from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig

    # Configure memory settings if provided
    if memory_limit_gb is None:
        # Auto-detect available memory
        try:
            import psutil

            available_gb = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = min(8.0, available_gb * 0.7)  # Use 70% of available
        except ImportError:
            memory_limit_gb = 8.0  # Conservative default

    # Create memory configuration
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    # Default ceiling: a tenth of the data, but never below the 1000-point floor
    # (or the whole input if smaller), so tiny inputs fit as one chunk, not many.
    max_chunk_size = (
        min(1_000_000, max(n_points // 10, min(1000, n_points)))
        if chunk_size is None
        else chunk_size
    )
    memory_config = MemoryConfig(
        progress_reporting=show_progress,
        # The floor must not exceed the ceiling (small n_points or chunk_size)
        min_chunk_size=min(max(1000, n_points // 10000), max_chunk_size),
        max_chunk_size=max_chunk_size,
    )

    # Create large dataset configuration (v0.2.0: no more sampling params)
    large_dataset_config = LargeDatasetConfig(
        enable_automatic_solver_selection=True,
    )

    # Use context managers to temporarily set configuration
    # Layer only the chunk bounds and progress flag over the user's global
    # settings; the fitter below gets its limit explicitly.
    context_config = replace(
        get_memory_config(),
        progress_reporting=memory_config.progress_reporting,
        min_chunk_size=memory_config.min_chunk_size,
        max_chunk_size=memory_config.max_chunk_size,
    )
    with memory_context(context_config), large_dataset_context(large_dataset_config):
        # Create fitter with current configuration
        fitter = LargeDatasetFitter(
            memory_limit_gb=memory_limit_gb,
            config=LDMemoryConfig(
                memory_limit_gb=memory_limit_gb,
                min_chunk_size=memory_config.min_chunk_size,
                max_chunk_size=memory_config.max_chunk_size,
            ),
        )

        # Handle sigma parameter by including it in kwargs if provided
        if sigma is not None:
            kwargs["sigma"] = sigma
        kwargs["absolute_sigma"] = absolute_sigma
        kwargs["check_finite"] = check_finite

        # Add multi-start parameters to kwargs
        kwargs["multistart"] = multistart
        kwargs["n_starts"] = n_starts
        kwargs["sampler"] = sampler
        kwargs["center_on_p0"] = center_on_p0
        kwargs["scale_factor"] = scale_factor

        # Convert p0 to appropriate type for LargeDatasetFitter
        # LargeDatasetFitter expects np.ndarray | list | None (no tuple or jnp.ndarray)
        converted_p0: np.ndarray | list | None
        if p0 is None:
            converted_p0 = None
        elif isinstance(p0, list):
            converted_p0 = p0
        else:
            # Convert tuple, jnp.ndarray, or np.ndarray to np.ndarray
            converted_p0 = np.asarray(p0)

        # Provide default method if None
        final_method = method if method is not None else "trf"

        # Perform the fit
        if show_progress:
            result = fitter.fit_with_progress(
                f,
                xdata,
                ydata,
                p0=converted_p0,
                bounds=bounds,
                method=final_method,
                **kwargs,  # type: ignore
            )
        else:
            result = fitter.fit(
                f,
                xdata,
                ydata,
                p0=converted_p0,
                bounds=bounds,
                method=final_method,
                **kwargs,  # type: ignore
            )

        # Extract popt and pcov from result
        if hasattr(result, "popt") and hasattr(result, "pcov"):
            return result.popt, result.pcov
        if hasattr(result, "x"):
            # Fallback: construct basic covariance matrix
            popt = result.x
            # Create identity covariance matrix if not available
            pcov = np.eye(len(popt))
            return popt, pcov
        raise RuntimeError(
            f"Unexpected result format from large dataset fitter: {result}",
        )


# Optional: Provide convenience access to submodules for advanced users
# Users can still access internal functions via:
# from nlsq.core.loss_functions import LossFunctionsJIT
# from nlsq.core.trf import TrustRegionReflective
# etc.

# Check GPU availability on import (non-intrusive warning)
# This helps users realize when GPU acceleration is available but not being used
from nlsq.device import check_gpu_availability

check_gpu_availability()

_dbg("nlsq import complete")
