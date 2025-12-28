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

# =============================================================================
# CORE IMPORTS (Always Eager - Required for basic curve_fit usage)
# =============================================================================

# Version information
try:
    from nlsq._version import __version__  # type: ignore[import-not-found]
except ImportError:
    __version__ = "0.0.0+unknown"

# Standard library imports needed at module level
import warnings  # For deprecation warnings (v0.2.0)
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

# Core API - always needed for basic functionality
from nlsq import callbacks
from nlsq.core import functions
from nlsq.core._optimize import OptimizeResult, OptimizeWarning
from nlsq.core.least_squares import LeastSquares
from nlsq.core.minpack import CurveFit, curve_fit
from nlsq.types import ArrayLike, BoundsTuple, MethodLiteral, ModelFunction

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
    "StreamingOptimizer": "nlsq.streaming.optimizer",
    "StreamingConfig": "nlsq.streaming.optimizer",
    "DataGenerator": "nlsq.streaming.optimizer",
    "create_hdf5_dataset": "nlsq.streaming.optimizer",
    "fit_unlimited_data": "nlsq.streaming.optimizer",
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
    "MemoryPool": "nlsq.caching.memory_pool",
    "TRFMemoryPool": "nlsq.caching.memory_pool",
    "clear_global_pool": "nlsq.caching.memory_pool",
    "get_global_pool": "nlsq.caching.memory_pool",
    # Fallback & Recovery
    "FallbackOrchestrator": "nlsq.stability.fallback",
    "FallbackResult": "nlsq.stability.fallback",
    "FallbackStrategy": "nlsq.stability.fallback",
    "OptimizationRecovery": "nlsq.stability.recovery",
    # Sparse Jacobian
    "SparseJacobianComputer": "nlsq.core.sparse_jacobian",
    "SparseOptimizer": "nlsq.core.sparse_jacobian",
    "detect_jacobian_sparsity": "nlsq.core.sparse_jacobian",
    # Workflow System
    "WORKFLOW_PRESETS": "nlsq.core.workflow",
    "DatasetSizeTier": "nlsq.core.workflow",
    "MemoryTier": "nlsq.core.workflow",
    "OptimizationGoal": "nlsq.core.workflow",
    "WorkflowConfig": "nlsq.core.workflow",
    "WorkflowSelector": "nlsq.core.workflow",
    "WorkflowTier": "nlsq.core.workflow",
    "auto_select_workflow": "nlsq.core.workflow",
    # Algorithm Selection
    "AlgorithmSelector": "nlsq.precision.algorithm_selector",
    "auto_select_algorithm": "nlsq.precision.algorithm_selector",
    # Bounds Inference
    "BoundsInference": "nlsq.precision.bound_inference",
    "infer_bounds": "nlsq.precision.bound_inference",
    "merge_bounds": "nlsq.precision.bound_inference",
    # Robust Decomposition
    "RobustDecomposition": "nlsq.stability.robust_decomposition",
    "robust_decomp": "nlsq.stability.robust_decomposition",
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
    "enable_mixed_precision_fallback": "nlsq.config",
    "get_large_dataset_config": "nlsq.config",
    "get_memory_config": "nlsq.config",
    "large_dataset_context": "nlsq.config",
    "memory_context": "nlsq.config",
    "set_memory_limits": "nlsq.config",
    # Caching
    "SmartCache": "nlsq.caching.smart_cache",
    "cached_function": "nlsq.caching.smart_cache",
    "cached_jacobian": "nlsq.caching.smart_cache",
    "clear_all_caches": "nlsq.caching.smart_cache",
    "get_global_cache": "nlsq.caching.smart_cache",
    "get_jit_cache": "nlsq.caching.smart_cache",
    # Compilation Cache
    "CompilationCache": "nlsq.caching.compilation_cache",
    "cached_jit": "nlsq.caching.compilation_cache",
    "clear_compilation_cache": "nlsq.caching.compilation_cache",
    "get_global_compilation_cache": "nlsq.caching.compilation_cache",
    # Validators
    "InputValidator": "nlsq.utils.validators",
}

# Cache for lazily-loaded attributes
_LAZY_CACHE: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazily load specialty modules on first access.

    This enables faster import time by deferring loading of specialty
    modules until they are actually needed.
    """
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
            # Cache for future access
            _LAZY_CACHE[name] = attr
            return attr
        except ImportError as e:
            # Handle missing optional dependencies gracefully
            raise ImportError(
                f"Cannot import '{name}' from '{module_path}'. "
                f"This may require an optional dependency. Error: {e}"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Module '{module_path}' does not have attribute '{name}'"
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
    "WORKFLOW_PRESETS",
    "AdaptiveHybridStreamingOptimizer",
    "AlgorithmSelector",
    "BoundsInference",
    "CompilationCache",
    "ConvergenceMonitor",
    "CurveFit",
    "DatasetSizeTier",
    "DefenseLayerTelemetry",
    "FallbackOrchestrator",
    "FallbackResult",
    "FallbackStrategy",
    "GlobalOptimizationConfig",
    "HybridStreamingConfig",
    "InputValidator",
    "LargeDatasetConfig",
    "LargeDatasetFitter",
    "LeastSquares",
    "MemoryConfig",
    "MemoryManager",
    "MemoryPool",
    "MemoryTier",
    "MultiStartOrchestrator",
    "NumericalStabilityGuard",
    "OptimizationDiagnostics",
    "OptimizationGoal",
    "OptimizationRecovery",
    "OptimizeResult",
    "OptimizeWarning",
    "ParameterNormalizer",
    "PerformanceProfiler",
    "ProfileMetrics",
    "ProfilerVisualization",
    "ProfilingDashboard",
    "RobustDecomposition",
    "SmartCache",
    "SparseJacobianComputer",
    "SparseOptimizer",
    "TRFMemoryPool",
    "TournamentSelector",
    "WorkflowConfig",
    "WorkflowSelector",
    "WorkflowTier",
    "__version__",
    "apply_automatic_fixes",
    "auto_select_algorithm",
    "auto_select_workflow",
    "cached_function",
    "cached_jacobian",
    "cached_jit",
    "callbacks",
    "check_problem_stability",
    "clear_all_caches",
    "clear_compilation_cache",
    "clear_global_pool",
    "clear_memory_pool",
    "clear_profiling_data",
    "configure_for_large_datasets",
    "curve_fit",
    "curve_fit_large",
    "detect_collinearity",
    "detect_jacobian_sparsity",
    "detect_parameter_scale_mismatch",
    "enable_mixed_precision_fallback",
    "estimate_condition_number",
    "estimate_memory_requirements",
    "fit",
    "fit_large_dataset",
    "functions",
    "get_defense_telemetry",
    "get_global_cache",
    "get_global_compilation_cache",
    "get_global_pool",
    "get_global_profiler",
    "get_jit_cache",
    "get_large_dataset_config",
    "get_memory_config",
    "get_memory_manager",
    "get_memory_stats",
    "infer_bounds",
    "large_dataset_context",
    "memory_context",
    "merge_bounds",
    "reset_defense_telemetry",
    "robust_decomp",
    "set_memory_limits",
]

# Add streaming features to public API
# Streaming modules are lazily loaded; add them to __all__ unconditionally
# (the ImportError is raised lazily when accessed if h5py is missing)
__all__.extend(
    [
        "DataGenerator",
        "StreamingConfig",
        "StreamingOptimizer",
        "create_hdf5_dataset",
        "fit_unlimited_data",
    ]
)


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
    auto_select_workflow : Automatic workflow selection function
    WorkflowSelector : Class-based workflow selection
    """
    # Input validation
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    n_points = len(xdata)

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

    elif use_large_dataset or n_points >= size_threshold:
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

    else:
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
    method: MethodLiteral | None = None,
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
    - Small (< 1M points): Standard curve_fit
    - Medium (1M - 100M points): Chunked processing
    - Large (> 100M points): Streaming optimization

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

    Important: Model Function Requirements for Chunking
    ----------------------------------------------------
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
    import numpy as np

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
                "This parameter was removed in v0.2.0. Use streaming instead."
            )

    # Input validation
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Check for edge cases
    if len(xdata) == 0:
        raise ValueError("`xdata` cannot be empty.")
    if len(ydata) == 0:
        raise ValueError("`ydata` cannot be empty.")
    if len(xdata) != len(ydata):
        raise ValueError(
            f"`xdata` and `ydata` must have the same length: {len(xdata)} vs {len(ydata)}."
        )
    if len(xdata) < 2:
        raise ValueError(f"Need at least 2 data points for fitting, got {len(xdata)}.")

    n_points = len(xdata)

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

        config = (
            HybridStreamingConfig(**config_overrides)
            if config_overrides
            else HybridStreamingConfig()
        )

        # Prepare p0 and bounds
        if p0 is None:
            from inspect import signature

            sig = signature(f)
            args = sig.parameters
            if len(args) < 2:
                raise ValueError("Unable to determine number of fit parameters.")
            n_params = len(args) - 1
            p0 = np.ones(n_params)

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
        if bounds != (-float("inf"), float("inf")):
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
    from nlsq.config import (
        LargeDatasetConfig,
        MemoryConfig,
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
    memory_config = MemoryConfig(
        memory_limit_gb=memory_limit_gb,
        progress_reporting=show_progress,
        min_chunk_size=max(1000, n_points // 10000),  # Dynamic min chunk size
        max_chunk_size=min(1_000_000, n_points // 10)
        if chunk_size is None
        else chunk_size,
    )

    # Create large dataset configuration (v0.2.0: no more sampling params)
    large_dataset_config = LargeDatasetConfig(
        enable_automatic_solver_selection=True,
    )

    # Use context managers to temporarily set configuration
    with memory_context(memory_config), large_dataset_context(large_dataset_config):
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
        if not absolute_sigma:
            kwargs["absolute_sigma"] = absolute_sigma
        if not check_finite:
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
        elif hasattr(result, "x"):
            # Fallback: construct basic covariance matrix
            popt = result.x
            # Create identity covariance matrix if not available
            pcov = np.eye(len(popt))
            return popt, pcov
        else:
            raise RuntimeError(
                f"Unexpected result format from large dataset fitter: {result}"
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
