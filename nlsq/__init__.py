"""
NLSQ: JAX-accelerated nonlinear least squares curve fitting.

GPU/TPU-accelerated curve fitting with automatic differentiation.
Provides drop-in SciPy compatibility with curve_fit function.
Supports large datasets through automatic chunking and sampling.

Examples
--------
>>> import jax.numpy as jnp
>>> from nlsq import curve_fit
>>> def model(x, a, b): return a * jnp.exp(-b * x)
>>> popt, _pcov = curve_fit(model, xdata, ydata)

"""

# Version information
try:
    from nlsq._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

# Main API imports
# Common functions library (Sprint 4 - User Experience)
# Progress callbacks (Day 3 - User Experience)
from nlsq import callbacks, functions
from nlsq._optimize import OptimizeResult, OptimizeWarning

# Stability and optimization imports
from nlsq.algorithm_selector import AlgorithmSelector, auto_select_algorithm

# Bounds inference (Phase 3 - Day 17)
from nlsq.bound_inference import BoundsInference, infer_bounds, merge_bounds

# Performance optimization modules (Sprint 2)
from nlsq.compilation_cache import (
    CompilationCache,
    cached_jit,
    clear_compilation_cache,
    get_global_compilation_cache,
)

# Configuration support
from nlsq.config import (
    LargeDatasetConfig,
    MemoryConfig,
    configure_for_large_datasets,
    enable_mixed_precision_fallback,
    get_large_dataset_config,
    get_memory_config,
    large_dataset_context,
    memory_context,
    set_memory_limits,
)
from nlsq.diagnostics import ConvergenceMonitor, OptimizationDiagnostics

# Fallback strategies (Phase 3 - Days 15-16)
from nlsq.fallback import FallbackOrchestrator, FallbackResult, FallbackStrategy

# Large dataset support
from nlsq.large_dataset import (
    LargeDatasetFitter,
    LDMemoryConfig,  # Use renamed class to avoid conflicts with config.py MemoryConfig
    estimate_memory_requirements,
    fit_large_dataset,
)
from nlsq.least_squares import LeastSquares
from nlsq.memory_manager import (
    MemoryManager,
    clear_memory_pool,
    get_memory_manager,
    get_memory_stats,
)
from nlsq.memory_pool import (
    MemoryPool,
    TRFMemoryPool,
    clear_global_pool,
    get_global_pool,
)
from nlsq.minpack import CurveFit, curve_fit

# Performance profiling (Days 20-21)
from nlsq.profiler import (
    PerformanceProfiler,
    ProfileMetrics,
    clear_profiling_data,
    get_global_profiler,
)

# Performance profiling visualization (Days 22-23)
from nlsq.profiler_visualization import (
    ProfilerVisualization,
    ProfilingDashboard,
)
from nlsq.recovery import OptimizationRecovery
from nlsq.robust_decomposition import RobustDecomposition, robust_decomp
from nlsq.smart_cache import (
    SmartCache,
    cached_function,
    cached_jacobian,
    clear_all_caches,
    get_global_cache,
    get_jit_cache,
)

# Sparse Jacobian support
from nlsq.sparse_jacobian import (
    SparseJacobianComputer,
    SparseOptimizer,
    detect_jacobian_sparsity,
)

# Stability checks (Phase 3 - Day 18)
from nlsq.stability import (
    NumericalStabilityGuard,
    apply_automatic_fixes,
    check_problem_stability,
    detect_collinearity,
    detect_parameter_scale_mismatch,
    estimate_condition_number,
)

# Streaming optimizer support (requires h5py - optional dependency)
try:
    from nlsq.streaming_optimizer import (
        DataGenerator,
        StreamingConfig,
        StreamingOptimizer,
        create_hdf5_dataset,
        fit_unlimited_data,
    )

    _HAS_STREAMING = True
except ImportError:
    # h5py not available - streaming features disabled
    _HAS_STREAMING = False

from nlsq.validators import InputValidator

# Public API - only expose main user-facing functions
__all__ = [
    # Stability and optimization modules
    "AlgorithmSelector",
    # Bounds inference (Phase 3)
    "BoundsInference",
    # Performance optimization (Sprint 2)
    "CompilationCache",
    "ConvergenceMonitor",
    "CurveFit",
    # Fallback strategies (Phase 3)
    "FallbackOrchestrator",
    "FallbackResult",
    "FallbackStrategy",
    "InputValidator",
    "LargeDatasetConfig",
    "LargeDatasetFitter",
    # Advanced API
    "LeastSquares",
    # Configuration classes
    "MemoryConfig",
    "MemoryManager",
    "MemoryPool",
    "NumericalStabilityGuard",
    "OptimizationDiagnostics",
    "OptimizationRecovery",
    # Result types
    "OptimizeResult",
    "OptimizeWarning",
    # Performance profiling (Days 20-21)
    "PerformanceProfiler",
    "ProfileMetrics",
    # Performance profiling visualization (Days 22-23)
    "ProfilerVisualization",
    "ProfilingDashboard",
    "RobustDecomposition",
    "SmartCache",
    # Sparse Jacobian support
    "SparseJacobianComputer",
    "SparseOptimizer",
    "TRFMemoryPool",
    # Version
    "__version__",
    "apply_automatic_fixes",
    "auto_select_algorithm",
    # Caching support
    "cached_function",
    "cached_jacobian",
    "cached_jit",
    # Progress callbacks (Day 3)
    "callbacks",
    # Stability checks (Phase 3)
    "check_problem_stability",
    "clear_all_caches",
    "clear_compilation_cache",
    "clear_global_pool",
    "clear_memory_pool",
    "clear_profiling_data",
    # Configuration functions
    "configure_for_large_datasets",
    # Main curve fitting API
    "curve_fit",
    "curve_fit_large",
    "detect_collinearity",
    "detect_jacobian_sparsity",
    "detect_parameter_scale_mismatch",
    "enable_mixed_precision_fallback",
    "estimate_condition_number",
    "estimate_memory_requirements",
    # Large dataset utilities
    "fit_large_dataset",
    # Common functions library
    "functions",
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
    "robust_decomp",
    "set_memory_limits",
]

# Add streaming features to public API if h5py is available
if _HAS_STREAMING:
    __all__.extend(
        [
            "DataGenerator",
            "StreamingConfig",
            "StreamingOptimizer",
            "create_hdf5_dataset",
            "fit_unlimited_data",
        ]
    )


# Convenience function for large dataset curve fitting
def curve_fit_large(
    f,
    xdata,
    ydata,
    p0=None,
    sigma=None,
    absolute_sigma=False,
    check_finite=True,
    bounds=(-float("inf"), float("inf")),
    method=None,
    # Large dataset specific parameters
    memory_limit_gb=None,
    auto_size_detection=True,
    size_threshold=1_000_000,  # 1M points
    show_progress=False,
    enable_sampling=True,
    sampling_threshold=100_000_000,
    max_sampled_size=10_000_000,
    chunk_size=None,
    **kwargs,
):
    """Curve fitting with automatic memory management for large datasets.

    Automatically selects processing strategy based on dataset size:
    - Small (< 1M points): Standard curve_fit
    - Medium (1M - 100M points): Chunked processing
    - Large (> 100M points): Sampling with chunking

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
    enable_sampling : bool, optional
        Enable sampling for extremely large datasets.
    sampling_threshold : int, optional
        Point threshold for sampling (default: 100M).
    max_sampled_size : int, optional
        Maximum sample size when sampling (default: 10M).
    chunk_size : int, optional
        Override automatic chunk size calculation.
    **kwargs
        Additional optimization parameters (ftol, xtol, gtol, max_nfev, loss)

    Returns
    -------
    popt : ndarray
        Fitted parameters.
    pcov : ndarray
        Parameter covariance matrix.

    Examples
    --------
    Basic usage:

    >>> popt, _pcov = curve_fit_large(model_func, xdata, ydata, p0=[1, 2, 3])

    Large dataset with progress bar:

    >>> popt, _pcov = curve_fit_large(model_func, big_xdata, big_ydata,
    ...                             show_progress=True, memory_limit_gb=8)
    """
    import numpy as np

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

        return curve_fit(f, xdata, ydata, **fit_kwargs)

    # Use large dataset processing
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

    # Create large dataset configuration
    large_dataset_config = LargeDatasetConfig(
        enable_sampling=enable_sampling,
        sampling_threshold=sampling_threshold,
        max_sampled_size=max_sampled_size,
    )

    # Use context managers to temporarily set configuration
    with memory_context(memory_config), large_dataset_context(large_dataset_config):
        # Create fitter with current configuration
        fitter = LargeDatasetFitter(
            memory_limit_gb=memory_limit_gb,
            config=LDMemoryConfig(
                memory_limit_gb=memory_limit_gb,
                enable_sampling=enable_sampling,
                sampling_threshold=sampling_threshold,
                max_sampled_size=max_sampled_size,
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

        # Perform the fit
        if show_progress:
            result = fitter.fit_with_progress(
                f, xdata, ydata, p0=p0, bounds=bounds, method=method, **kwargs
            )
        else:
            result = fitter.fit(
                f, xdata, ydata, p0=p0, bounds=bounds, method=method, **kwargs
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
# from nlsq.loss_functions import LossFunctionsJIT
# from nlsq.trf import TrustRegionReflective
# etc.
