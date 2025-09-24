"""
NLSQ: Nonlinear Least Squares Curve Fitting for GPU/TPU

A JAX-based implementation of curve fitting algorithms with automatic
differentiation and GPU/TPU acceleration.
"""

# Version information
try:
    from nlsq._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

# Main API imports
from nlsq._optimize import OptimizeResult, OptimizeWarning

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

# Large dataset support
from nlsq.large_dataset import (
    LargeDatasetFitter,
    LDMemoryConfig,  # Use renamed class to avoid conflicts with config.py MemoryConfig
    estimate_memory_requirements,
    fit_large_dataset,
)
from nlsq.least_squares import LeastSquares
from nlsq.minpack import CurveFit, curve_fit

# Sparse Jacobian support
from nlsq.sparse_jacobian import (
    SparseJacobianComputer,
    SparseOptimizer,
    detect_jacobian_sparsity,
)

# Streaming optimizer support
from nlsq.streaming_optimizer import (
    DataGenerator,
    StreamingConfig,
    StreamingOptimizer,
    create_hdf5_dataset,
    fit_unlimited_data,
)

# Public API - only expose main user-facing functions
__all__ = [
    # Main curve fitting API
    "curve_fit",
    "curve_fit_large",
    "CurveFit",
    # Advanced API
    "LeastSquares",
    "LargeDatasetFitter",
    # Sparse Jacobian support
    "SparseJacobianComputer",
    "SparseOptimizer",
    "detect_jacobian_sparsity",
    # Streaming optimizer support
    "StreamingOptimizer",
    "StreamingConfig",
    "DataGenerator",
    "fit_unlimited_data",
    "create_hdf5_dataset",
    # Configuration classes
    "MemoryConfig",
    "LargeDatasetConfig",
    # Result types
    "OptimizeResult",
    "OptimizeWarning",
    # Version
    "__version__",
    # Large dataset utilities
    "fit_large_dataset",
    "estimate_memory_requirements",
    # Configuration functions
    "configure_for_large_datasets",
    "set_memory_limits",
    "enable_mixed_precision_fallback",
    "memory_context",
    "large_dataset_context",
    "get_memory_config",
    "get_large_dataset_config",
]


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
    """Curve fitting with automatic large dataset handling.

    This function provides a drop-in replacement for `curve_fit` with automatic
    detection and handling of large datasets. For small datasets (< 1M points),
    it behaves identically to `curve_fit`. For larger datasets, it automatically
    switches to memory-efficient processing with chunking, sampling, and progress
    reporting.

    Parameters
    ----------
    f : callable
        The model function f(x, *params) -> y
    xdata : np.ndarray
        Independent variable data
    ydata : np.ndarray
        Dependent variable data
    p0 : array-like, optional
        Initial parameter guess
    sigma : array-like, optional
        Uncertainties in ydata (for weighted fitting)
    absolute_sigma : bool, optional
        Whether sigma represents absolute uncertainties (default: False)
    check_finite : bool, optional
        Whether to check for finite values in inputs (default: True)
    bounds : tuple, optional
        Parameter bounds as (lower, upper) (default: (-inf, inf))
    method : str, optional
        Optimization method (default: None for automatic selection)
    memory_limit_gb : float, optional
        Memory limit in GB (default: None for automatic detection)
    auto_size_detection : bool, optional
        Whether to automatically detect dataset size and switch methods (default: True)
    size_threshold : int, optional
        Point count threshold for switching to large dataset processing (default: 1M)
    show_progress : bool, optional
        Whether to show progress for large dataset processing (default: False)
    enable_sampling : bool, optional
        Whether to enable sampling for extremely large datasets (default: True)
    sampling_threshold : int, optional
        Point count threshold above which sampling is considered (default: 100M)
    max_sampled_size : int, optional
        Maximum size when sampling is enabled (default: 10M)
    chunk_size : int, optional
        Override automatic chunk size calculation
    **kwargs
        Additional arguments passed to the underlying curve_fit implementation

    Returns
    -------
    popt : np.ndarray
        Optimal parameters
    pcov : np.ndarray
        Covariance matrix of the parameters

    Examples
    --------
    Small dataset (uses regular curve_fit):

    >>> x = np.linspace(0, 4, 50)
    >>> y = 2.5 * np.exp(-1.3 * x) + noise
    >>> popt, pcov = curve_fit_large(lambda x, a, b: a * np.exp(-b * x), x, y)

    Large dataset (automatic large dataset processing):

    >>> x_large = np.linspace(0, 10, 10_000_000)
    >>> y_large = 2.5 * np.exp(-1.3 * x_large) + noise
    >>> popt, pcov = curve_fit_large(
    ...     lambda x, a, b: a * np.exp(-b * x),
    ...     x_large, y_large,
    ...     memory_limit_gb=8.0,
    ...     show_progress=True
    ... )

    Custom configuration for very large datasets:

    >>> popt, pcov = curve_fit_large(
    ...     func, x_very_large, y_very_large,
    ...     memory_limit_gb=16.0,
    ...     enable_sampling=True,
    ...     sampling_threshold=50_000_000,
    ...     show_progress=True
    ... )
    """
    import numpy as np

    # Input validation
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Check for edge cases
    if len(xdata) == 0:
        raise ValueError("xdata cannot be empty")
    if len(ydata) == 0:
        raise ValueError("ydata cannot be empty")
    if len(xdata) != len(ydata):
        raise ValueError(f"xdata and ydata must have the same length: got {len(xdata)} vs {len(ydata)}")
    if len(xdata) < 2:
        raise ValueError(f"Need at least 2 data points for fitting, got {len(xdata)}")

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
