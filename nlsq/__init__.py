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

import contextlib
import faulthandler
import inspect
import os
import sys

# Enable faulthandler FIRST so any SIGBUS/SIGSEGV during import prints a
# traceback to stderr instead of dying silently.  This is deliberately before
# the environment lock and all library imports below.
# sys.stderr may be None (pythonw, embedded hosts) or lack a usable fileno
# (captured streams); a missing crash dump must not break import.
with contextlib.suppress(AttributeError, ValueError, RuntimeError, OSError):
    faulthandler.enable()

_NLSQ_DEBUG = os.environ.get("NLSQ_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _dbg(msg: str) -> None:
    # sys.stderr may be None (pythonw, embedded hosts)
    if _NLSQ_DEBUG and sys.stderr is not None:
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
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ.setdefault("QT_API", "pyside6")
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    # Force software OpenGL — prevents Metal/OpenGL context conflicts that
    # cause SIGBUS on macOS 26+ Apple Silicon with PySide6 / Qt 6.10.
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

if sys.platform == "linux" and os.environ.get(
    "NLSQ_FORCE_CPU", ""
).strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:  # same truthiness as config._env_flag
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
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal, cast

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

if TYPE_CHECKING:
    from nlsq.result import CurveFitResult
    from nlsq.streaming.specs import LargeStrategy

# =============================================================================
# LAZY IMPORT SYSTEM
# Specialty modules are loaded on-demand to reduce import time.
# This achieves 50%+ reduction in cold import time (SC-001).
# =============================================================================

# Mapping of lazy export names to their source modules
_LAZY_MODULES: dict[str, str] = {
    # Result container returned by curve_fit (scipy.stats is imported with it)
    "CurveFitResult": "nlsq.result",
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
        import importlib

        try:
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
            # Keep the original text: it may come from inside the submodule's import
            raise AttributeError(
                f"Module '{module_path}' does not have attribute '{name}' ({e})",
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
    "CurveFitResult",
    "DefenseLayerTelemetry",
    "DiagnosticLevel",
    "DiagnosticPlugin",
    "DiagnosticsConfig",
    "EarlyStopping",
    "FallbackOrchestrator",
    "FallbackResult",
    "FallbackStrategy",
    "GlobalOptimizationConfig",
    "GradientHealthReport",
    "GradientMonitor",
    "HealthStatus",
    "HybridStreamingConfig",
    "IdentifiabilityAnalyzer",
    "IdentifiabilityReport",
    "InputValidator",
    "IssueCategory",
    "IssueSeverity",
    "IterationLogger",
    "LDMemoryConfig",
    "LargeDatasetConfig",
    "LargeDatasetFitter",
    "LeastSquares",
    "MemoryBudget",
    "MemoryBudgetSelector",
    "MemoryConfig",
    "MemoryManager",
    "ModelHealthIssue",
    "ModelHealthReport",
    "MultiStartOrchestrator",
    "NumericalStabilityGuard",
    "OptimizationDiagnostics",
    "OptimizationGoal",
    "OptimizationRecovery",
    "OptimizeResult",
    "OptimizeWarning",
    "ParameterNormalizer",
    "ParameterSensitivityAnalyzer",
    "ParameterSensitivityReport",
    "PluginRegistry",
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
    "create_health_report",
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
    "run_plugins",
    "set_memory_limits",
]


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
    workflow: str | None = None,
    *,
    # Multi-start parameters
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
    """Unified curve fitting function.

    Without ``workflow``, this selects ``curve_fit`` below ``size_threshold``
    points and ``curve_fit_large`` at or above it. With ``workflow`` it delegates
    to ``nlsq.core.minpack.fit`` (``"auto"``, ``"auto_global"``, ``"hpc"``, or a
    config object).

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
    workflow : str or config object, optional
        ``"auto"`` (memory-aware), ``"auto_global"`` (multi-start / CMA-ES, needs
        finite bounds), ``"hpc"``, or a ``HybridStreamingConfig`` /
        ``LDMemoryConfig`` / ``GlobalOptimizationConfig``. If None, the engine is
        chosen by point count as described above.
    goal : {'fast', 'robust', 'global', 'memory_efficient', 'quality'}, optional
        Optimization goal (case-insensitive, or an ``OptimizationGoal``). It sets
        the ``gtol``/``ftol``/``xtol`` defaults from the number of points;
        tolerances passed explicitly in ``**kwargs`` take precedence. Unknown
        names raise ValueError.
    multistart : bool, optional
        Enable multi-start. It only takes effect with ``n_starts > 0``; pass
        ``n_starts`` explicitly (``multistart=True`` alone runs a single fit below
        ``size_threshold``). Left unset at or above ``size_threshold`` it is off (the
        exact streaming default does not support multi-start); an explicit True warns
        and uses the chunked fit.
    n_starts : int, optional
        Number of starts used when ``multistart=True``. Default 0 (multi-start
        bypassed), or 5 at or above ``size_threshold``.
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
        Additional optimization parameters (ftol, xtol, gtol, max_nfev, loss)
        forwarded to the selected backend. ``strategy`` (a ``Chunked``, ``Exact``,
        ``Hybrid``, ... spec from ``nlsq.streaming``) routes the call to
        ``curve_fit_large`` at every size; it cannot be combined with the other
        engine options (``method``, ``multistart``, ``chunk_size``, ...), and the
        ``goal`` tolerances do not apply to it (set them on the strategy).

    Returns
    -------
    result : CurveFitResult or tuple
        Below ``size_threshold`` a ``CurveFitResult`` (popt, pcov,
        multistart_diagnostics, ...). At or above it, whatever ``curve_fit_large``
        returns (a plain ``(popt, pcov)`` tuple for the exact default).
        Both support tuple unpacking: popt, pcov = fit(...)

    Examples
    --------
    Basic usage:

    >>> popt, pcov = fit(model_func, xdata, ydata, p0=[1, 2, 3])

    Multi-start for robustness:

    >>> result = fit(model_func, xdata, ydata, p0=[1, 2, 3],
    ...              bounds=([0, 0, 0], [10, 10, 10]), multistart=True, n_starts=20)

    Memory-aware workflow on a large dataset:

    >>> result = fit(model_func, big_xdata, big_ydata, workflow='auto')

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

    if kwargs.pop("preset", None) is not None:  # preset=None was the old default
        raise TypeError(
            "fit() no longer accepts 'preset'. Use workflow='auto' | 'auto_global' | "
            "'hpc' (or a config object), or multistart=/n_starts= directly."
        )

    # If workflow parameter is specified, delegate to minpack.fit()
    if workflow is not None:
        import warnings as _warnings

        from nlsq.core.minpack import fit as minpack_fit

        # Options that used to be dropped silently on this path: forward the ones
        # minpack.fit honours (through its **kwargs), warn about the one it can't.
        # Only the plain "auto" workflow feeds these into curve_fit (via
        # **kwargs); the other workflows pass multistart/show_progress/... to
        # their backends explicitly, so forwarding would raise a duplicate
        # keyword TypeError. Say so instead of dropping them silently.
        requested: dict[str, Any] = {}
        if multistart is not None:
            requested["multistart"] = multistart
        if sampler != "lhs":
            requested["sampler"] = sampler
        if not center_on_p0:
            requested["center_on_p0"] = center_on_p0
        if scale_factor != 1.0:
            requested["scale_factor"] = scale_factor
        if show_progress:
            requested["show_progress"] = show_progress
        if chunk_size is not None:
            requested["chunk_size"] = chunk_size
        if size_threshold != 1_000_000:
            requested["size_threshold"] = size_threshold
        multistart_keys = ("multistart", "sampler", "center_on_p0", "scale_factor")
        wf_name = workflow.strip().lower() if isinstance(workflow, str) else None
        if wf_name == "auto":
            forwarded = {k: v for k, v in requested.items() if k in multistart_keys}
        elif wf_name == "auto_global":
            # auto_global passes multistart=True itself, but not the sampler
            # options, so those are safe to forward.
            forwarded = {k: v for k, v in requested.items() if k in multistart_keys[1:]}
        else:
            forwarded = {}
        ignored = sorted(k for k in requested if k not in forwarded)
        if ignored:
            _warnings.warn(
                f"{ignored} are not used when `workflow` is given and are "
                "ignored; configure them through the workflow config or use "
                "curve_fit()/curve_fit_large().",
                UserWarning,
                stacklevel=2,
            )
        kwargs = {**forwarded, **kwargs}

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

    # No workflow: pick curve_fit or curve_fit_large by point count.
    # Input validation
    # Out-of-core sources (h5py/zarr datasets, memmaps, chunk factories) must reach
    # curve_fit_large unread: count points from shape/len and convert only
    # objects without a shape (lists, tuples).
    from nlsq.streaming.normal_equations import is_chunk_factory

    factory = is_chunk_factory(xdata)
    if not factory:
        if not hasattr(xdata, "shape"):
            xdata = np.asarray(xdata)
        if not hasattr(ydata, "shape"):
            ydata = np.asarray(ydata)
        # scipy convention: 2-D xdata is (k, M), point count is the last axis
        n_points = xdata.shape[-1] if len(xdata.shape) > 1 else len(xdata)
    else:
        # a factory's length is unknown: it can only be streamed
        n_points = size_threshold

    # goal: validated and applied here exactly as minpack.fit does for
    # workflow=..., instead of leaking into curve_fit's ignored **kwargs.
    goal = kwargs.pop("goal", None)
    strategy = kwargs.pop("strategy", None)
    if factory and strategy is None:
        # a chunk factory can only be streamed, and only Exact streams it
        from nlsq.streaming import Exact

        strategy, chunk_size = Exact(chunk_size=chunk_size), None
    if goal is not None:
        from nlsq.core.minpack import _parse_goal_parameter
        from nlsq.core.workflow import calculate_adaptive_tolerances

        goal_tols = calculate_adaptive_tolerances(n_points, _parse_goal_parameter(goal))
        # The hybrid engine has no ftol/xtol/gtol (curve_fit and curve_fit_large
        # reject them) and an explicit strategy owns its tolerances.
        if method != "hybrid_streaming" and strategy is None:
            for tol_key in ("gtol", "ftol", "xtol"):
                if tol_key not in kwargs:
                    kwargs[tol_key] = goal_tols[tol_key]
        else:
            import warnings as _warnings

            _warnings.warn(
                "goal tolerances are not applied with "
                + ("strategy=" if strategy is not None else "method='hybrid_streaming'")
                + "; set them on the strategy / HybridStreamingConfig instead.",
                UserWarning,
                stacklevel=2,
            )

    if strategy is not None:
        # An explicit strategy owns the engine at every size; curve_fit_large
        # rejects any legacy option set alongside it, so forward only those the
        # caller changed (not fit()'s own n_starts/sampler defaults).
        changed: dict[str, Any] = {
            "method": method,
            "multistart": multistart,
            "n_starts": n_starts,
            "chunk_size": chunk_size,
        }
        changed = {k: v for k, v in changed.items() if v is not None}
        if sampler != "lhs":
            changed["sampler"] = sampler
        if not center_on_p0:
            changed["center_on_p0"] = center_on_p0
        if scale_factor != 1.0:
            changed["scale_factor"] = scale_factor
        # size_threshold only picks the engine in fit(); the strategy replaces that
        return curve_fit_large(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            memory_limit_gb=memory_limit_gb,
            show_progress=show_progress,
            strategy=strategy,
            **changed,
            **kwargs,
        )

    large = n_points >= size_threshold
    effective_multistart = bool(multistart)
    effective_n_starts: int = n_starts if n_starts is not None else (5 if large else 0)

    if large:
        # At or above size_threshold curve_fit_large defaults to exact streaming,
        # which rejects multistart; it is only on if the caller asked for it.
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

    auto_kwargs: dict[str, Any] = (
        {"memory_limit_gb": memory_limit_gb}
        if method == "auto" and memory_limit_gb is not None
        else {}
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
        # only method="auto" reads it (memory routing); other methods warn on it
        **auto_kwargs,
        **kwargs,
    )


_EXACT_DEFAULTS: dict[str, Any] = {
    "stability": False,
    "rescale_data": True,
    "max_jacobian_elements_for_svd": 10_000_000,
    "auto_size_detection": True,
    "size_threshold": 1_000_000,
}
_EXACT_SOLVER_KWARGS = frozenset({"ftol", "xtol", "gtol", "max_nfev", "n_points"})
_NOT_EXACT = "is not supported by method='exact_streaming'"


def _check_exact_legacy_options(
    unsupported: dict[str, Any], multistart: bool, solver_kwargs: dict[str, Any]
) -> None:
    """Reject options ``method="exact_streaming"`` does not support (first failure wins)."""
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


# The legacy path options of curve_fit_large, in signature order; a strategy owns them
_LEGACY_PATH_OPTIONS = (
    "method",
    "stability",
    "rescale_data",
    "max_jacobian_elements_for_svd",
    "auto_size_detection",
    "size_threshold",
    "chunk_size",
    "multistart",
    "n_starts",
    "global_search",
    "sampler",
    "center_on_p0",
    "scale_factor",
)


def _same_value(value: Any, default: Any) -> bool:
    """``value == default`` as a plain bool; ndarray values (any size) never match."""
    if isinstance(value, np.ndarray):
        return False
    try:
        return bool(value == default)
    except (TypeError, ValueError):
        return False


def _legacy_to_strategy(**legacy: Any) -> "LargeStrategy":
    """Translate curve_fit_large's legacy options (and extra kwargs) into a strategy.

    Omitted legacy path options take their ``curve_fit_large`` defaults; any
    other keyword is an engine option (``**kwargs`` of ``curve_fit_large``).
    """
    from nlsq.streaming.specs import (
        Auto,
        Chunked,
        Exact,
        FitMethod,
        Hybrid,
        InMemory,
        Multistart,
    )

    kwargs = dict(legacy)
    opts = {
        name: kwargs.pop(name, _LEGACY_DEFAULTS[name]) for name in _LEGACY_PATH_OPTIONS
    }
    method = opts["method"]
    multistart = opts["multistart"]
    n_starts = opts["n_starts"]
    if opts["global_search"]:
        multistart = True
        n_starts = 20

    if method == "exact_streaming":
        _check_exact_legacy_options(
            {name: opts[name] for name in _EXACT_DEFAULTS}, multistart, kwargs
        )
        # kwargs now holds only Exact fields (ftol, xtol, gtol, max_nfev, n_points)
        return Exact(chunk_size=opts["chunk_size"], **kwargs)

    if method == "hybrid_streaming":
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        verbose = kwargs.pop("verbose", 0)
        if kwargs.pop("callback", None) is not None:
            raise TypeError(
                "callback is not supported with method='hybrid_streaming' in "
                "curve_fit_large; use curve_fit(method='hybrid_streaming', "
                "callback=...)"
            )
        overrides = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if hasattr(HybridStreamingConfig, key)
        }
        if kwargs:
            raise TypeError(
                "Unrecognized keyword argument(s) for method="
                f"'hybrid_streaming': {sorted(kwargs)}. These are not "
                "HybridStreamingConfig fields.",
            )
        # Legacy options that have a HybridStreamingConfig field are mapped;
        # an explicit override of the config field itself wins.
        if opts["chunk_size"] is not None:
            overrides.setdefault("chunk_size", opts["chunk_size"])
        if multistart:
            overrides.setdefault("enable_multistart", True)
            overrides.setdefault("n_starts", n_starts)
            overrides.setdefault("multistart_sampler", opts["sampler"])
            overrides.setdefault("center_on_p0", opts["center_on_p0"])
            overrides.setdefault("scale_factor", opts["scale_factor"])
        # size_threshold is irrelevant with an explicit method; the rest have no
        # hybrid counterpart, so say so instead of dropping them silently.
        unused = sorted(
            name
            for name in (
                "stability",
                "rescale_data",
                "max_jacobian_elements_for_svd",
                "auto_size_detection",
            )
            if not _same_value(opts[name], _LEGACY_DEFAULTS[name])
        )
        if unused:
            import warnings

            warnings.warn(
                f"{unused} have no effect with method='hybrid_streaming' and are "
                "ignored.",
                UserWarning,
                stacklevel=4,
            )
        return Hybrid(config=HybridStreamingConfig(**overrides), verbose=verbose)

    # Always built, even when disabled: method="multi-start" uses the sampling values
    ms = Multistart(
        n_starts=n_starts,
        sampler=opts["sampler"],
        center_on_p0=opts["center_on_p0"],
        scale_factor=opts["scale_factor"],
        enabled=multistart,
    )
    fit_method = cast("FitMethod | None", method)
    return Auto(
        size_threshold=opts["size_threshold"],
        auto_size_detection=opts["auto_size_detection"],
        in_memory=InMemory(
            method=fit_method,
            stability=opts["stability"],
            rescale_data=opts["rescale_data"],
            max_jacobian_elements_for_svd=opts["max_jacobian_elements_for_svd"],
            multistart=ms,
            options=kwargs,
        ),
        # Chunked has no stability fields: the chunk fits ignore them (documented)
        chunked=Chunked(
            chunk_size=opts["chunk_size"],
            method=fit_method,
            multistart=ms,
            options=kwargs,
        ),
    )


# Convenience function for large dataset curve fitting
def curve_fit_large(
    f: ModelFunction,
    xdata: ArrayLike | Callable[[int], Iterator[tuple[Any, ...]]],
    ydata: ArrayLike | None,
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
    strategy: "LargeStrategy | None" = None,
    compute_covariance: bool = True,
    **kwargs: Any,
) -> "CurveFitResult | tuple[Any, ...]":
    """Curve fitting with automatic memory management for large datasets.

    Automatically selects the processing strategy by dataset size:

    - Below ``size_threshold`` (default 1M points): standard ``curve_fit``.
    - At or above ``size_threshold``: exact streaming Levenberg-Marquardt, the
      full-data fit of ``method="exact_streaming"``. It solves the full-batch
      least-squares problem, every iteration using the whole data set, and
      stops at a stationary point within the tolerances. The result is
      mathematically independent of the chunk size (up to floating-point
      reduction order and the stopping tolerance) and is not a GLS
      approximation.

    Pass ``strategy=Chunked()`` to get the chunked fit instead (each chunk is fit
    independently and combined by precision-weighted GLS, an approximation for
    nonlinear models). ``method="hybrid_streaming"`` selects the adaptive
    hybrid optimizer explicitly.

    The exact default needs 1-D ``xdata`` and a pointwise model (see Notes).
    Input it cannot fit is handled in one of two ways:

    - The call asks for a chunked-only feature (``method``, ``multistart`` or
      ``global_search``, or an extra keyword other than ``ftol``, ``xtol``,
      ``gtol`` and ``max_nfev``): a ``UserWarning`` is issued and the chunked
      fit is used.
    - The data cannot be fit exactly (multi-feature 2-D ``xdata``, a model that
      is not pointwise, or non-finite ``sigma``, ``xdata`` or ``ydata`` with
      ``check_finite=False``): ``ValueError`` naming ``strategy=Chunked()``. A 2-D
      ``sigma`` also raises ``ValueError``, without that hint (chunked rejects it too). Nothing falls back silently,
      so a result is never replaced by a GLS approximation without a notice.

    Parameters
    ----------
    f : callable
        Model function f(x, \\*params) -> y. Must use jax.numpy operations.
    xdata : array_like or callable
        Independent variable data. With ``method="exact_streaming"`` it may be a
        replayable chunk factory ``factory(chunk_size) -> Iterator[(x, y) | (x, y,
        sigma)]``; then ``ydata`` must be None (see ``method``).
    ydata : array_like or None
        Dependent variable data; None only when ``xdata`` is a chunk factory.
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
        full-data Levenberg-Marquardt fit whose result is mathematically
        independent of chunking, for any dataset size (``size_threshold`` is not consulted).
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
        count is checked on every pass. A scalar or 1-D ``sigma``
        (nonzero; ``|sigma|`` is used) and box ``bounds`` (``lb < ub`` elementwise) are supported; bounds are
        handled by an active set, not Coleman-Li TRF: the returned point is a
        KKT point of the box-constrained problem to within the stopping
        tolerances, reached by a different path
        than ``method='trf'``. ``stability``, ``rescale_data``,
        ``max_jacobian_elements_for_svd``, ``auto_size_detection`` and
        ``size_threshold`` must stay at their defaults (``ValueError``
        otherwise), and the only extra keyword arguments accepted are
        ``ftol``, ``xtol``, ``gtol``, ``max_nfev`` (full-data passes,
        default ``100 * len(p0)``) and ``n_points``. A failed fit (stalled or out of passes)
        raises :class:`nlsq.streaming.ExactSolverError` with the full result
        as ``.result``. ``pcov`` is computed adaptively: from the normal
        equations ``J^T J`` when ``sqrt(cond(J^T J)) <= 1e3`` (``TSQR_KAPPA``),
        otherwise by one extra TSQR pass for the ``R`` factor of ``J`` at the
        solution, which avoids squaring ``cond(J)``. A ``UserWarning`` is issued
        on the TSQR route beyond ``1e4``, where the fitted parameters can also
        stop short: check such results. ``compute_covariance=False`` skips
        both routes (and their warnings).
    stability : {'auto', 'check', False}, optional
        Stability checks, as in ``curve_fit``: 'auto' and 'check' warn without
        modifying data, False skips the checks. Only the in-memory path applies it;
        the default exact path and the chunked path ignore it without a warning
        (see "Options ignored by chunked processing"). Must stay at its default
        with ``method="exact_streaming"``; ignored with a warning by
        ``method="hybrid_streaming"``. Default: False.
    rescale_data : bool, optional
        As in ``curve_fit``: no data is rescaled; with ``stability='auto'`` it only
        controls a warning. Same restrictions as ``stability``. Default: True.
    max_jacobian_elements_for_svd : int, optional
        Largest Jacobian (m x n elements) for which the stability checks compute
        an SVD, as in ``curve_fit``. Same restrictions as ``stability``.
        Default: 10,000,000.
    memory_limit_gb : float, optional
        Maximum memory usage in GB.
    auto_size_detection : bool, optional
        If False, always take the large path: exact, except that engine-only
        options fall back to chunked with a warning and other ineligible input
        raises ``ValueError``.
        Default: True (choose by ``size_threshold``).
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
    strategy : Auto, InMemory, Chunked, Hybrid or Exact, optional
        An explicit fitting strategy from ``nlsq.streaming``. Default ``None``
        builds one from the options above, as before. When given, the
        strategy owns every path option: ``method``, ``stability``,
        ``rescale_data``, ``max_jacobian_elements_for_svd``,
        ``auto_size_detection``, ``size_threshold``, ``chunk_size``,
        ``multistart``, ``n_starts``, ``global_search``, ``sampler``,
        ``center_on_p0`` and ``scale_factor`` must stay at their defaults and
        no extra keyword arguments may be passed (``TypeError`` otherwise).
    compute_covariance : bool, optional
        If False, ``pcov`` is returned as None. In-memory path: ``curve_fit`` skips
        the covariance. Exact path: the final covariance is skipped. Chunked path:
        no saving, because the per-chunk covariances are still computed (the GLS
        combiner needs them) and only the final ``pcov`` is dropped.
        ``method="hybrid_streaming"`` raises ``ValueError``. Default: True.
    **kwargs
        Additional optimization parameters (ftol, xtol, gtol, max_nfev, loss). With
        ``method="exact_streaming"`` only ftol, xtol, gtol, max_nfev and n_points are
        accepted (``loss`` raises ``TypeError``). With ``method="hybrid_streaming"``
        only ``HybridStreamingConfig`` fields and ``verbose`` are accepted (ftol,
        xtol, gtol and max_nfev raise ``TypeError``); ``chunk_size`` and
        ``multistart``/``n_starts`` map to the config's ``chunk_size`` and
        ``enable_multistart``/``n_starts``.

    Returns
    -------
    popt, pcov : ndarray
        The chunked, ``hybrid_streaming`` and ``exact_streaming`` paths return
        the tuple ``(popt, pcov)``: fitted parameters and their covariance
        (``pcov`` is None when ``compute_covariance=False``).
    result : CurveFitResult or tuple
        The in-memory path returns whatever ``curve_fit`` returns: a
        ``CurveFitResult`` (which unpacks as ``popt, pcov``), or the special-output
        tuple when options such as ``full_output=True`` are passed.

    Notes
    -----
    All large datasets use streaming optimization for 100% data utilization.

    **Exact default (at or above** ``size_threshold`` **)**

    - ``sigma`` may be a 1-D array, a scalar, or contain negative nonzero values
      (the objective uses ``(r / sigma)**2``, so ``|sigma|`` is used).
    - ``bounds`` are handled by an active set, as for ``method="exact_streaming"``.
    - ``max_nfev`` counts full-data passes (default ``100 * len(p0)``), not
      function evaluations. ``ftol``, ``xtol`` and ``gtol`` are accepted, and
      ``chunk_size`` sets the pass block size.
    - A stalled fit or one that runs out of passes raises
      :class:`nlsq.streaming.ExactSolverError` (the result is in ``.result``)
      instead of returning an unconverged point. There is no retry on the
      chunked fit; the message names ``strategy=Chunked()``.
    - The model must be pointwise: ``f(x, *p)[i]`` depends only on ``x[i]`` and
      ``p``. A probe on the first points raises ``ValueError`` for a model that
      mixes points, but it is a guard, not a proof: a model that mixes points
      beyond the probe is not detected and gives a wrong fit.
    - ``strategy=Chunked()`` restores the previous chunked behaviour, including
      ``method``. Chunked ignores ``multistart`` (see below).

    **Options ignored by chunked processing**

    The chunked path (selected with ``strategy=Chunked()`` or by the fallback
    above) does not forward ``stability``, ``rescale_data`` or
    ``max_jacobian_elements_for_svd`` to the per-chunk fits, and the multi-start
    options (``multistart``, ``n_starts``, ``global_search``, ``sampler``,
    ``center_on_p0``, ``scale_factor``) have no effect on them. These options
    only take effect on the in-memory path (fewer than ``size_threshold``
    points).

    **Explicit strategies**

    Without ``strategy``, the options are translated into
    ``Auto(size_threshold, auto_size_detection, in_memory=InMemory(...),
    chunked=Chunked(...), exact=Exact())``, ``Hybrid(config, verbose)`` for
    ``method="hybrid_streaming"`` or ``Exact(...)`` for
    ``method="exact_streaming"``. ``Auto.exact`` takes ``chunk_size`` from
    ``Chunked`` and ``ftol``, ``xtol``, ``gtol`` and ``max_nfev`` from the call
    unless ``Exact`` sets them. ``Auto(exact=None)`` restores the chunked fit
    above the threshold. Passing the strategy directly gives the same fit with
    the settings in one object.

    **Important: Model Function Requirements for Chunking**

    When the large path is used (>=1M points by default), your model function
    MUST respect the size of xdata and be pointwise. Model output shape must match
    ydata shape.

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

    With multi-start optimization (below ``size_threshold``; at or above it,
    ``multistart=True`` forces the chunked fallback with a ``UserWarning``):

    >>> popt, _pcov = curve_fit_large(model_func, xdata, ydata,
    ...                             p0=[1, 2, 3], bounds=([0, 0, 0], [10, 10, 10]),
    ...                             multistart=True, n_starts=10)

    Opt out of the exact default and use the chunked fit, 500 points per chunk:

    >>> from nlsq.streaming import Chunked
    >>> popt, _pcov = curve_fit_large(model_func, xdata, ydata, p0=[1, 2, 3],
    ...                             strategy=Chunked(chunk_size=500))

    Using external logger for diagnostics:

    >>> import logging
    >>> my_logger = logging.getLogger("myapp")
    >>> fitter = LargeDatasetFitter(memory_limit_gb=8, logger=my_logger)
    >>> result = fitter.fit(model_func, xdata, ydata, p0=[1, 2])
    >>> # Chunk failures now appear in myapp's logs
    """
    # Reject removed sampling parameters
    removed_params = {"enable_sampling", "sampling_threshold", "max_sampled_size"}
    for param in removed_params:
        if param in kwargs:
            raise TypeError(
                f"curve_fit_large() got an unexpected keyword argument '{param}'. "
                "This parameter was removed in v0.2.0. Use streaming instead.",
            )

    from nlsq.core.minpack import _CURVE_FIT_METHODS, _check_method

    _check_method(method, (*_CURVE_FIT_METHODS, "exact_streaming"))

    legacy = {
        "method": method,
        "stability": stability,
        "rescale_data": rescale_data,
        "max_jacobian_elements_for_svd": max_jacobian_elements_for_svd,
        "auto_size_detection": auto_size_detection,
        "size_threshold": size_threshold,
        "chunk_size": chunk_size,
        "multistart": multistart,
        "n_starts": n_starts,
        "global_search": global_search,
        "sampler": sampler,
        "center_on_p0": center_on_p0,
        "scale_factor": scale_factor,
    }
    if strategy is None:
        strategy = _legacy_to_strategy(**legacy, **kwargs)
    else:
        # The strategy owns every path option; detection is by default value.
        for name, value in legacy.items():
            if not _same_value(value, _LEGACY_DEFAULTS[name]):
                raise TypeError(
                    f"curve_fit_large() got {name}={value!r} together with "
                    "strategy=; set it on the strategy instead",
                )
        if kwargs:
            raise TypeError(
                f"curve_fit_large() got keyword argument(s) {sorted(kwargs)} "
                "together with strategy=; set them on the strategy instead",
            )

    from nlsq.core.minpack import _bounds_as_tuple
    from nlsq.core.problem import coerce_real_sigma, reject_all_zero_weight
    from nlsq.streaming.large_fit import fit_with_strategy

    sigma = coerce_real_sigma(sigma)
    reject_all_zero_weight(sigma, kwargs.get("data_mask"))
    bounds = _bounds_as_tuple(bounds)

    return fit_with_strategy(
        strategy,
        f,
        xdata,
        ydata,
        p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        bounds=bounds,
        check_finite=check_finite,
        memory_limit_gb=memory_limit_gb,
        show_progress=show_progress,
        compute_covariance=compute_covariance,
    )


# Read from the signature so the defaults are written down once
_LEGACY_DEFAULTS: dict[str, Any] = {
    name: param.default
    for name, param in inspect.signature(curve_fit_large).parameters.items()
    if name in _LEGACY_PATH_OPTIONS
}


# Optional: Provide convenience access to submodules for advanced users
# Users can still access internal functions via:
# from nlsq.core.loss_functions import LossFunctionsJIT
# from nlsq.core.trf import TrustRegionReflective
# etc.

# Check GPU availability on import (non-intrusive warning)
# This helps users realize when GPU acceleration is available but not being used
from nlsq.device import check_gpu_availability

check_gpu_availability(warn_on_error=False)

_dbg("nlsq import complete")
