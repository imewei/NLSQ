"""Fitting execution adapter for NLSQ GUI.

This module provides a GUI-friendly wrapper around nlsq.minpack.fit()
for executing curve fitting operations from the Streamlit interface.

The adapter handles:
- Converting GUI state to fit parameters
- Progress callback integration for real-time updates
- Result extraction and statistics computation
- Error handling for failed fits
- Support for streaming optimizer with large datasets

Functions
---------
execute_fit
    Execute a curve fit with the given parameters.
extract_fit_statistics
    Extract fit quality statistics from a result.
extract_convergence_info
    Extract convergence information from a result.
extract_confidence_intervals
    Compute parameter confidence intervals.
validate_fit_inputs
    Validate fit inputs before execution.
create_fit_config_from_state
    Create a FitConfig from SessionState.
"""

import contextlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from nlsq.core.minpack import fit
from nlsq.result import CurveFitResult

# =============================================================================
# Configuration and Protocol Definitions
# =============================================================================


@dataclass(slots=True)
class FitConfig:
    """Configuration for curve fitting execution.

    Attributes
    ----------
    p0 : list[float] | None
        Initial parameter guesses.
    bounds : tuple[list, list] | None
        Parameter bounds as (lower, upper).
    gtol : float
        Gradient tolerance for convergence.
    ftol : float
        Function tolerance for convergence.
    xtol : float
        Parameter tolerance for convergence.
    max_iterations : int
        Maximum number of iterations.
    method : str
        Optimization method ('trf', 'lm', 'dogbox').
    loss : str
        Loss function type.
    workflow : str | None
        Workflow preset name ('fast', 'robust', 'quality', etc.).
    goal : str | None
        Optimization goal ('fast', 'robust', 'quality').
    enable_multistart : bool
        Whether to enable multi-start optimization.
    n_starts : int
        Number of starting points for multi-start.
    sampler : str
        Sampling method for multi-start.
    chunk_size : int
        Chunk size for streaming optimization.
    absolute_sigma : bool
        Whether sigma represents absolute uncertainties.
    """

    p0: list[float] | None = None
    bounds: tuple[list[float], list[float]] | None = None
    gtol: float = 1e-8
    ftol: float = 1e-8
    xtol: float = 1e-8
    max_iterations: int = 200
    max_function_evals: int = 2000
    method: str = "trf"
    loss: str = "linear"
    workflow: str | None = None
    goal: str | None = None
    enable_multistart: bool = False
    n_starts: int = 10
    sampler: str = "lhs"
    center_on_p0: bool = True
    scale_factor: float = 1.0
    chunk_size: int = 10000
    absolute_sigma: bool = False


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress callback during fitting.

    Implementations should track iteration progress and optionally
    signal early termination via should_abort().

    Methods
    -------
    on_iteration(iteration, cost, params)
        Called after each iteration with current state.
    should_abort() -> bool
        Return True to signal early termination.
    """

    def on_iteration(self, iteration: int, cost: float, params: np.ndarray) -> None:
        """Handle iteration update.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        cost : float
            Current cost (objective) value.
        params : np.ndarray
            Current parameter values.
        """
        ...

    def should_abort(self) -> bool:
        """Check if fitting should be aborted.

        Returns
        -------
        bool
            True if fitting should stop early.
        """
        ...


# =============================================================================
# Input Validation
# =============================================================================


def validate_fit_inputs(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
) -> None:
    """Validate fit inputs before execution.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    sigma : np.ndarray | None
        Optional uncertainties.

    Raises
    ------
    ValueError
        If inputs are invalid (empty, NaN, mismatched lengths).
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Check for empty data
    if len(xdata) == 0 or len(ydata) == 0:
        raise ValueError("Data arrays must not be empty")

    # Check for length mismatch
    if len(xdata) != len(ydata):
        raise ValueError(
            f"xdata and ydata must have the same length: "
            f"got {len(xdata)} and {len(ydata)}"
        )

    # Check for NaN/Inf in xdata
    if np.any(~np.isfinite(xdata)):
        raise ValueError("xdata contains NaN or Inf values")

    # Check for NaN/Inf in ydata
    if np.any(~np.isfinite(ydata)):
        raise ValueError("ydata contains NaN or Inf values")

    # Check sigma if provided
    if sigma is not None:
        sigma = np.asarray(sigma)
        if len(sigma) != len(ydata):
            raise ValueError(
                f"sigma must have the same length as ydata: "
                f"got {len(sigma)} and {len(ydata)}"
            )
        if np.any(~np.isfinite(sigma)):
            raise ValueError("sigma contains NaN or Inf values")
        if np.any(sigma <= 0):
            raise ValueError("sigma must contain only positive values")


# =============================================================================
# Fit Execution
# =============================================================================


class _CallbackWrapper:
    """Internal wrapper to adapt GUI callback to nlsq callback interface."""

    def __init__(self, callback: ProgressCallback | None):
        self.callback = callback
        self.iteration = 0
        self._aborted = False

    def __call__(self, xk: np.ndarray) -> bool:
        """Handle callback from optimizer.

        Parameters
        ----------
        xk : np.ndarray
            Current parameter values.

        Returns
        -------
        bool
            True to continue, False to abort (if supported).
        """
        if self.callback is None:
            return True

        self.iteration += 1

        # Compute cost (not available directly, pass 0 as placeholder)
        # The actual cost would need to be computed from the residuals
        # Don't let callback errors stop the fit
        with contextlib.suppress(Exception):
            self.callback.on_iteration(
                iteration=self.iteration,
                cost=0.0,  # Placeholder, actual cost not easily available
                params=np.asarray(xk),
            )

        # Check for abort signal
        try:
            if self.callback.should_abort():
                self._aborted = True
                return False
        except Exception:
            pass

        return True

    @property
    def aborted(self) -> bool:
        """Whether the fit was aborted."""
        return self._aborted


def execute_fit(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    model: Any,
    config: FitConfig,
    progress_callback: ProgressCallback | None,
) -> CurveFitResult:
    """Execute a curve fit with the given parameters.

    This is the main entry point for fitting from the GUI. It wraps
    nlsq.minpack.fit() with GUI-specific handling for callbacks,
    error handling, and result processing.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    sigma : np.ndarray | None
        Optional uncertainties in ydata.
    model : callable
        Model function ``f(x, *params) -> y``.
    config : FitConfig
        Fitting configuration parameters.
    progress_callback : ProgressCallback | None
        Optional callback for progress updates.

    Returns
    -------
    CurveFitResult
        Fitting result with popt, pcov, and diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If fitting fails catastrophically.
    """
    # Validate inputs
    validate_fit_inputs(xdata, ydata, sigma)

    # Convert to numpy arrays
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    if sigma is not None:
        sigma = np.asarray(sigma)

    # Prepare initial guess
    p0 = config.p0
    if p0 is not None:
        p0 = np.asarray(p0)

    # Prepare bounds - convert None values to ±inf
    bounds = config.bounds
    if bounds is not None:
        lower = [-float("inf") if v is None else float(v) for v in bounds[0]]
        upper = [float("inf") if v is None else float(v) for v in bounds[1]]
        bounds = (np.asarray(lower), np.asarray(upper))

    # Prepare callback wrapper
    callback_wrapper = _CallbackWrapper(progress_callback)

    # Build kwargs for fit()
    fit_kwargs: dict[str, Any] = {
        "gtol": config.gtol,
        "ftol": config.ftol,
        "xtol": config.xtol,
        "max_nfev": config.max_function_evals,
        "loss": config.loss,
        "absolute_sigma": config.absolute_sigma,
    }

    # Determine workflow
    workflow = config.workflow
    if workflow is None:
        workflow = "auto"

    # Determine goal
    goal = config.goal

    # Handle multi-start configuration
    if config.enable_multistart and not workflow.startswith("hpc"):
        # Use quality or large_robust workflow for multi-start
        if len(xdata) > 1_000_000:
            workflow = "large_robust"
        else:
            workflow = "quality"

    # Add method if not auto-selecting
    if config.method and config.method != "auto":
        fit_kwargs["method"] = config.method

    # Execute fit
    try:
        result = fit(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            bounds=bounds if bounds else (-float("inf"), float("inf")),
            workflow=workflow,
            goal=goal,
            **fit_kwargs,
        )
    except Exception as e:
        # Wrap exception with more context
        raise RuntimeError(f"Fitting failed: {e}") from e

    # Check for abort
    if callback_wrapper.aborted:
        # Mark result as aborted
        if hasattr(result, "__setitem__"):
            result["aborted"] = True

    return result


# =============================================================================
# Result Extraction Functions
# =============================================================================


def extract_fit_statistics(result: CurveFitResult) -> dict[str, float]:
    """Extract fit quality statistics from a result.

    Parameters
    ----------
    result : CurveFitResult
        The fitting result.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - r_squared: Coefficient of determination
        - adj_r_squared: Adjusted R-squared
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion
    """
    stats: dict[str, float] = {}

    try:
        stats["r_squared"] = float(result.r_squared)
    except Exception:
        stats["r_squared"] = float("nan")

    try:
        stats["adj_r_squared"] = float(result.adj_r_squared)
    except Exception:
        stats["adj_r_squared"] = float("nan")

    try:
        stats["rmse"] = float(result.rmse)
    except Exception:
        stats["rmse"] = float("nan")

    try:
        stats["mae"] = float(result.mae)
    except Exception:
        stats["mae"] = float("nan")

    try:
        stats["aic"] = float(result.aic)
    except Exception:
        stats["aic"] = float("nan")

    try:
        stats["bic"] = float(result.bic)
    except Exception:
        stats["bic"] = float("nan")

    return stats


def extract_convergence_info(result: CurveFitResult) -> dict[str, Any]:
    """Extract convergence information from a result.

    Parameters
    ----------
    result : CurveFitResult
        The fitting result.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - success: Whether optimization converged
        - message: Convergence message
        - nfev: Number of function evaluations
        - cost: Final cost value
        - optimality: Optimality measure
    """
    info: dict[str, Any] = {}

    info["success"] = getattr(result, "success", False)
    info["message"] = getattr(result, "message", "Unknown")
    info["nfev"] = getattr(result, "nfev", 0)
    info["cost"] = getattr(result, "cost", float("nan"))
    info["optimality"] = getattr(result, "optimality", float("nan"))

    # Convert numpy types to Python types
    if hasattr(info["success"], "item"):
        info["success"] = info["success"].item()
    if hasattr(info["nfev"], "item"):
        info["nfev"] = info["nfev"].item()
    if hasattr(info["cost"], "item"):
        info["cost"] = info["cost"].item()
    if hasattr(info["optimality"], "item"):
        info["optimality"] = info["optimality"].item()

    return info


def extract_confidence_intervals(
    result: CurveFitResult,
    alpha: float = 0.95,
) -> list[tuple[float, float]]:
    """Compute parameter confidence intervals.

    Parameters
    ----------
    result : CurveFitResult
        The fitting result.
    alpha : float
        Confidence level (default: 0.95 for 95% CI).

    Returns
    -------
    list[tuple[float, float]]
        List of (lower, upper) bounds for each parameter.
    """
    try:
        intervals = result.confidence_intervals(alpha=alpha)
        return [(float(low), float(high)) for low, high in intervals]
    except Exception:
        # Return infinite intervals if calculation fails
        n_params = len(result.popt) if hasattr(result, "popt") else 0
        return [(-float("inf"), float("inf"))] * n_params


def extract_parameter_uncertainties(result: CurveFitResult) -> list[float]:
    """Extract parameter uncertainties from covariance matrix.

    Parameters
    ----------
    result : CurveFitResult
        The fitting result.

    Returns
    -------
    list[float]
        Standard errors for each parameter.
    """
    try:
        pcov = result.pcov
        if pcov is None:
            return [float("nan")] * len(result.popt)
        return [float(np.sqrt(pcov[i, i])) for i in range(len(result.popt))]
    except Exception:
        n_params = len(result.popt) if hasattr(result, "popt") else 0
        return [float("nan")] * n_params


# =============================================================================
# Configuration Creation
# =============================================================================


def create_fit_config_from_state(state: Any) -> FitConfig:
    """Create a FitConfig from SessionState.

    Parameters
    ----------
    state : SessionState
        The GUI session state.

    Returns
    -------
    FitConfig
        Configuration for fitting.
    """
    # Determine workflow from state
    workflow = None
    if hasattr(state, "preset") and state.preset:
        preset_map = {
            "fast": "fast",
            "robust": "standard",
            "quality": "quality",
        }
        workflow = preset_map.get(state.preset.lower())

    return FitConfig(
        p0=state.p0,
        bounds=state.bounds,
        gtol=state.gtol,
        ftol=state.ftol,
        xtol=state.xtol,
        max_iterations=state.max_iterations,
        max_function_evals=getattr(state, "max_function_evals", 2000),
        method=state.method,
        loss=state.loss,
        workflow=workflow,
        goal=None,
        enable_multistart=state.enable_multistart,
        n_starts=state.n_starts,
        sampler=state.sampler,
        center_on_p0=getattr(state, "center_on_p0", True),
        scale_factor=getattr(state, "scale_factor", 1.0),
        chunk_size=state.chunk_size,
        absolute_sigma=False,
    )


# =============================================================================
# Streaming Support
# =============================================================================


def is_large_dataset(xdata: np.ndarray, threshold: int = 1_000_000) -> bool:
    """Check if dataset is large enough for streaming optimization.

    Parameters
    ----------
    xdata : np.ndarray
        Data array.
    threshold : int
        Size threshold for considering dataset large.

    Returns
    -------
    bool
        True if dataset is large.
    """
    return len(xdata) >= threshold


def get_recommended_chunk_size(data_size: int, target_chunks: int = 10) -> int:
    """Get recommended chunk size for streaming optimization.

    Parameters
    ----------
    data_size : int
        Total number of data points.
    target_chunks : int
        Target number of chunks.

    Returns
    -------
    int
        Recommended chunk size.
    """
    chunk_size = max(data_size // target_chunks, 1000)
    # Round to nice numbers
    if chunk_size > 100000:
        chunk_size = (chunk_size // 10000) * 10000
    elif chunk_size > 10000:
        chunk_size = (chunk_size // 1000) * 1000
    return chunk_size
