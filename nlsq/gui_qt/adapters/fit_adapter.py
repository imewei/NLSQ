"""Fitting execution adapter for NLSQ Qt GUI.

This module provides a GUI-friendly wrapper around nlsq.minpack.fit()
for executing curve fitting operations from the Qt interface.

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
import logging
import re
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from nlsq.callbacks import StopOptimization
from nlsq.core.minpack import fit
from nlsq.gui_qt.session_state import VALID_WORKFLOWS, n_points
from nlsq.result import CurveFitResult, OptimizeWarning

logger = logging.getLogger(__name__)

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
        Workflow name: 'auto' (local), 'auto_global' (global), 'hpc'.
    goal : str | None
        Optimization goal ('fast', 'robust', 'quality').
    n_starts : int
        Number of starting points for global optimization (auto_global).
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
    n_starts: int = 10
    sampler: str = "lhs"
    center_on_p0: bool = True
    scale_factor: float = 1.0
    chunk_size: int = 10000
    absolute_sigma: bool = False
    check_finite: bool = True
    auto_bounds: bool = False
    # Hybrid-streaming-only controls: carried for reporting, not used by fit().
    normalize: bool = True
    warmup_iterations: int = 200
    defense_preset: str | None = None
    layer1_enabled: bool = True
    layer2_enabled: bool = True
    layer3_enabled: bool = True
    layer4_enabled: bool = True
    enable_checkpoints: bool = False
    checkpoint_dir: str | None = None


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
    check_finite: bool = True,
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
    check_finite : bool, default True
        When False, NaN/Inf in ``xdata``/``ydata`` are left to ``curve_fit``
        (``nan_policy``); sigma and shape checks still apply.

    Raises
    ------
    ValueError
        If inputs are invalid (empty, NaN, mismatched lengths).
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Check for empty data
    n_x = n_points(xdata)
    if n_x == 0 or len(ydata) == 0:
        raise ValueError("Data arrays must not be empty")

    # Check for length mismatch (2-D xdata is (n_coords, n_points))
    if n_x != len(ydata):
        raise ValueError(
            f"xdata and ydata must have the same length: got {n_x} and {len(ydata)}",
        )

    # Check for NaN/Inf in xdata
    if check_finite and np.any(~np.isfinite(xdata)):
        raise ValueError("xdata contains NaN or Inf values")

    # Check for NaN/Inf in ydata
    if check_finite and np.any(~np.isfinite(ydata)):
        raise ValueError("ydata contains NaN or Inf values")

    # Check sigma if provided
    if sigma is not None:
        sigma = np.asarray(sigma)
        if len(sigma) != len(ydata):
            raise ValueError(
                f"sigma must have the same length as ydata: "
                f"got {len(sigma)} and {len(ydata)}",
            )
        if np.any(~np.isfinite(sigma)):
            raise ValueError("sigma contains NaN or Inf values")
        if np.any(sigma <= 0):
            raise ValueError("sigma must contain only positive values")


# =============================================================================
# Fit Execution
# =============================================================================


class _CallbackWrapper:
    """Internal wrapper to adapt GUI callback to nlsq callback interface.

    The nlsq callback signature is: callback(iteration, cost, params, info)
    """

    def __init__(self, callback: ProgressCallback | None):
        self.callback = callback
        self._aborted = False

    def __deepcopy__(self, memo: dict) -> "_CallbackWrapper":
        # multi_start deep-copies the callback per start; the abort flag and the
        # wrapped ProgressCallback must stay shared with the caller's instance.
        return self

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict,
    ) -> None:
        """Handle callback from optimizer.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        cost : float
            Current cost value.
        params : np.ndarray
            Current parameter values.
        info : dict
            Additional info from optimizer.

        Returns
        -------
        None
            Always None; abort is signalled by raising ``StopOptimization``.
        """
        if self.callback is None:
            return None

        # Don't let callback errors stop the fit
        with contextlib.suppress(Exception):
            self.callback.on_iteration(
                iteration=iteration,
                cost=cost,
                params=np.asarray(params),
            )

        # Check for abort signal
        try:
            if self.callback.should_abort():
                self._aborted = True
        except Exception:
            pass

        # core/trf.py ignores callback return values; only this exception stops it
        if self._aborted:
            raise StopOptimization("Fit aborted by user")

        return None

    @property
    def aborted(self) -> bool:
        """Whether the fit was aborted."""
        return self._aborted


class _ThreadLogCollector(logging.Handler):
    """Collect WARNING+ records the given thread emits on the ``nlsq`` loggers."""

    def __init__(self, thread_id: int, messages: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._thread_id = thread_id
        self._messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        # the adapter's own "setting not honoured" log lines are already notes
        if record.thread == self._thread_id and not record.name.startswith(
            "nlsq.gui_qt"
        ):
            self._messages.append(record.getMessage())


# thread id -> message sink of the fit currently collecting on that thread
_collectors: dict[int, list[str]] = {}
_hook_lock = threading.Lock()
_hook_filters: list[tuple] = []


def _warning_hook(message, category, filename, lineno, file=None, line=None):
    sink = _collectors.get(threading.get_ident())
    if sink is not None and issubclass(category, (RuntimeWarning, UserWarning)):
        # DeprecationWarning etc. concern the developer, not the fit
        sink.append(str(message))
    else:
        _hook_state["previous"](message, category, filename, lineno, file, line)


_hook_state = {"previous": warnings.showwarning}


def install_warning_hook() -> None:
    """Route a collecting thread's warnings to its fit; idempotent, never undone.

    One process-wide ``showwarning`` hook plus two permanent "always" filters
    (``OptimizeWarning`` and anything attributed to ``nlsq.*``, so a repeated
    fit warns again). Nothing here is restored later, so it cannot interleave
    badly with the ``catch_warnings`` blocks other threads (pyqtgraph, the
    results page) enter and leave while a fit runs. If something else swapped
    the hook out (a ``catch_warnings`` block that was entered before this ran),
    the next call reinstalls it.
    """
    with _hook_lock:
        if warnings.showwarning is not _warning_hook:
            _hook_state["previous"] = warnings.showwarning
            warnings.showwarning = _warning_hook
        # they must sit in front: a later ignore filter would shadow them
        if warnings.filters[: len(_hook_filters)] != _hook_filters or not _hook_filters:
            warnings.filterwarnings("always", category=OptimizeWarning)
            warnings.filterwarnings("always", module=r"nlsq\.")
            _hook_filters[:] = warnings.filters[:2]  # in list order, for the guard


@contextlib.contextmanager
def capture_fit_diagnostics() -> Iterator[list[str]]:
    """Collect the library warnings and log records the calling thread raises.

    The library reports degenerate fits (covariance not estimable, rank-deficient
    Jacobian, ignored settings, non-full-batch routes) only through
    ``warnings.warn`` and ``nlsq.*`` WARNING log records, which a GUI never
    sees. Output is collected per thread (see ``install_warning_hook``); other
    threads' warnings go to the previous ``showwarning`` untouched.

    Limits: only the thread that runs the fit is collected, so warnings and log
    records raised on library-owned pool threads (multi-start runs its starts in
    a ``ThreadPoolExecutor``) are not; and a warning raised outside ``nlsq.*``
    (e.g. a numpy RuntimeWarning from a user model) follows Python's default
    once-per-location rule, so a repeated fit may not report it again.
    A fit-thread warning raised while the GUI thread sits inside a bare
    ``catch_warnings()``/``simplefilter("ignore")`` block (pyqtgraph's
    ``dataBounds``) can still be dropped; diagnostics that are also logged
    survive via the log handler. ``catch_warnings`` blocks inside the library
    on the fit thread (``streaming/specs.py``, ``stability/fallback.py``) can
    leave a stale ignore filter; that is outside ``gui_qt`` and not fixed here.
    """
    install_warning_hook()
    messages: list[str] = []
    thread_id = threading.get_ident()
    nlsq_logger = logging.getLogger("nlsq")
    handler = _ThreadLogCollector(thread_id, messages)
    _collectors[thread_id] = messages
    nlsq_logger.addHandler(handler)
    try:
        yield messages
    finally:
        nlsq_logger.removeHandler(handler)
        _collectors.pop(thread_id, None)


_MAX_FIT_NOTES = 5
_NOTE_SUFFIX = re.compile(r"\s+reason=\S+\s*$")


_FILLER = frozenset({"the", "of", "a", "parameters"})


def _dedupe_notes(messages: list[str]) -> list[str]:
    """Unique messages in order, capped so the status line stays readable.

    A diagnostic often arrives twice, as a log record and as a warning with
    slightly different wording. Texts that differ only by a ``reason=...``
    suffix, a ``Category:`` prefix or filler words, or whose words are all in
    another note, count as one (the more detailed text is kept).
    """
    kept: list[tuple[frozenset[str], str]] = []
    for m in messages:
        text = _NOTE_SUFFIX.sub("", re.sub(r"^\w*Warning:\s*", "", m.strip())).strip()
        words = frozenset(re.findall(r"\w+", text.lower())) - _FILLER
        if not text or any(words <= w for w, _ in kept):
            continue
        kept = [(w, t) for w, t in kept if not w <= words]
        kept.append((words, text))
    notes = [t for _, t in kept]
    if len(notes) > _MAX_FIT_NOTES:
        extra = len(notes) - _MAX_FIT_NOTES
        notes = [*notes[:_MAX_FIT_NOTES], f"(+{extra} more library warnings)"]
    return notes


def _resolve_workflow(
    workflow: str | None,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    notes: list[str],
    can_multistart: bool = False,
) -> str:
    """Pick the workflow; auto_global strictly requires finite bounds.

    Without finite bounds on every parameter it degrades to the 'auto'
    workflow and says so in ``notes`` instead of doing it silently. With
    ``can_multistart`` (a p0 is available) the caller still runs multi-start,
    sampling inside finite bounds inferred from p0 and the data.
    """
    workflow = workflow or "auto"
    if workflow not in VALID_WORKFLOWS:
        raise ValueError(
            f"Unknown workflow {workflow!r}; expected one of {sorted(VALID_WORKFLOWS)}"
        )
    if workflow == "auto_global" and not (
        bounds is not None
        and np.all(np.isfinite(bounds[0]))
        and np.all(np.isfinite(bounds[1]))
    ):
        if can_multistart:
            notes.append(
                "global search needs finite bounds on every parameter (lower "
                "and upper); multi-start sampled starting points inside bounds "
                "inferred from p0 and the data instead (your bounds still "
                "constrain every fit)"
            )
        else:
            notes.append(
                "global multi-start search needs finite bounds on every "
                "parameter (lower and upper); ran a local fit instead (set all "
                "bounds to enable it)"
            )
        return "auto"
    return workflow


def _note_ignored_streaming_controls(config: FitConfig, notes: list[str]) -> None:
    """Append a note for non-default streaming/defense settings fit() ignores."""
    defaults = FitConfig()
    ignored = []
    if config.normalize != defaults.normalize:
        ignored.append("normalize")
    if config.warmup_iterations != defaults.warmup_iterations:
        ignored.append("warmup iterations")
    if config.defense_preset not in (None, "default"):
        ignored.append("defense preset")
    ignored += [
        f"defense layer {i}"
        for i in range(1, 5)
        if not getattr(config, f"layer{i}_enabled")
    ]
    if ignored:
        notes.append(
            f"{', '.join(ignored)} apply only to hybrid-streaming runs and "
            "are not used by this fit"
        )


def _global_search_kwargs(
    config: FitConfig, workflow: str, notes: list[str]
) -> dict[str, Any]:
    """Multi-start and checkpoint kwargs for the workflows that support them."""
    # hpc delegates to auto_global, so it takes the same multi-start controls
    if workflow in ("auto_global", "hpc") and config.n_starts > 0:
        kwargs: dict[str, Any] = {
            "n_starts": config.n_starts,
            "sampler": config.sampler,
            "center_on_p0": config.center_on_p0,
            "scale_factor": config.scale_factor,
        }
        # checkpointing is implemented by the library for these workflows only
        if config.enable_checkpoints and config.checkpoint_dir:
            kwargs["checkpoint_dir"] = config.checkpoint_dir
        return kwargs
    if config.enable_checkpoints:
        notes.append(
            "checkpointing applies only to the auto_global and hpc workflows "
            "and is not used by this fit"
        )
    return {}


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
    validate_fit_inputs(xdata, ydata, sigma, config.check_finite)

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
        if bounds[0].size == 1 and bounds[1].size == 1:
            # a single (scalar) bound applies to every parameter: fit() only
            # broadcasts 0-d bounds, so (1,) would fail for n_params > 1
            bounds = (bounds[0].reshape(()), bounds[1].reshape(()))

    # Prepare callback wrapper
    callback_wrapper = (
        _CallbackWrapper(progress_callback) if progress_callback else None
    )

    # Build kwargs for fit()
    fit_kwargs: dict[str, Any] = {
        "gtol": config.gtol,
        "ftol": config.ftol,
        "xtol": config.xtol,
        # nfev counts the initial residual evaluation, and every iteration costs
        # >= 1 more, so max_iterations + 1 bounds the iteration count (the
        # solver has no separate iteration limit).
        "max_nfev": min(config.max_function_evals, config.max_iterations + 1),
        "loss": config.loss,
        "absolute_sigma": config.absolute_sigma,
        "check_finite": config.check_finite,
    }

    # Add callback if provided
    if callback_wrapper is not None:
        fit_kwargs["callback"] = callback_wrapper

    notes: list[str] = []
    can_multistart = p0 is not None and config.n_starts > 0
    workflow = _resolve_workflow(config.workflow, bounds, notes, can_multistart)
    if can_multistart and config.workflow == "auto_global" and workflow == "auto":
        fit_kwargs.update(
            multistart=True,
            n_starts=config.n_starts,
            sampler=config.sampler,
            center_on_p0=config.center_on_p0,
            scale_factor=config.scale_factor,
        )

    # Determine goal
    goal = config.goal

    fit_kwargs.update(_global_search_kwargs(config, workflow, notes))

    # fit()'s workflow API has no chunking control (memory-based selection).
    if config.chunk_size != FitConfig().chunk_size:
        notes.append("chunk size is not configurable here and was ignored")

    # Streaming/defense controls only configure hybrid-streaming runs.
    _note_ignored_streaming_controls(config, notes)

    # Add method if not auto-selecting
    if config.method and config.method != "auto":
        fit_kwargs["method"] = config.method

    if config.auto_bounds:
        fit_kwargs["auto_bounds"] = True

    # Execute fit
    try:
        with capture_fit_diagnostics() as diagnostics:
            result = fit(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                sigma=sigma,
                bounds=bounds or (-float("inf"), float("inf")),
                workflow=workflow,
                goal=goal,
                **fit_kwargs,
            )
    except Exception as e:
        # Wrap exception with more context
        raise RuntimeError(f"Fitting failed: {e}") from e

    for note in notes:
        logger.warning("GUI fit setting not honoured: %s", note)
    # Library warnings (degenerate fit, ignored settings, non-full-batch route)
    # would otherwise never leave stderr/the log: show them with the result.
    notes = _dedupe_notes([*notes, *diagnostics])
    if notes and hasattr(result, "__setitem__"):
        # Shown on the Results page status line
        result["gui_notes"] = notes

    # Check for abort
    if callback_wrapper is not None and callback_wrapper.aborted:
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


def parameter_confidence_intervals(result: Any, n_data: int) -> np.ndarray | None:
    """95% parameter intervals as an ``(n_params, 2)`` array, or None without pcov.

    The result owns the statistics (known absolute sigma gives a normal
    quantile, otherwise Student-t with ``n - p`` degrees of freedom) so the
    table, the CSV/JSON exports and the library never disagree. NaN means
    undefined (``n <= p`` with an estimated noise scale), never a made-up
    interval. Results that cannot compute their own (plain objects, or a
    result with no data attached) fall back to the t interval with ``n_data``.
    """
    pcov = getattr(result, "pcov", None)
    if pcov is None:
        return None
    method = getattr(result, "confidence_intervals", None)
    if callable(method):
        with contextlib.suppress(Exception), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # n <= p is flagged by the caller
            return np.asarray(method(alpha=0.95), dtype=float)
    from scipy import stats

    popt = getattr(result, "x", None)
    if popt is None:
        popt = getattr(result, "popt", None)
    if popt is None:
        return None
    popt = np.asarray(popt, dtype=float)
    dof = n_data - len(popt)
    half = (
        stats.t.ppf(0.975, dof) * np.sqrt(np.diag(np.asarray(pcov, dtype=float)))
        if dof > 0
        else np.full(len(popt), np.nan)
    )
    return np.column_stack([popt - half, popt + half])


def confidence_band(result: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """95% confidence band of the fitted curve at ``x`` from the result.

    Delegates to ``CurveFitResult.confidence_band`` (domain-aware derivative,
    non-finite covariance handled). None when the result cannot provide one.
    """
    method = getattr(result, "confidence_band", None)
    if not callable(method) or getattr(result, "pcov", None) is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # n <= p is flagged by the caller
            lower, upper = method(x)
    except Exception:
        logger.warning("Confidence band unavailable", exc_info=True)
        return None
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


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
    # Determine workflow from state using 3-workflow system (v0.6.3)
    # Get workflow directly from state, or default to 'auto'
    workflow = getattr(state, "workflow", None)
    if workflow is None:
        workflow = "auto"

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
        n_starts=state.n_starts,
        sampler=state.sampler,
        center_on_p0=getattr(state, "center_on_p0", True),
        scale_factor=getattr(state, "scale_factor", 1.0),
        chunk_size=state.chunk_size,
        absolute_sigma=bool(getattr(state, "absolute_sigma", False)),
        check_finite=bool(getattr(state, "check_finite", True)),
        auto_bounds=bool(getattr(state, "auto_bounds", False)),
        normalize=getattr(state, "normalize", True),
        warmup_iterations=getattr(state, "warmup_iterations", 200),
        defense_preset=getattr(state, "defense_preset", None),
        layer1_enabled=getattr(state, "layer1_enabled", True),
        layer2_enabled=getattr(state, "layer2_enabled", True),
        layer3_enabled=getattr(state, "layer3_enabled", True),
        layer4_enabled=getattr(state, "layer4_enabled", True),
        enable_checkpoints=bool(getattr(state, "enable_checkpoints", False)),
        checkpoint_dir=getattr(state, "checkpoint_dir", None),
    )


# =============================================================================
# Convenience Wrapper
# =============================================================================


class _SimpleProgressCallback:
    """Simple progress callback adapter for run_fit."""

    def __init__(self, callback: Any | None):
        self._callback = callback
        self._abort = False

    def on_iteration(self, iteration: int, cost: float, params: np.ndarray) -> None:
        """Handle iteration update."""
        if self._callback is not None:
            try:
                result = self._callback(iteration, cost)
                if result is False:
                    self._abort = True
            except Exception:
                pass

    def should_abort(self) -> bool:
        """Check if fitting should be aborted."""
        return self._abort


def run_fit(
    state: Any,
    progress_callback: Any | None = None,
) -> CurveFitResult:
    """Run a curve fit from GUI session state.

    This is a convenience wrapper around execute_fit that extracts
    parameters from the session state object.

    Parameters
    ----------
    state : SessionState
        The GUI session state containing data, model, and configuration.
    progress_callback : callable, optional
        Callback function with signature (iteration: int, cost: float) -> bool.
        Return False to abort the fit.

    Returns
    -------
    CurveFitResult
        Fitting result with popt, pcov, and diagnostics.

    Raises
    ------
    ValueError
        If required data or model is missing from state.
    RuntimeError
        If fitting fails.
    """
    # Validate state has required data
    if state.xdata is None or state.ydata is None:
        raise ValueError("Data not loaded. Please load data before fitting.")

    if state.model_func is None:
        raise ValueError("Model not selected. Please select a model before fitting.")

    if not callable(state.model_func):
        raise TypeError(
            f"model_func must be callable, got {type(state.model_func).__name__}",
        )

    # Extract data
    xdata = np.asarray(state.xdata)
    ydata = np.asarray(state.ydata)
    sigma = np.asarray(state.sigma) if state.sigma is not None else None

    # Create config from state
    config = create_fit_config_from_state(state)

    # Wrap the simple callback into ProgressCallback protocol
    wrapped_callback = (
        _SimpleProgressCallback(progress_callback) if progress_callback else None
    )

    # Execute fit
    return execute_fit(
        xdata=xdata,
        ydata=ydata,
        sigma=sigma,
        model=state.model_func,
        config=config,
        progress_callback=wrapped_callback,
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
    return n_points(xdata) >= threshold


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
