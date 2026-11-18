"""
Automatic Fallback Strategies for Robust Optimization
======================================================

This module provides automatic fallback mechanisms that attempt to recover
from optimization failures by trying alternative approaches.

Key Features:
- Alternative method selection (currently only "trf"; a no-op when TRF is
  already the method, so it never repeats the original fit)
- Initial guess perturbation for escaping local minima
- Tolerance adjustment for difficult problems
- Parameter bound inference when needed
- Robust loss function application
- Problem rescaling for numerical stability

Example:
    >>> from nlsq.stability.fallback import FallbackOrchestrator
    >>>
    >>> orchestrator = FallbackOrchestrator(verbose=True)
    >>> result = orchestrator.fit_with_fallback(
    ...     model, xdata, ydata, p0=[1, 0.5],
    ...     max_attempts=5
    ... )
    >>> print(f"Success with strategy: {result.fallback_strategy_used}")
"""

import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import Any, ClassVar, Protocol

import numpy as np

from nlsq.precision.bound_inference import infer_bounds
from nlsq.result import OptimizeWarning
from nlsq.utils.logging import get_logger

__all__ = [
    "FallbackOrchestrator",
    "FallbackResult",
    "FallbackStrategy",
]


class StrategyFactory(Protocol):
    """Protocol for strategy classes that can be instantiated with no arguments."""

    def __call__(self) -> "FallbackStrategy": ...


class FallbackStrategy:
    """Base class for fallback strategies."""

    def __init__(self, name: str, description: str, priority: int = 0):
        """
        Initialize fallback strategy.

        Parameters
        ----------
        name : str
            Strategy name
        description : str
            Human-readable description
        priority : int, optional
            Execution priority (higher = earlier). Default: 0
        """
        self.name = name
        self.description = description
        self.priority = priority
        self.attempts = 0
        self.successes = 0

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Apply strategy by modifying fit parameters.

        Parameters
        ----------
        kwargs : dict
            Original curve_fit keyword arguments

        Returns
        -------
        modified_kwargs : dict
            Modified keyword arguments
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def has_more(self) -> bool:
        """Whether another apply() would yield a new variant (progressive strategies).

        Queried by the orchestrator after each apply(); single-shot strategies
        keep the default ``False`` and are tried once per fit.
        """
        return False

    def reset(self) -> None:
        """Reset per-fit progressive state before a new fit_with_fallback() run.

        Subclasses that advance through a sequence across successive apply()
        calls within one fit (e.g. trying loss functions or perturbations in
        order) must override this to reset that sequence position -- without
        it, a second fit on a reused orchestrator instance silently resumes
        wherever the previous fit's sequence left off (or has already
        exhausted it, making the strategy a permanent no-op). `attempts` and
        `successes` are intentionally NOT reset here: they are lifetime
        statistics across every fit this strategy instance has taken part in.
        """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name='{self.name}', priority={self.priority})"
        )


class AlternativeMethodStrategy(FallbackStrategy):
    """Try alternative optimization methods.

    Only ``"trf"`` is in the sequence, so this is a no-op (the kwargs come back
    unchanged and the orchestrator skips the attempt) whenever the fit already
    uses TRF -- the default when ``method`` is unset.
    """

    def __init__(self):
        super().__init__(
            name="alternative_method",
            description="Try alternative optimization method",
            priority=10,  # High priority - cheap to try
        )
        # Placeholder: only "trf" is currently supported, so this strategy is a no-op
        # when TRF is already in use. Extend method_sequence to add alternatives.
        self.method_sequence = ["trf"]
        self.current_index = 0

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Try next method in sequence."""
        modified = kwargs.copy()
        if self.current_index < len(self.method_sequence):
            method = self.method_sequence[self.current_index]
            self.current_index += 1
            # Re-running the method the fit already used repeats the original solve.
            if method != (kwargs.get("method") or "trf"):
                modified["method"] = method
        return modified

    def has_more(self) -> bool:
        return self.current_index < len(self.method_sequence)

    def reset(self) -> None:
        """Restart the method sequence from the beginning for a new fit."""
        self.current_index = 0


class PerturbInitialGuessStrategy(FallbackStrategy):
    """Perturb initial guess to escape local minima."""

    def __init__(
        self,
        perturbation_scale: float = 0.1,
        max_perturbations: int = 3,
        seed: int = 0,
    ):
        super().__init__(
            name="perturb_p0",
            description=f"Perturb initial guess by {perturbation_scale * 100:.0f}%",
            priority=8,
        )
        self.perturbation_scale = perturbation_scale
        self.max_perturbations = max_perturbations
        self.perturbation_count = 0
        self.seed = seed

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Add random perturbation to p0."""
        modified = kwargs.copy()
        p0 = np.array(modified.get("p0", [1.0]))

        if self.perturbation_count < self.max_perturbations:
            # Add random noise scaled by parameter magnitude (additive for p0=0)
            # Deterministic per (seed, attempt): no global RNG, no stored generator.
            rng = np.random.default_rng((self.seed, self.perturbation_count))
            noise = rng.standard_normal(p0.shape) * self.perturbation_scale
            perturbed_p0 = p0 + noise * np.maximum(np.abs(p0), 1e-3)

            # Ensure p0 stays within bounds if provided
            bounds = modified.get("bounds", (-np.inf, np.inf))
            lower, upper = bounds
            lower = np.atleast_1d(lower)
            upper = np.atleast_1d(upper)

            perturbed_p0 = np.clip(perturbed_p0, lower, upper)

            modified["p0"] = perturbed_p0
            self.perturbation_count += 1

        return modified

    def has_more(self) -> bool:
        return self.perturbation_count < self.max_perturbations

    def reset(self) -> None:
        """Restart the perturbation budget from zero for a new fit."""
        self.perturbation_count = 0


class AdjustTolerancesStrategy(FallbackStrategy):
    """Relax optimization tolerances."""

    def __init__(self, relaxation_factor: float = 10.0, max_relaxations: int = 3):
        super().__init__(
            name="adjust_tolerances",
            description=f"Relax tolerances by {relaxation_factor}x",
            priority=7,
        )
        self.relaxation_factor = relaxation_factor
        self.max_relaxations = max_relaxations
        self.current_factor = 1.0
        self._relaxations = 0

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Relax ftol, xtol, gtol."""
        modified = kwargs.copy()

        # Default tolerances
        # Relax by factor
        self.current_factor *= self.relaxation_factor
        self._relaxations += 1
        for key in ("ftol", "xtol", "gtol"):
            tol = modified.get(key, 1e-8)
            if tol is not None:  # None disables that test: nothing to relax
                modified[key] = tol * self.current_factor

        return modified

    def has_more(self) -> bool:
        return self._relaxations < self.max_relaxations

    def reset(self) -> None:
        """Restart the relaxation factor from 1.0 (unrelaxed) for a new fit."""
        self.current_factor = 1.0
        self._relaxations = 0


class AddParameterBoundsStrategy(FallbackStrategy):
    """Infer and add parameter bounds if not provided.

    The inferred box is a heuristic (it assumes e.g. a positive rate in the
    second parameter, and ``p0``-centred windows), so it can exclude the
    unconstrained solution. ``enforce_positivity`` is off, but the box is still
    not guaranteed sign-neutral; the safeguard is that the orchestrator reports
    the attempt as unsuccessful when the fitted parameters sit on an inferred
    bound.
    """

    def __init__(self):
        super().__init__(
            name="add_bounds",
            description="Add inferred parameter bounds",
            priority=6,
        )

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Infer reasonable bounds from data and p0."""
        modified = kwargs.copy()

        # Check if bounds already provided. Bounds may be per-parameter arrays,
        # so compare scalar-safe instead of `!= (-np.inf, np.inf)` (that tuple
        # comparison raises ValueError for array bounds: ambiguous truth value).
        if "bounds" in modified:
            lower, upper = modified["bounds"]
            is_default = (
                np.isscalar(lower)
                and np.isscalar(upper)
                and lower == -np.inf
                and upper == np.inf
            )
            if not is_default:
                return modified  # Already has bounds

        p0 = np.array(modified.get("p0", [1.0]))
        xdata = modified.get("_xdata")  # Internal use
        ydata = modified.get("_ydata")  # Internal use

        if xdata is None or ydata is None:
            return modified  # Can't infer without data

        # enforce_positivity=False: p0 may be the auto-filled ones(n), which says
        # nothing about the sign of the true parameters.
        lower, upper = infer_bounds(xdata, ydata, p0, enforce_positivity=False)

        modified["bounds"] = (lower, upper)
        modified["_invented_bounds"] = True  # popped by _run_strategy

        return modified


class UseRobustLossStrategy(FallbackStrategy):
    """Apply robust loss function to handle outliers."""

    def __init__(self):
        super().__init__(
            name="robust_loss",
            description="Apply Huber loss for outliers",
            priority=5,
        )
        self.loss_sequence = ["soft_l1", "huber", "cauchy"]
        self.current_index = 0

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Try next robust loss function."""
        modified = kwargs.copy()

        if self.current_index < len(self.loss_sequence):
            modified["loss"] = self.loss_sequence[self.current_index]
            self.current_index += 1

        return modified

    def has_more(self) -> bool:
        return self.current_index < len(self.loss_sequence)

    def reset(self) -> None:
        """Restart the loss-function sequence from the beginning for a new fit."""
        self.current_index = 0


def _rescale_sigma(sigma: Any, y_scale: float) -> np.ndarray:
    """Rescale a `sigma` uncertainty under y' = (y - y_offset) / y_scale.

    1-D sigma is a std-dev, which scales linearly with y; a 2-D sigma is a
    covariance matrix, which scales quadratically.
    """
    sigma_arr = np.asarray(sigma)
    if sigma_arr.ndim == 2:
        return sigma_arr / y_scale / y_scale  # not y_scale**2: that can overflow
    return sigma_arr / y_scale


def _fit_rescaled(curve_fit, f, xdata, ydata, kwargs, scale_info):
    """Rescale attempt: callback-free pre-solve, then one flagged solve.

    Contract: each attempt runs one caller-visible, flagged solve; the rescale
    attempt additionally runs a callback-free pre-solve on the scaled problem to
    find a starting point, so a callback runs only in the solve whose result can
    be returned. The pre-solve uses the caller's bounds, loss, etc. but drops
    callbacks, output flags and side effects; only its ``popt`` (unit-invariant)
    is used, and only if the pre-solve succeeded. The final solve is on the original problem from that start, with the
    caller's kwargs, and its result is returned verbatim (original units by
    construction). A failed, unsuccessful or non-finite pre-solve (and any error or
    warning inside it) falls back to the caller's ``p0``. The pre-solve's warnings
    are silenced with ``warnings.catch_warnings``, which is process-global (not
    thread-safe): warnings emitted by other threads during the pre-solve window
    are suppressed too.
    """
    start = {}
    # Everything the pre-solve needs (scaling, sigma, jac, f_scale, the call) is
    # inside the guard: any failure means "no better start". Its warnings are
    # silenced (and cannot be promoted to errors) so only the final solve speaks.
    try:
        args, scaled_kwargs = _presolve_args(f, xdata, ydata, kwargs, scale_info)
        # warnings.catch_warnings is process-global (not thread-safe): warnings
        # emitted by other threads during the pre-solve window are suppressed
        # too. curve_fit has no per-call switch for them. Same trade-off as
        # multi_start._suppress_optimize_warning; only the call is inside.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = curve_fit(*args, **scaled_kwargs)
        popt = np.asarray(scaled.x, dtype=float)
        if _fit_succeeded(scaled) and np.all(np.isfinite(popt)):
            start = {"p0": popt}
        else:
            get_logger(__name__).debug("Rescale pre-solve unsuccessful; using p0")
    except Exception as exc:  # the pre-solve only improves the start
        get_logger(__name__).debug(f"Rescale pre-solve failed ({exc}); using p0")
    return curve_fit(f, xdata, ydata, **{**kwargs, **start})


def _presolve_args(f, xdata, ydata, kwargs, scale_info):
    """Positional args and kwargs of the scaled pre-solve (see ``_fit_rescaled``)."""
    xs, ys = scale_info["_x_scale"], scale_info["_y_scale"]
    xo, yo = scale_info["_x_offset"], scale_info["_y_offset"]

    def scaled_f(x_scaled, *params):
        return (f(x_scaled * xs + xo, *params) - yo) / ys

    scaled_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("full_output", "return_eval", "timeit", "callback")
    }
    scaled_kwargs.update(compute_diagnostics=False, compute_covariance=False, verbose=0)
    jac = kwargs.get("jac")
    if callable(jac):
        # d/dp of (f(x_s*xs + xo, p) - yo) / ys
        scaled_kwargs["jac"] = lambda x_scaled, *p: jac(x_scaled * xs + xo, *p) / ys
    sigma = kwargs.get("sigma")
    if sigma is not None:
        scaled_kwargs["sigma"] = _rescale_sigma(sigma, ys)
    elif kwargs.get("f_scale") is not None or kwargs.get("loss", "linear") != "linear":
        # Without sigma, residuals have y units. With sigma, scaling both
        # data and weights leaves whitened residuals (and f_scale) unchanged.
        # An omitted f_scale is the solver default 1.0 and must scale too.
        f_scale = kwargs.get("f_scale")
        scaled_kwargs["f_scale"] = (1.0 if f_scale is None else f_scale) / ys
    return (scaled_f, (xdata - xo) / xs, (ydata - yo) / ys), scaled_kwargs


class RescaleProblemStrategy(FallbackStrategy):
    """Rescale data for numerical stability."""

    def __init__(self):
        super().__init__(
            name="rescale_problem",
            description="Rescale parameters and data",
            priority=4,
        )

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize data to [0, 1] range."""
        modified = kwargs.copy()

        xdata = modified.get("_xdata")
        ydata = modified.get("_ydata")
        p0 = modified.get("p0")

        if xdata is None or ydata is None or p0 is None:
            return modified

        # Store scaling factors for later inverse transform
        x_scale = np.ptp(xdata)
        y_scale = np.ptp(ydata)
        x_offset = np.min(xdata)
        y_offset = np.min(ydata)

        if x_scale > 0 and y_scale > 0:
            modified["_x_scale"] = x_scale
            modified["_y_scale"] = y_scale
            modified["_x_offset"] = x_offset
            modified["_y_offset"] = y_offset
            modified["_scaled"] = True

        return modified


_OUTPUT_FLAGS = ("full_output", "return_eval", "timeit")


def _flags(kwargs: dict[str, Any]) -> dict[str, bool]:
    return {k: bool(kwargs.get(k)) for k in _OUTPUT_FLAGS}


def _tuple_kind(result, flags) -> str | None:
    """Layout of a flagged ``curve_fit`` return (same precedence as curve_fit).

    ``"eval"``: ``(popt, pcov, f(x))``; ``"full"``: ``(popt, pcov, infodict,
    mesg, ier)``; ``"timeit"``: ``(popt, pcov, res, post_time, ctime)``.
    ``None`` for a result object.
    """
    if not isinstance(result, tuple) or not flags:
        return None
    if flags.get("return_eval"):
        return "eval"
    if flags.get("full_output"):
        return "full"
    return "timeit" if flags.get("timeit") else None


def _fit_succeeded(result, flags=None) -> bool:
    """Whether a ``curve_fit`` return value reports convergence.

    Result objects use ``.success``. For flagged tuples: ``full_output`` carries
    ``ier`` (1-4 converged); ``timeit`` carries the raw result at index 2 and
    uses its ``success``. ``return_eval`` takes precedence over the other flags
    and returns ``(popt, pcov, f(x))`` with no status (also when combined with
    ``full_output``/``timeit``), so a finite ``popt`` counts as success (HEAD
    behaviour for that mode).
    """
    kind = _tuple_kind(result, flags)
    if kind is None:
        return bool(getattr(result, "success", True))
    if kind == "full":
        # A callback stop (status -2, success=True) is reported as ier=5; the
        # message is the only trace left, so keep it a success (no retries).
        from nlsq.core.least_squares import TERMINATION_MESSAGES

        return result[4] in (1, 2, 3, 4) or result[3] == TERMINATION_MESSAGES[-2]
    finite = bool(np.all(np.isfinite(np.asarray(result[0], dtype=float))))
    if kind == "timeit":
        return finite and bool(result[2].get("success", True))
    return finite


def _fitted_params(result, flags=None) -> np.ndarray | None:
    """popt of a curve_fit return value (object or flagged tuple), or None."""
    if _tuple_kind(result, flags) is not None:
        return np.asarray(result[0], dtype=float)
    popt = getattr(result, "popt", None)
    if popt is None:
        popt = getattr(result, "x", None)
    return None if popt is None else np.asarray(popt, dtype=float)


def _ulp(v, toward):
    """Spacing from ``|v|`` one step toward ``toward`` (+inf/-inf).

    Unlike ``np.spacing`` it is finite at finfo.max (falls back to the opposite
    direction when the step leaves the float range), and it is direction-aware:
    the step above an exact power of two is twice the step below. Callers hold
    ``np.errstate(under/over/invalid="ignore")`` (subnormal steps underflow).
    """
    step = np.abs(np.nextafter(v, toward) - v)
    return np.where(np.isfinite(step), step, np.abs(v - np.nextafter(v, -toward)))


def _hits_bound(result, bounds, flags=None, p0=None) -> bool:
    """Whether any fitted parameter sits on a finite bound.

    Allowance per parameter and bound ``b``: ``1e-6 * scale + 64 ulps``.
    ``scale`` is the interval width when finite; otherwise the parameter's own
    natural scale ``max(|b|, |p0_i|)`` (never another parameter's), or 1.0 when
    that is zero (an uninformative zero start at a zero bound). A finite
    interval caps the allowance at 1% of its width, but a distance of up to 2
    spacings of ``b`` always counts as a hit (a strictly feasible solver cannot
    land closer). The solver stays ~1e-14 of the scale from a bound, inside it.
    """
    popt = _fitted_params(result, flags)
    if popt is None:
        return False
    lower, upper = (
        np.broadcast_to(np.asarray(b, dtype=float), popt.shape) for b in bounds
    )
    p0_abs = (
        np.zeros(popt.shape)
        if p0 is None
        else np.broadcast_to(np.abs(np.asarray(p0, dtype=float)), popt.shape)
    )
    # Overflow is expected at the edge of the float range (|b| ~ finfo.max):
    # a width that overflows is simply "not a finite interval".
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        width = upper - lower  # inf when either side is unbounded
        finite_width = np.isfinite(width)
        for b, toward in ((lower, np.inf), (upper, -np.inf)):
            own = np.maximum(np.abs(b), p0_abs)
            scale = np.where(finite_width, width, np.where(own > 0, own, 1.0))
            tol = 1e-6 * scale + 64 * _ulp(np.maximum(np.abs(b), np.abs(popt)), np.inf)
            tol = np.where(finite_width, np.minimum(tol, 0.01 * width), tol)
            # Two representable steps into the interval always count (a strictly
            # feasible solver cannot land closer); stepping the bound itself,
            # not scaling one spacing, stays correct across a power of two.
            b2 = np.nextafter(np.nextafter(b, toward), toward)
            b2 = np.where(np.isfinite(b2), b2, b)
            on_edge = (popt >= np.minimum(b, b2)) & (popt <= np.maximum(b, b2))
            near = (np.abs(popt - b) <= tol) | on_edge
            if np.any(np.isfinite(b) & near):
                return True
    return False


def _mark_unsuccessful(
    result, flags=None, reason: str = "Fit pinned to an inferred bound"
):
    """Flag ``result`` as not converged where its format has a status field.

    A ``return_eval``-only tuple has none and is returned unchanged; the
    orchestrator then relies on the ``pinned`` verdict from ``_run_strategy``.
    """
    kind = _tuple_kind(result, flags)
    if kind == "full":
        return (*result[:3], reason, 5)
    if kind == "timeit":
        result[2]["success"] = False
        result[2]["message"] = reason
        return result
    if kind == "eval":
        return result
    result["success"] = False
    result["message"] = reason
    return result


def _same_kwargs(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether a strategy left the fit kwargs unchanged (a no-op attempt)."""
    if a.keys() != b.keys():
        return False
    for k, v in a.items():
        w = b[k]
        if v is w:
            continue
        try:
            if np.array_equal(np.asarray(v), np.asarray(w)):
                continue
        except Exception:
            pass
        return False
    return True


@lru_cache(maxsize=1)
def _loss_funcs():
    from nlsq.core.loss_functions import LossFunctionsJIT

    return LossFunctionsJIT()


def _robust_cost(fvec, loss, f_scale) -> float:
    """Cost of residuals ``fvec`` under a built-in or callable loss.

    Same formula as curve_fit: ``0.5 * f_scale**2 * sum(rho(z))``,
    ``z = (f / f_scale)**2``. A callable that raises or returns non-finite
    values ranks ``inf``.
    """
    f = np.asarray(fvec, dtype=float)
    if isinstance(loss, str) and loss == "linear":
        return 0.5 * float(np.sum(np.square(f)))
    import jax.numpy as jnp

    try:
        loss_function = _loss_funcs().get_loss_function(loss)
        cost = float(
            loss_function(
                jnp.asarray(f),
                f_scale,
                data_mask=jnp.ones(f.shape, bool),
                cost_only=True,
            )
        )
    except Exception:
        return np.inf
    return cost if np.isfinite(cost) else np.inf


def _attempt_cost(result, attempt_loss, caller_loss, flags, f_scale) -> float:
    """Comparable cost of a failed attempt, ``inf`` when it cannot be derived."""
    same = attempt_loss == caller_loss
    kind = _tuple_kind(result, flags)
    if kind is None:
        if same:
            return float(getattr(result, "cost", np.inf))
        return 0.5 * float(np.sum(np.square(np.asarray(result.fun))))
    if kind == "full":
        fvec = result[2].get("fvec")
        if fvec is None:
            return np.inf
        # the attempt's own loss when it is the caller's, else linear SSR
        return _robust_cost(fvec, attempt_loss if same else "linear", f_scale)
    if kind == "timeit" and same:  # raw result has cost but no fun
        return float(result[2].get("cost", np.inf))
    return np.inf


def _keep_best(
    best,
    result,
    attempt_loss,
    caller_loss,
    strategy_name=None,
    flags=None,
    f_scale=1.0,
):
    """Keep the best unsuccessful ``(key, result, strategy_name)`` triple.

    Robust-loss costs (sum of rho) are always <= 0.5*SSR and not comparable with
    linear ones, so an attempt run under a different loss than the caller's is
    ranked by its linear ``0.5*sum(fun**2)``; for a non-linear caller loss only
    same-loss attempts compete first. Attempts with non-finite parameters rank
    last; an attempt whose cost cannot be derived (status-less tuples, callable
    losses) ranks ``inf`` and so the first such finite attempt is kept.
    """
    try:
        cost = _attempt_cost(result, attempt_loss, caller_loss, flags, f_scale)
    except (AttributeError, TypeError, ValueError, KeyError):
        cost = np.inf
    if not np.isfinite(cost):
        cost = np.inf
    popt = _fitted_params(result, flags)
    nonfinite = popt is not None and not bool(np.all(np.isfinite(popt)))
    key = (nonfinite, caller_loss not in (attempt_loss, "linear"), cost)
    if best is None or key < best[0]:
        return (key, result, strategy_name)
    return best


def _is_noop(modified, kwargs) -> bool:
    """A proposal equal to the current kwargs (an in-place mutation, returning
    the very same object, cannot be told apart and counts as a real attempt)."""
    return modified is not kwargs and _same_kwargs(modified, kwargs)


def _propose(strategy, kwargs):
    """``(strategy.apply(kwargs), None)``, or ``(None, error)`` if apply() raised."""
    try:
        return strategy.apply(kwargs), None
    except Exception as e:
        return None, e


class FallbackResult:
    """Enhanced optimization result with fallback information."""

    def __init__(self, result, strategy_used: str | None = None, attempts: int = 1):
        """
        Initialize fallback result.

        Parameters
        ----------
        result : OptimizeResult
            Underlying optimization result
        strategy_used : str, optional
            Name of the fallback strategy that produced ``result`` (None for
            the original attempt)
        attempts : int, optional
            Number of attempts before success
        """
        self.result = result
        self.fallback_strategy_used = strategy_used
        self.fallback_attempts = attempts

    def __getattr__(self, name):
        """Delegate attribute access to underlying result."""
        if name == "result":  # not yet set (e.g. during copy/unpickle): avoid recursion
            raise AttributeError(name)
        return getattr(self.result, name)

    # Dunder lookups bypass __getattr__, so delegate explicitly to let
    # ``popt, pcov = curve_fit(..., fallback=True)`` work like a normal return.
    def __iter__(self):
        return iter(self.result)

    def __getitem__(self, key):
        return self.result[key]

    def __len__(self):
        return len(self.result)

    def __contains__(self, key):
        # Without this, ``in`` falls back to __iter__ and compares arrays (ValueError).
        # Mirrors attribute delegation: ``"popt" in r`` iff ``r.popt`` resolves.
        return key in self.result or (isinstance(key, str) and hasattr(self, key))


class FallbackOrchestrator:
    """
    Orchestrates automatic fallback strategies for robust optimization.

    The orchestrator tries multiple recovery strategies when optimization fails,
    including alternative methods, perturbed initial guesses, relaxed tolerances,
    and robust loss functions.

    Parameters
    ----------
    strategies : list of FallbackStrategy, optional
        List of strategies to try. If None, uses default strategies.
    max_attempts : int, optional
        Maximum total attempts across all strategies. Default: 10
    verbose : bool, optional
        Print progress messages. Default: False

    Attributes
    ----------
    strategies : list of FallbackStrategy
        Active fallback strategies, sorted by priority
    total_attempts : int
        Number of fit attempts made in the most recent ``fit_with_fallback()``
        call; reset at the start of every call. ``get_statistics()`` reports
        the lifetime total across every call this instance has made.
    successful_strategies : dict
        Count of successes per strategy

    Examples
    --------
    >>> from nlsq.stability.fallback import FallbackOrchestrator
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>>
    >>> def model(x, a, b):
    ...     return a * jnp.exp(-b * x)
    >>>
    >>> x = np.linspace(0, 10, 100)
    >>> y = 2.5 * np.exp(-0.5 * x) + 0.1 * np.random.randn(100)
    >>>
    >>> orchestrator = FallbackOrchestrator(verbose=True)
    >>> result = orchestrator.fit_with_fallback(
    ...     model, x, y, p0=[1, 0.1]  # Deliberately poor p0
    ... )
    >>>
    >>> if result.fallback_strategy_used:
    ...     print(f"Recovered using: {result.fallback_strategy_used}")
    """

    DEFAULT_STRATEGIES: ClassVar[list[StrategyFactory]] = [
        AlternativeMethodStrategy,
        PerturbInitialGuessStrategy,
        AdjustTolerancesStrategy,
        AddParameterBoundsStrategy,
        UseRobustLossStrategy,
        RescaleProblemStrategy,
    ]

    def __init__(
        self,
        strategies: list[FallbackStrategy] | None = None,
        max_attempts: int = 10,
        verbose: bool = False,
    ):
        """Initialize fallback orchestrator."""
        self.logger = get_logger(__name__)
        self.verbose = verbose

        # Initialize strategies
        if strategies is None:
            self.strategies = [strategy() for strategy in self.DEFAULT_STRATEGIES]
        else:
            self.strategies = strategies

        # Sort by priority (highest first); a copy, so the caller's list is untouched
        self.strategies = sorted(
            self.strategies, key=lambda s: s.priority, reverse=True
        )

        self.max_attempts = max_attempts
        # Per-fit attempt counter used for the max_attempts cutoff and for
        # numbering attempts within the current fit; reset every call to
        # fit_with_fallback() (see test_second_fit_resets_total_attempts).
        self.total_attempts = 0
        # Lifetime stat across every fit_with_fallback() call this instance
        # has made; exposed via get_statistics()["total_attempts"]. Never
        # reset (see test_statistics_tracking).
        self._lifetime_attempts = 0
        self.successful_strategies: dict[str, int] = {}

    def fit_with_fallback(self, f: Callable, xdata, ydata, **kwargs) -> FallbackResult:
        """
        Attempt curve fit with automatic fallback on failure.

        Parameters
        ----------
        f : callable
            Model function
        xdata : array_like
            Independent variable data
        ydata : array_like
            Dependent variable data
        **kwargs
            Additional arguments passed to curve_fit

        Returns
        -------
        result : FallbackResult
            Optimization result with fallback metadata

        Notes
        -----
        ``full_output`` / ``return_eval`` / ``timeit`` are passed to every
        attempt and the winning ``curve_fit`` return is handed back verbatim
        (no re-run, no re-formatting), so stateful callbacks run once per
        attempt exactly as in an ordinary fit. Success detection depends on
        what the flagged return carries: ``full_output`` has ``ier`` and
        ``timeit`` the raw result's ``success``, but ``return_eval`` (alone or
        combined with the other flags, which it overrides) has no status, so
        there a finite fit counts as converged and fallback only triggers on
        exceptions or non-finite parameters. Use ``full_output=True`` or
        ``timeit=True`` without ``return_eval`` for status-aware fallback.

        Raises
        ------
        RuntimeError
            If all fallback strategies fail
        """
        from nlsq.core.minpack import (
            curve_fit,  # Deferred import to avoid circular dependency
        )
        from nlsq.core.problem import _is_auto_p0, _n_params_from_signature

        # Reset per-fit state: this orchestrator instance is commonly reused
        # across multiple fits, but total_attempts and each strategy's
        # progressive sequence position (e.g. which robust loss function or
        # tolerance relaxation to try next) must start fresh for *this* fit,
        # not resume wherever the previous fit's sequence left off.
        # self._lifetime_attempts is a lifetime stat and is NOT reset here.
        self.total_attempts = 0
        for strategy in self.strategies:
            strategy.reset()

        # Inject xdata/ydata for strategies that need it
        kwargs["_xdata"] = xdata
        kwargs["_ydata"] = ydata

        # Try original parameters first
        self.total_attempts += 1
        self._lifetime_attempts += 1

        if self.verbose:
            print(f"Attempt 1/{self.max_attempts}: Original parameters")

        best: tuple[Any, Any, str | None] | None = None  # best unsuccessful attempt
        caller_loss = kwargs.get("loss", "linear")
        flags = _flags(kwargs)
        last_exception: Exception | None = None
        try:
            # _xdata/_ydata are for strategies only; curve_fit would warn on them.
            result = curve_fit(
                f,
                xdata,
                ydata,
                **{k: v for k, v in kwargs.items() if k not in ("_xdata", "_ydata")},
            )
            if _fit_succeeded(result, flags):
                if self.verbose:
                    print("Success with original parameters!")
                return FallbackResult(result, strategy_used=None, attempts=1)
            if self.verbose:
                print("[FAIL] Fit returned success=False")
            best = _keep_best(
                best,
                result,
                caller_loss,
                caller_loss,
                flags=flags,
                f_scale=kwargs.get("f_scale") or 1.0,
            )
        except Exception as e:
            if self.verbose:
                print(f"[FAIL] Failed: {type(e).__name__}: {e}")
            last_exception = e

        # Strategies perturb/derive bounds from p0 and default it to [1.0];
        # give them curve_fit's own default (ones, sized from the signature).
        # The original attempt above keeps curve_fit's estimate_p0/auto path.
        p0 = kwargs.get("p0")
        if p0 is None or _is_auto_p0(p0):
            n_params = _n_params_from_signature(f)
            if n_params >= 1:
                kwargs["p0"] = np.ones(n_params)

        return self._try_strategies(
            curve_fit, f, xdata, ydata, kwargs, best, last_exception, caller_loss
        )

    def _try_strategies(
        self, curve_fit, f, xdata, ydata, kwargs, best, last_exception, caller_loss
    ) -> FallbackResult:
        """Run the strategies; return the first success, else best result or raise."""
        # Try fallback strategies: round-robin passes, so progressive strategies
        # (perturbations, loss sequence, ...) get re-queried until exhausted.
        pending = list(self.strategies)
        flags = _flags(kwargs)
        idle_passes = 0
        while pending and self.total_attempts < self.max_attempts:
            spent = self.total_attempts
            for strategy in list(pending):
                if self.total_attempts >= self.max_attempts:
                    break

                # A strategy with nothing new to propose would just repeat the
                # previous solve: skip it without spending an attempt.
                modified, apply_error = _propose(strategy, kwargs)
                if apply_error is None and _is_noop(modified, kwargs):
                    if not strategy.has_more():
                        pending.remove(strategy)
                    continue

                self.total_attempts += 1
                self._lifetime_attempts += 1
                strategy.attempts += 1

                self._say(
                    f"\nAttempt {self.total_attempts}/{self.max_attempts}: "
                    f"{strategy.description}"
                )

                try:
                    if apply_error is not None:
                        raise apply_error
                    result, attempt_loss, pinned = self._run_strategy(
                        strategy, curve_fit, f, xdata, ydata, kwargs, modified
                    )

                    if _fit_succeeded(result, flags) and not pinned:
                        self._record_success(strategy)
                        self._say(f"[OK] Success with {strategy.name}!")
                        return FallbackResult(
                            result,
                            strategy_used=strategy.name,
                            attempts=self.total_attempts,
                        )
                    self._say(f"[FAIL] {strategy.name} returned success=False")
                    best = _keep_best(
                        best,
                        result,
                        attempt_loss,
                        caller_loss,
                        strategy.name,
                        flags=flags,
                        f_scale=modified.get("f_scale") or 1.0,
                    )

                except Exception as e:
                    self._say(f"[FAIL] Failed: {type(e).__name__}")
                    last_exception = e

                if not strategy.has_more():
                    pending.remove(strategy)

            if self.total_attempts == spent:
                # A pass of only no-ops spends no attempt, but progressive
                # strategies advanced (pending keeps those with has_more()), so a
                # later pass can still propose a new variant. Bound the idle
                # passes so a strategy stuck on has_more() cannot loop forever.
                idle_passes += 1
                if idle_passes > self.max_attempts:
                    break
            else:
                idle_passes = 0

        if best is not None:
            # No attempt converged: hand back the lowest-cost result, still
            # flagged success=False, rather than raising. strategy_used names the
            # attempt that produced it (None = the original attempt).
            self.logger.warning(
                f"All {self.total_attempts} fallback attempts failed to converge; "
                "returning best result"
            )
            if _tuple_kind(best[1], flags) == "eval":
                # (popt, pcov, f(x)) has no status field, so the caller cannot
                # tell this best-effort return from a converged fit.
                warnings.warn(
                    "No fallback attempt converged; returning the best "
                    "return_eval result, which carries no status (popt may be "
                    "unconverged or non-finite). Check the fit before using it.",
                    OptimizeWarning,
                    stacklevel=3,
                )
            return FallbackResult(
                best[1], strategy_used=best[2], attempts=self.total_attempts
            )

        # All strategies failed
        error_msg = (
            f"All {self.total_attempts} fallback attempts failed. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )
        self.logger.error(error_msg)
        raise RuntimeError(error_msg) from last_exception

    def _say(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _record_success(self, strategy) -> None:
        strategy.successes += 1
        self.successful_strategies[strategy.name] = (
            self.successful_strategies.get(strategy.name, 0) + 1
        )

    @staticmethod
    def _run_strategy(strategy, curve_fit, f, xdata, ydata, kwargs, modified=None):
        """Apply one strategy's kwargs modifications and run the fit.

        ``modified`` is the strategy's already-computed ``apply(kwargs)`` output
        (``apply`` is stateful for progressive strategies, so it is not re-run).
        Returns ``(result, attempt_loss, pinned)``; ``pinned`` is True when the
        fit ended on a bound this orchestrator invented (it is never a success).
        """
        modified_kwargs = strategy.apply(kwargs) if modified is None else modified

        # Remove internal markers
        modified_kwargs.pop("_xdata", None)
        modified_kwargs.pop("_ydata", None)
        attempt_loss = modified_kwargs.get("loss", "linear")

        # RescaleProblemStrategy can't rewrite xdata/ydata/f itself (its apply()
        # only sees kwargs), so it signals a rescaled fit via internal keys;
        # _fit_rescaled builds the scaled problem and returns a result in the
        # user's original units.
        scale_info = {
            k: modified_kwargs.pop(k, None)
            for k in ("_x_scale", "_y_scale", "_x_offset", "_y_offset")
        }
        if modified_kwargs.pop("_scaled", False):
            return (
                _fit_rescaled(curve_fit, f, xdata, ydata, modified_kwargs, scale_info),
                attempt_loss,
                False,
            )
        invented_bounds = modified_kwargs.pop("_invented_bounds", False)
        result = curve_fit(f, xdata, ydata, **modified_kwargs)
        flags = _flags(modified_kwargs)
        pinned = invented_bounds and _hits_bound(
            result, modified_kwargs["bounds"], flags, modified_kwargs.get("p0")
        )
        if pinned:
            # The box was our guess, not the user's: a solution pinned to it
            # is an artefact of the guess, never a converged fit.
            result = _mark_unsuccessful(result, flags)
        return result, attempt_loss, pinned

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics on fallback strategy performance.

        Returns
        -------
        stats : dict
            Dictionary with success rates and attempt counts
        """
        stats: dict[str, Any] = {
            "total_attempts": self._lifetime_attempts,
            "strategies": [],
        }

        for strategy in self.strategies:
            strategy_stats = {
                "name": strategy.name,
                "description": strategy.description,
                "attempts": strategy.attempts,
                "successes": strategy.successes,
                "success_rate": (
                    strategy.successes / strategy.attempts
                    if strategy.attempts > 0
                    else 0.0
                ),
            }
            stats["strategies"].append(strategy_stats)

        return stats

    def print_statistics(self):
        """Print human-readable statistics."""
        stats = self.get_statistics()

        print("=" * 60)
        print("FALLBACK ORCHESTRATOR STATISTICS")
        print("=" * 60)
        print(f"Total attempts: {stats['total_attempts']}")
        print("\nStrategy Performance:")
        print("-" * 60)

        for s in stats["strategies"]:
            if s["attempts"] > 0:
                print(
                    f"  {s['name']:20s}: {s['successes']:3d}/{s['attempts']:3d} "
                    f"({s['success_rate'] * 100:5.1f}%)",
                )
                print(f"    └─ {s['description']}")

        print("=" * 60)
