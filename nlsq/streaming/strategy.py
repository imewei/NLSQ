"""Builder for the large-dataset engines.

``build_large_strategy(kind, config, **common)`` constructs the engine for a path
the caller has already chosen (``"hybrid"``: ``AdaptiveHybridStreamingOptimizer``;
``"chunked"``: ``LargeDatasetFitter``; ``"exact"``: ``exact_lm_fit``;
``"in_memory"``: ``nlsq.curve_fit``). It never decides routing. The returned
``Strategy`` owns the input kwargs translation for its engine and its ``fit``
returns the engine's native result unchanged (a dict for hybrid, an
``OptimizeResult`` for chunked and exact, a ``CurveFitResult`` or special-output
tuple for in-memory); ``unpack`` turns that into what ``curve_fit_large``
returns (``(popt, pcov)``, or the in-memory result unchanged).

The engines are looked up when a strategy is built, not at import time,
so ``mock.patch("nlsq.streaming.<module>.<Class>")`` still takes effect and
``large_dataset`` can import this module without a cycle.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Callable
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike

from nlsq.result.optimize_warning import OptimizeWarning

__all__ = [
    "Strategy",
    "build_large_strategy",
    "check_exact_sources",
    "default_p0",
    "normalize_p0_bounds",
]

# AdaptiveHybridStreamingOptimizer.fit has no gtol/ftol/xtol: its single
# convergence knob is config.gauss_newton_tol, set at construction.
_HYBRID_DROPPED_TOLERANCES = ("gtol", "ftol", "xtol")


class Strategy(Protocol):
    """A constructed large-dataset engine behind a uniform ``fit`` call."""

    engine: Any

    def fit(self, f: Any, x: Any, y: Any, p0: Any, /, **common: Any) -> Any:
        """Run the engine; return its native result unchanged."""
        ...

    def unpack(self, native: Any, /) -> Any:
        """Turn ``fit``'s native result into what ``curve_fit_large`` returns."""
        ...


class _HybridStrategy:
    def __init__(self, config: Any, **common: Any) -> None:
        from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer

        unknown = set(common) - {"gtol"}
        if unknown:
            raise TypeError(f"hybrid strategy got unexpected {sorted(unknown)}")
        # gtol maps onto gauss_newton_tol on a copy; without it the caller's
        # config object is passed through as is.
        if "gtol" in common:
            config = dataclasses.replace(config, gauss_newton_tol=common["gtol"])
        self.engine = AdaptiveHybridStreamingOptimizer(config=config)

    def fit(self, f: Any, x: Any, y: Any, p0: Any, /, **common: Any) -> Any:
        """Call ``AdaptiveHybridStreamingOptimizer.fit``.

        ``gtol`` is honoured only at build time; given here it is dropped (with
        ``ftol``/``xtol``) and the config's ``gauss_newton_tol`` is used.
        """
        for key in _HYBRID_DROPPED_TOLERANCES:
            common.pop(key, None)
        return self.engine.fit(data_source=(x, y), func=f, p0=p0, **common)

    def unpack(self, native: Any, /) -> tuple[np.ndarray, np.ndarray]:
        x = native["x"]
        if not native.get("success", True):
            if native.get("message") != "Maximum iterations reached":
                raise RuntimeError(
                    f"hybrid_streaming fit failed: {native.get('message')}"
                )
            # An iteration-budget stop returns its best point, like the
            # chunked "nonconverged" case (the engine already warned too).
            warnings.warn(
                "hybrid_streaming stopped at max iterations; result may not be "
                "converged",
                OptimizeWarning,
                stacklevel=2,
            )
        return x, native.get("pcov", np.full((len(x), len(x)), np.inf))


def scipy_layout_to_points_first(f: Any, x: Any) -> tuple[Any, Any]:
    """Adapt a scipy-layout ``(k, M)`` problem to the hybrid engine's contract.

    The hybrid engine (and ``_HybridStrategy.fit``) takes multi-dimensional x
    points-first, ``(M, k)``, and chunks along axis 0. Callers holding scipy's
    ``(k, M)`` layout pass ``f, x`` through here: it returns the transposed x
    and a model that gives the user's ``f`` its ``(k, chunk)`` layout back.
    No-op for 1-D x.
    """
    if np.ndim(x) <= 1:
        return f, x

    @functools.wraps(f)
    def f_points_first(xc: Any, *params: Any) -> Any:
        return f(xc.T, *params)

    return f_points_first, np.asarray(x).T


class _ChunkedStrategy:
    def __init__(
        self, config: Any, *, show_progress: bool = False, **common: Any
    ) -> None:
        from nlsq.streaming.large_dataset import LargeDatasetFitter

        # show_progress is consumed here, at build time, so a show_progress
        # kwarg given to fit() is forwarded to the engine untouched.
        self._show_progress = show_progress
        self.engine = LargeDatasetFitter(config=config, **common)

    def fit(self, f: Any, x: Any, y: Any, p0: Any, /, **common: Any) -> Any:
        run = self.engine.fit_with_progress if self._show_progress else self.engine.fit
        return run(f, x, y, p0=p0, **common)

    def unpack(self, native: Any, /) -> tuple[np.ndarray, np.ndarray | None]:
        # nonconverged: curve_fit already warned and returned its best point,
        # so only a hard failure raises here.
        if not getattr(native, "success", True) and not getattr(
            native, "nonconverged", False
        ):
            raise RuntimeError(
                f"chunked fit failed: {getattr(native, 'message', None)}"
            )
        if hasattr(native, "popt") and hasattr(native, "pcov"):
            return native.popt, native.pcov
        raise RuntimeError(
            f"Unexpected result format from large dataset fitter: {native}"
        )


class _ExactStrategy:
    def __init__(self, config: Any, **common: Any) -> None:
        from nlsq.streaming import exact_lm

        if config is not None or common:
            raise TypeError(
                "exact strategy takes no config and no build kwargs; "
                "pass fit kwargs to Strategy.fit"
            )
        self.engine = exact_lm.exact_lm_fit

    def fit(self, f: Any, x: Any, y: Any, p0: Any, /, **common: Any) -> Any:
        """Call ``exact_lm_fit(f, x, y, p0, **common)``."""
        return self.engine(f, x, y, p0, **common)

    def unpack(self, native: Any, /) -> tuple[np.ndarray, np.ndarray | None]:
        if not native["success"]:
            from nlsq.streaming.exact_lm import ExactSolverError

            raise ExactSolverError(
                f"exact_streaming fit failed ({native['termination']}): "
                f"{native['message']}",
                native,
            )
        return native["x"], native["pcov"]


class _InMemoryStrategy:
    def __init__(self, config: Any, **common: Any) -> None:
        if config is not None or common:
            raise TypeError(
                "in_memory strategy takes no config and no build kwargs; "
                "pass fit kwargs to Strategy.fit"
            )
        import nlsq

        # Looked up on the package, as curve_fit_large always has, so a patch of
        # nlsq.curve_fit takes effect (nlsq.core.minpack.curve_fit does not).
        self.engine = nlsq.curve_fit

    def fit(self, f: Any, x: Any, y: Any, p0: Any, /, **common: Any) -> Any:
        """Call ``curve_fit(f, x, y, **common)``, with ``p0=`` only when given."""
        if p0 is not None:
            common["p0"] = p0
        return self.engine(f, x, y, **common)

    def unpack(self, native: Any, /) -> Any:
        return native


def build_large_strategy(
    kind: Literal["hybrid", "chunked", "exact", "in_memory"],
    config: Any,
    **common: Any,
) -> Strategy:
    """Construct the engine for ``kind`` now and return it as a ``Strategy``.

    Parameters
    ----------
    kind : {"hybrid", "chunked", "exact", "in_memory"}
        The path the caller already chose.
    config : HybridStreamingConfig or LDMemoryConfig or None
        Must be ``None`` for ``"exact"`` and ``"in_memory"``.
        Engine config. Hybrid: used as is unless ``gtol`` is given, in which
        case a copy with ``gauss_newton_tol=gtol`` is used. Chunked: passed to
        ``LargeDatasetFitter(config=...)`` (``None`` lets the fitter build one).
    **common
        Hybrid: optional ``gtol``. Chunked: ``LargeDatasetFitter`` constructor
        kwargs (``memory_limit_gb``, ``logger``, ``multistart``, ...) plus
        ``show_progress`` to make ``fit`` call ``fit_with_progress``. Exact and
        in-memory: none (build-time kwargs raise ``TypeError``).

    Notes
    -----
    ``Strategy.fit(f, x, y, p0, **common)`` forwards ``common`` to the engine's
    fit. Hybrid honours ``gtol`` only at build time: passed to ``fit`` it is
    dropped, as are ``ftol``/``xtol`` (no equivalent), and the config's
    ``gauss_newton_tol`` is used; everything else goes to
    ``AdaptiveHybridStreamingOptimizer.fit``. Building hybrid with ``ftol`` or
    ``xtol`` raises ``TypeError``. Chunked forwards everything, tolerances
    included. Exact forwards ``common`` to ``exact_lm_fit`` (``sigma``,
    ``bounds``, ``chunk_size``, tolerances, ...) and returns its
    ``OptimizeResult``. In-memory forwards ``common`` to ``nlsq.curve_fit`` (``p0``
    only when not ``None``) and its ``unpack`` returns the result unchanged.
    """
    if kind == "hybrid":
        return _HybridStrategy(config, **common)
    if kind == "chunked":
        return _ChunkedStrategy(config, **common)
    if kind == "exact":
        return _ExactStrategy(config, **common)
    if kind == "in_memory":
        return _InMemoryStrategy(config, **common)
    raise ValueError(
        f"unknown large-strategy kind {kind!r}; expected 'hybrid', 'chunked', "
        "'exact' or 'in_memory'"
    )


def default_p0(f: Callable[..., Any]) -> np.ndarray:
    """Ones for positional fit parameters after ``x`` (curve_fit's p0=None)."""
    from nlsq.core.problem import _n_params_from_signature

    n_params = _n_params_from_signature(f)
    if n_params < 1:
        raise ValueError("Unable to determine number of fit parameters.")
    return np.ones(n_params)


def normalize_p0_bounds(
    f: Callable[..., Any], p0: ArrayLike | None, bounds: Any
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return ``(p0, (lb, ub))``: ``p0`` defaulted from ``f`` and 1-D, bounds broadcast."""
    from nlsq.core.least_squares import prepare_bounds

    p0_arr = default_p0(f) if p0 is None else np.atleast_1d(p0)
    return p0_arr, prepare_bounds(bounds, len(p0_arr))


def check_exact_sources(
    xdata: Any, ydata: Any, sigma: Any, n_points: int | None
) -> None:
    """Validate ``method="exact_streaming"`` data sources from ``shape``/``len`` only.

    Array-like sources (``np.memmap``, ``h5py.Dataset``, ``zarr.Array``, any object
    with ``shape``, ``len`` and slicing) are never read whole here; objects without
    ``shape`` (lists, tuples) are converted. A callable ``xdata`` is a chunk factory:
    ``ydata`` and ``sigma`` must be None and ``n_points`` (accepted only then) is
    optional; blocks are validated as they arrive.
    """
    from nlsq.streaming import exact_lm

    factory = exact_lm.is_chunk_factory(xdata)
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
        return
    if n_points is not None:
        raise ValueError(
            "n_points is accepted only with a chunk factory (callable xdata); array "
            "sources take their length from their shape"
        )

    lengths = {}
    for name, a in (("xdata", xdata), ("ydata", ydata), ("sigma", sigma)):
        if name == "sigma" and a is None:
            continue
        shape = a.shape if hasattr(a, "shape") else np.shape(a)
        if name == "sigma" and len(shape) == 0:
            continue  # a scalar sigma is broadcast per chunk
        if len(shape) != 1:
            why = (
                "; multi-feature (2-D) xdata is not supported by exact_streaming"
                if name == "xdata" and len(shape) > 1
                else ""
            )
            raise ValueError(
                f"{name} must be 1-D for method='exact_streaming', got shape {shape}{why}"
            )
        lengths[name] = shape[0]
    n_x, n_y = lengths["xdata"], lengths["ydata"]
    if n_x != n_y:
        raise ValueError(
            f"`xdata` and `ydata` must have the same length: {n_x} vs {n_y}.",
        )
    if n_x < 2:
        raise ValueError(f"Need at least 2 data points for fitting, got {n_x}.")
