"""Builder for the two large-dataset engines.

``build_large_strategy(kind, config, **common)`` constructs the engine for a path
the caller has already chosen (``"hybrid"``: ``AdaptiveHybridStreamingOptimizer``;
``"chunked"``: ``LargeDatasetFitter``). It never decides routing. The returned
``Strategy`` owns the input kwargs translation for its engine and returns the
engine's native result unchanged (a dict for hybrid, an ``OptimizeResult`` for
chunked); callers keep their own result unpacking.

The engine classes are looked up when a strategy is built, not at import time,
so ``mock.patch("nlsq.streaming.<module>.<Class>")`` still takes effect and
``large_dataset`` can import this module without a cycle.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Protocol

__all__ = ["Strategy", "build_large_strategy"]

# AdaptiveHybridStreamingOptimizer.fit has no gtol/ftol/xtol: its single
# convergence knob is config.gauss_newton_tol, set at construction.
_HYBRID_DROPPED_TOLERANCES = ("gtol", "ftol", "xtol")


class Strategy(Protocol):
    """A constructed large-dataset engine behind a uniform ``fit`` call."""

    engine: Any

    def fit(self, f: Any, x: Any, y: Any, p0: Any, /, **common: Any) -> Any:
        """Run the engine; return its native result unchanged."""
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
        for key in _HYBRID_DROPPED_TOLERANCES:
            common.pop(key, None)
        return self.engine.fit(data_source=(x, y), func=f, p0=p0, **common)


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


def build_large_strategy(
    kind: Literal["hybrid", "chunked"], config: Any, **common: Any
) -> Strategy:
    """Construct the engine for ``kind`` now and return it as a ``Strategy``.

    Parameters
    ----------
    kind : {"hybrid", "chunked"}
        The path the caller already chose.
    config : HybridStreamingConfig or LDMemoryConfig or None
        Engine config. Hybrid: used as is unless ``gtol`` is given, in which
        case a copy with ``gauss_newton_tol=gtol`` is used. Chunked: passed to
        ``LargeDatasetFitter(config=...)`` (``None`` lets the fitter build one).
    **common
        Hybrid: optional ``gtol``. Chunked: ``LargeDatasetFitter`` constructor
        kwargs (``memory_limit_gb``, ``logger``, ``multistart``, ...) plus
        ``show_progress`` to make ``fit`` call ``fit_with_progress``.

    ``Strategy.fit(f, x, y, p0, **common)`` forwards ``common`` to the engine's
    fit. Hybrid drops ``gtol``/``ftol``/``xtol`` (no equivalent at fit time) and
    passes everything else to ``AdaptiveHybridStreamingOptimizer.fit``; chunked
    forwards everything, tolerances included.
    """
    if kind == "hybrid":
        return _HybridStrategy(config, **common)
    if kind == "chunked":
        return _ChunkedStrategy(config, **common)
    raise ValueError(
        f"unknown large-strategy kind {kind!r}; expected 'hybrid' or 'chunked'"
    )
