"""Curve fitting adapter implementing OptimizerProtocol.

This module provides an adapter that wraps the core curve_fit functionality
behind the OptimizerProtocol interface, enabling dependency injection and
loose coupling in the NLSQ architecture.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nlsq.caching.unified_cache import UnifiedCache
    from nlsq.diagnostics.types import DiagnosticsConfig
    from nlsq.stability.guard import NumericalStabilityGuard


class CurveFitAdapter:
    """Adapter that provides curve fitting via protocol interface.

    This adapter wraps the core curve_fit functionality and implements
    CurveFitProtocol, allowing it to be used interchangeably with other
    curve fitting implementations.

    Parameters
    ----------
    cache : UnifiedCache | None
        Optional cache for JIT compilation and results.
    stability_guard : NumericalStabilityGuard | None
        Optional numerical stability guard.
    diagnostics_config : DiagnosticsConfig | None
        Optional diagnostics configuration.

    Examples
    --------
    >>> adapter = CurveFitAdapter()
    >>> popt, pcov = adapter.curve_fit(model, xdata, ydata, p0=[1.0, 0.1])
    """

    __slots__ = (
        "_cache",
        "_diagnostics_config",
        "_global_config",
        "_stability_guard",
    )

    def __init__(
        self,
        cache: "UnifiedCache | None" = None,
        stability_guard: "NumericalStabilityGuard | None" = None,
        diagnostics_config: "DiagnosticsConfig | None" = None,
    ) -> None:
        """Initialize the adapter with optional dependencies."""
        self._cache = cache
        self._stability_guard = stability_guard
        self._diagnostics_config = diagnostics_config
        self._global_config: Any = None

    def curve_fit(
        self,
        f: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray] | Any:
        """Fit a function to data.

        Delegates to the core curve_fit implementation while providing
        a clean protocol-based interface.

        Parameters
        ----------
        f : Callable
            Model function ``f(x, *params) -> y``.
        xdata : np.ndarray
            Independent variable data.
        ydata : np.ndarray
            Dependent variable data.
        p0 : np.ndarray or None
            Initial parameter guess.
        sigma : np.ndarray or None
            Uncertainty in ydata.
        bounds : tuple or None
            (lower, upper) bounds for parameters.
        **kwargs : Any
            Additional keyword arguments passed to curve_fit.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (popt, pcov) - optimal parameters and covariance matrix.
        """
        # Import here to avoid circular dependency
        from nlsq.core.minpack import curve_fit as _curve_fit

        # Inject dependencies if provided
        if self._cache is not None:
            kwargs.setdefault("_cache", self._cache)
        if self._stability_guard is not None:
            kwargs.setdefault("_stability_guard", self._stability_guard)
        if self._diagnostics_config is not None:
            kwargs.setdefault("diagnostics", self._diagnostics_config)

        return _curve_fit(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            **kwargs,
        )

    @staticmethod
    def with_global_optimization(
        global_config: Any | None = None,
    ) -> "CurveFitAdapter":
        """Create an adapter configured for global optimization.

        Parameters
        ----------
        global_config : GlobalOptimizationConfig | None
            Configuration for global optimization.

        Returns
        -------
        CurveFitAdapter
            Adapter configured for global optimization.
        """
        adapter = CurveFitAdapter()
        if global_config is not None:
            adapter._global_config = global_config
        return adapter


# Note: Protocol conformance assertion moved to tests/core/adapters/test_curve_fit_adapter.py
# to avoid import-time overhead (~5-10ms per module load)
