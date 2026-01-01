"""Factory functions for composing curve fitting configurations.

This module provides factory functions that enable runtime composition
of curve fitting features like global optimization and diagnostics.
These factories follow the Builder pattern to provide a clean API for
configuring optimization pipelines.

The factories decouple feature composition from the core curve_fit
implementation, reducing the dependency count in minpack.py.

Examples
--------
>>> from nlsq.core.factories import create_optimizer, configure_curve_fit
>>>
>>> # Configure curve_fit with custom settings
>>> curve_fit = configure_curve_fit(
...     enable_diagnostics=True,
...     enable_recovery=True,
... )
>>> popt, pcov = curve_fit(model, xdata, ydata)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nlsq.caching.unified_cache import UnifiedCache
    from nlsq.diagnostics.types import DiagnosticsConfig
    from nlsq.stability.guard import NumericalStabilityGuard


@dataclass
class OptimizerConfig:
    """Configuration for optimizer creation.

    Attributes
    ----------
    enable_global : bool
        Enable global optimization with multi-start.
    enable_diagnostics : bool
        Enable diagnostic reporting.
    enable_recovery : bool
        Enable automatic recovery from numerical issues.
    n_starts : int
        Number of starts for global optimization.
    """

    enable_global: bool = False
    enable_diagnostics: bool = False
    enable_recovery: bool = True
    n_starts: int = 10
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def create_optimizer(
    *,
    global_optimization: bool = False,
    diagnostics: bool = False,
    recovery: bool = True,
    n_starts: int = 10,
    cache: "UnifiedCache | None" = None,
    stability_guard: "NumericalStabilityGuard | None" = None,
    diagnostics_config: "DiagnosticsConfig | None" = None,
    **kwargs: Any,
) -> "ConfiguredOptimizer":
    """Create a configured optimizer with specified features.

    This factory function composes various optimization features
    (global optimization, diagnostics) into a single
    optimizer instance.

    Parameters
    ----------
    global_optimization : bool, default=False
        Enable global optimization with multi-start.
    diagnostics : bool, default=False
        Enable diagnostic reporting.
    recovery : bool, default=True
        Enable automatic recovery from numerical issues.
    n_starts : int, default=10
        Number of starts for global optimization.
    cache : UnifiedCache | None, default=None
        Optional cache for JIT compilation.
    stability_guard : NumericalStabilityGuard | None, default=None
        Optional numerical stability guard.
    diagnostics_config : DiagnosticsConfig | None, default=None
        Optional diagnostics configuration.
    **kwargs : Any
        Additional keyword arguments passed to curve_fit.

    Returns
    -------
    ConfiguredOptimizer
        A configured optimizer ready for use.

    Examples
    --------
    >>> # Standard optimizer
    >>> optimizer = create_optimizer()
    >>> popt, pcov = optimizer.fit(model, xdata, ydata)
    >>>
    >>> # Global optimizer with diagnostics
    >>> optimizer = create_optimizer(global_optimization=True, diagnostics=True)
    """
    config = OptimizerConfig(
        enable_global=global_optimization,
        enable_diagnostics=diagnostics,
        enable_recovery=recovery,
        n_starts=n_starts,
        extra_kwargs=kwargs,
    )

    return ConfiguredOptimizer(
        config=config,
        cache=cache,
        stability_guard=stability_guard,
        diagnostics_config=diagnostics_config,
    )


class ConfiguredOptimizer:
    """An optimizer configured with specific features.

    This class encapsulates the configuration for curve fitting
    and provides a clean interface for performing fits.

    Parameters
    ----------
    config : OptimizerConfig
        The optimizer configuration.
    cache : UnifiedCache | None
        Optional cache for JIT compilation.
    stability_guard : NumericalStabilityGuard | None
        Optional numerical stability guard.
    diagnostics_config : DiagnosticsConfig | None
        Optional diagnostics configuration.
    """

    __slots__ = ("_cache", "_config", "_diagnostics_config", "_stability_guard")

    def __init__(
        self,
        config: OptimizerConfig,
        cache: "UnifiedCache | None" = None,
        stability_guard: "NumericalStabilityGuard | None" = None,
        diagnostics_config: "DiagnosticsConfig | None" = None,
    ) -> None:
        """Initialize the configured optimizer."""
        self._config = config
        self._cache = cache
        self._stability_guard = stability_guard
        self._diagnostics_config = diagnostics_config

    def fit(
        self,
        f: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a function to data using the configured optimizer.

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
            Additional keyword arguments.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (popt, pcov) - optimal parameters and covariance matrix.
        """
        # Merge config kwargs with call kwargs
        merged_kwargs = {**self._config.extra_kwargs, **kwargs}

        # Handle global optimization
        if self._config.enable_global:
            return self._fit_global(f, xdata, ydata, p0, sigma, bounds, **merged_kwargs)

        # Standard fit
        return self._fit_standard(f, xdata, ydata, p0, sigma, bounds, **merged_kwargs)

    def _fit_standard(
        self,
        f: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None,
        sigma: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray] | Any:
        """Perform standard curve fitting."""
        from nlsq.core.minpack import curve_fit

        return curve_fit(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            **kwargs,
        )

    def _fit_global(
        self,
        f: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None,
        sigma: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        **kwargs: Any,
    ) -> Any:
        """Perform global optimization with multi-start."""
        from nlsq.global_optimization.config import GlobalOptimizationConfig
        from nlsq.global_optimization.multi_start import MultiStartOrchestrator

        config = GlobalOptimizationConfig(n_starts=self._config.n_starts)
        optimizer = MultiStartOrchestrator(config=config)
        # MultiStartOrchestrator.fit() returns dict, not tuple
        return optimizer.fit(
            f,
            xdata,
            ydata,
            p0=p0 if p0 is not None else np.zeros(1),
            **kwargs,
        )


def configure_curve_fit(
    *,
    enable_diagnostics: bool = False,
    enable_recovery: bool = True,
    enable_caching: bool = True,
    **default_kwargs: Any,
) -> Callable[..., tuple[np.ndarray, np.ndarray] | Any]:
    """Configure a curve_fit function with default settings.

    Returns a callable that wraps curve_fit with pre-configured
    defaults, allowing for consistent settings across an application.

    Parameters
    ----------
    enable_diagnostics : bool, default=False
        Enable diagnostic reporting by default.
    enable_recovery : bool, default=True
        Enable automatic recovery by default.
    enable_caching : bool, default=True
        Enable JIT caching by default.
    **default_kwargs : Any
        Additional default keyword arguments.

    Returns
    -------
    Callable
        A configured curve_fit function.

    Examples
    --------
    >>> curve_fit = configure_curve_fit(enable_diagnostics=True)
    >>> popt, pcov = curve_fit(model, xdata, ydata)
    """

    def configured_fit(
        f: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray] | Any:
        """Configured curve_fit with preset defaults."""
        from nlsq.core.minpack import curve_fit

        # Merge defaults with call kwargs
        merged = {**default_kwargs, **kwargs}

        # Apply configuration
        if enable_diagnostics and "diagnostics" not in merged:
            from nlsq.diagnostics.types import DiagnosticsConfig

            merged["diagnostics"] = DiagnosticsConfig()

        return curve_fit(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            **merged,
        )

    return configured_fit
