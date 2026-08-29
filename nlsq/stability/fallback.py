"""
Automatic Fallback Strategies for Robust Optimization
======================================================

This module provides automatic fallback mechanisms that attempt to recover
from optimization failures by trying alternative approaches.

Key Features:
- Automatic method selection (trf → alternative approaches)
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

from collections.abc import Callable
from typing import Any, ClassVar, Protocol

import numpy as np

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
    """Try alternative optimization methods."""

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
            modified["method"] = self.method_sequence[self.current_index]
            self.current_index += 1
        return modified

    def reset(self) -> None:
        """Restart the method sequence from the beginning for a new fit."""
        self.current_index = 0


class PerturbInitialGuessStrategy(FallbackStrategy):
    """Perturb initial guess to escape local minima."""

    def __init__(self, perturbation_scale: float = 0.1, max_perturbations: int = 3):
        super().__init__(
            name="perturb_p0",
            description=f"Perturb initial guess by {perturbation_scale * 100:.0f}%",
            priority=8,
        )
        self.perturbation_scale = perturbation_scale
        self.max_perturbations = max_perturbations
        self.perturbation_count = 0

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Add random perturbation to p0."""
        modified = kwargs.copy()
        p0 = np.array(modified.get("p0", [1.0]))

        if self.perturbation_count < self.max_perturbations:
            # Add random noise scaled by parameter magnitude (additive for p0=0)
            noise = np.random.randn(*p0.shape) * self.perturbation_scale
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

    def reset(self) -> None:
        """Restart the perturbation budget from zero for a new fit."""
        self.perturbation_count = 0


class AdjustTolerancesStrategy(FallbackStrategy):
    """Relax optimization tolerances."""

    def __init__(self, relaxation_factor: float = 10.0):
        super().__init__(
            name="adjust_tolerances",
            description=f"Relax tolerances by {relaxation_factor}x",
            priority=7,
        )
        self.relaxation_factor = relaxation_factor
        self.current_factor = 1.0

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Relax ftol, xtol, gtol."""
        modified = kwargs.copy()

        # Default tolerances
        ftol = modified.get("ftol", 1e-8)
        xtol = modified.get("xtol", 1e-8)
        gtol = modified.get("gtol", 1e-8)

        # Relax by factor
        self.current_factor *= self.relaxation_factor
        modified["ftol"] = ftol * self.current_factor
        modified["xtol"] = xtol * self.current_factor
        modified["gtol"] = gtol * self.current_factor

        return modified

    def reset(self) -> None:
        """Restart the relaxation factor from 1.0 (unrelaxed) for a new fit."""
        self.current_factor = 1.0


class AddParameterBoundsStrategy(FallbackStrategy):
    """Infer and add parameter bounds if not provided."""

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

        # Heuristic bounds based on data range
        y_range = np.ptp(ydata)
        y_min, y_max = np.min(ydata), np.max(ydata)

        # Conservative bounds: p0 ± 10x, but reasonable for data scale
        lower = np.maximum(p0 / 10, y_min - y_range)
        upper = np.minimum(p0 * 10, y_max + y_range)

        # Ensure positive for amplitude-like parameters (heuristic)
        if np.all(p0 > 0):
            lower = np.maximum(lower, 0)

        modified["bounds"] = (lower, upper)

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
            modified["f_scale"] = modified.get("f_scale", 1.0)
            self.current_index += 1

        return modified

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
        return sigma_arr / (y_scale**2)
    return sigma_arr / y_scale


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
            Name of fallback strategy that succeeded
        attempts : int, optional
            Number of attempts before success
        """
        self.result = result
        self.fallback_strategy_used = strategy_used
        self.fallback_attempts = attempts

    def __getattr__(self, name):
        """Delegate attribute access to underlying result."""
        return getattr(self.result, name)


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
    >>> import numpy as np
    >>>
    >>> def model(x, a, b):
    ...     return a * np.exp(-b * x)
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

        # Sort by priority (highest first)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)

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

        Raises
        ------
        RuntimeError
            If all fallback strategies fail
        """
        from nlsq.core.minpack import (
            curve_fit,  # Deferred import to avoid circular dependency
        )

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

        try:
            result = curve_fit(f, xdata, ydata, **kwargs)
            if self.verbose:
                print("Success with original parameters!")
            return FallbackResult(result, strategy_used=None, attempts=1)
        except Exception as e:
            if self.verbose:
                print(f"[FAIL] Failed: {type(e).__name__}: {e}")
            last_exception = e

        # Try fallback strategies
        for strategy in self.strategies:
            if self.total_attempts >= self.max_attempts:
                break

            self.total_attempts += 1
            self._lifetime_attempts += 1
            strategy.attempts += 1

            if self.verbose:
                print(
                    f"\nAttempt {self.total_attempts}/{self.max_attempts}: "
                    f"{strategy.description}",
                )

            try:
                # Apply strategy modifications
                modified_kwargs = strategy.apply(kwargs)

                # Remove internal markers
                modified_kwargs.pop("_xdata", None)
                modified_kwargs.pop("_ydata", None)

                # RescaleProblemStrategy can't rewrite xdata/ydata/f itself
                # (its apply() only sees kwargs), so it signals a rescaled
                # fit via these internal keys instead. Build the actual
                # rescaled data/model here so popt comes back in original
                # units with no separate inverse-transform step.
                x_scale = modified_kwargs.pop("_x_scale", None)
                y_scale = modified_kwargs.pop("_y_scale", None)
                x_offset = modified_kwargs.pop("_x_offset", None)
                y_offset = modified_kwargs.pop("_y_offset", None)
                is_scaled = modified_kwargs.pop("_scaled", False)

                fit_f, fit_xdata, fit_ydata = f, xdata, ydata
                if is_scaled:
                    fit_xdata = (xdata - x_offset) / x_scale
                    fit_ydata = (ydata - y_offset) / y_scale

                    def fit_f(
                        x_scaled,
                        *params,
                        _f=f,
                        _xs=x_scale,
                        _xo=x_offset,
                        _yo=y_offset,
                        _ys=y_scale,
                    ):
                        return (_f(x_scaled * _xs + _xo, *params) - _yo) / _ys

                    if modified_kwargs.get("sigma") is not None:
                        modified_kwargs["sigma"] = _rescale_sigma(
                            modified_kwargs["sigma"], y_scale
                        )

                # Try fit with modified parameters
                result = curve_fit(fit_f, fit_xdata, fit_ydata, **modified_kwargs)

                # Success!
                strategy.successes += 1
                self.successful_strategies[strategy.name] = (
                    self.successful_strategies.get(strategy.name, 0) + 1
                )

                if self.verbose:
                    print(f"[OK] Success with {strategy.name}!")

                return FallbackResult(
                    result,
                    strategy_used=strategy.name,
                    attempts=self.total_attempts,
                )

            except Exception as e:
                if self.verbose:
                    print(f"[FAIL] Failed: {type(e).__name__}")
                last_exception = e
                continue

        # All strategies failed
        error_msg = (
            f"All {self.total_attempts} fallback attempts failed. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )
        self.logger.error(error_msg)
        raise RuntimeError(error_msg) from last_exception

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
