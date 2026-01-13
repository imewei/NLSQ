"""Stability facade for breaking circular dependencies.

This facade provides lazy access to stability components,
breaking the circular import cycle between minpack and stability.fallback.

Reference: specs/017-curve-fit-decomposition/spec.md FR-012
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax.numpy as jnp

    from nlsq.stability.fallback import FallbackOrchestrator
    from nlsq.stability.guard import NumericalStabilityGuard
    from nlsq.stability.recovery import OptimizationRecovery


class StabilityFacade:
    """Facade for stability components with lazy loading.

    This facade breaks the circular dependency between minpack.py and
    stability.fallback by deferring all imports to method call time.

    Examples
    --------
    >>> facade = StabilityFacade()
    >>> svd_func = facade.get_fallback_svd()
    >>> U, s, V = svd_func(jacobian_matrix)
    """

    def get_fallback_svd(
        self,
    ) -> Callable[[jnp.ndarray, bool], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Get the SVD function with GPU/CPU fallback.

        Returns
        -------
        Callable
            Function that computes SVD with automatic fallback:
            compute_svd_with_fallback(J_h, full_matrices=False) -> (U, s, V)
        """
        from nlsq.stability.svd_fallback import compute_svd_with_fallback

        return compute_svd_with_fallback

    def get_stability_guard(self) -> type[NumericalStabilityGuard]:
        """Get the NumericalStabilityGuard class.

        Returns
        -------
        type[NumericalStabilityGuard]
            The NumericalStabilityGuard class for detecting numerical issues.
        """
        from nlsq.stability.guard import NumericalStabilityGuard

        return NumericalStabilityGuard

    def get_condition_monitor(
        self,
    ) -> Callable[[Any], float]:
        """Get the condition number estimation function.

        Returns
        -------
        Callable
            Function to estimate condition number of a matrix.
        """
        from nlsq.stability.guard import estimate_condition_number

        return estimate_condition_number

    def get_recovery_handler(self) -> type[OptimizationRecovery]:
        """Get the OptimizationRecovery class.

        Returns
        -------
        type[OptimizationRecovery]
            The OptimizationRecovery class for recovering from failures.
        """
        from nlsq.stability.recovery import OptimizationRecovery

        return OptimizationRecovery

    def get_fallback_orchestrator(self) -> type[FallbackOrchestrator]:
        """Get the FallbackOrchestrator class.

        Returns
        -------
        type[FallbackOrchestrator]
            The FallbackOrchestrator class for managing fallback strategies.
        """
        from nlsq.stability.fallback import FallbackOrchestrator

        return FallbackOrchestrator

    def create_stability_guard(self, **kwargs: Any) -> NumericalStabilityGuard:
        """Create a NumericalStabilityGuard instance.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to NumericalStabilityGuard constructor.

        Returns
        -------
        NumericalStabilityGuard
            A configured NumericalStabilityGuard instance.
        """
        NumericalStabilityGuard = self.get_stability_guard()
        return NumericalStabilityGuard(**kwargs)

    def create_recovery_handler(self, **kwargs: Any) -> OptimizationRecovery:
        """Create an OptimizationRecovery instance.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to OptimizationRecovery constructor.

        Returns
        -------
        OptimizationRecovery
            A configured OptimizationRecovery instance.
        """
        OptimizationRecovery = self.get_recovery_handler()
        return OptimizationRecovery(**kwargs)
