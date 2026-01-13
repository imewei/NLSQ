"""Optimization facade for breaking circular dependencies.

This facade provides lazy access to global_optimization components,
breaking the circular import cycle between minpack and global_optimization.

Reference: specs/017-curve-fit-decomposition/spec.md FR-011
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlsq.global_optimization.bipop import BIPOPRestarter
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer
    from nlsq.global_optimization.method_selector import MethodSelector
    from nlsq.global_optimization.multi_start import MultiStartOrchestrator


class OptimizationFacade:
    """Facade for global optimization components with lazy loading.

    This facade breaks the circular dependency between minpack.py and
    global_optimization by deferring all imports to method call time.

    Examples
    --------
    >>> facade = OptimizationFacade()
    >>> CMAESOptimizer = facade.get_cmaes_optimizer()
    >>> optimizer = CMAESOptimizer(bounds=([0, 0], [10, 10]))
    """

    def get_cmaes_optimizer(self) -> type[CMAESOptimizer]:
        """Get the CMAESOptimizer class.

        Returns
        -------
        type[CMAESOptimizer]
            The CMAESOptimizer class for creating optimizer instances.
        """
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        return CMAESOptimizer

    def get_method_selector(self) -> type[MethodSelector]:
        """Get the MethodSelector class.

        Returns
        -------
        type[MethodSelector]
            The MethodSelector class for automatic method selection.
        """
        from nlsq.global_optimization.method_selector import MethodSelector

        return MethodSelector

    def get_bipop_optimizer(self) -> type[BIPOPRestarter]:
        """Get the BIPOPRestarter class.

        Returns
        -------
        type[BIPOPRestarter]
            The BIPOPRestarter class for BIPOP restart strategy.
        """
        from nlsq.global_optimization.bipop import BIPOPRestarter

        return BIPOPRestarter

    def get_multistart_optimizer(self) -> type[MultiStartOrchestrator]:
        """Get the MultiStartOrchestrator class.

        Returns
        -------
        type[MultiStartOrchestrator]
            The MultiStartOrchestrator class for multi-start optimization.
        """
        from nlsq.global_optimization.multi_start import MultiStartOrchestrator

        return MultiStartOrchestrator

    def create_cmaes_optimizer(self, **kwargs: Any) -> CMAESOptimizer:
        """Create a CMAESOptimizer instance with given parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to CMAESOptimizer constructor.

        Returns
        -------
        CMAESOptimizer
            A configured CMAESOptimizer instance.
        """
        CMAESOptimizer = self.get_cmaes_optimizer()
        return CMAESOptimizer(**kwargs)

    def create_multistart_orchestrator(self, **kwargs: Any) -> MultiStartOrchestrator:
        """Create a MultiStartOrchestrator instance with given parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to MultiStartOrchestrator constructor.

        Returns
        -------
        MultiStartOrchestrator
            A configured MultiStartOrchestrator instance.
        """
        MultiStartOrchestrator = self.get_multistart_optimizer()
        return MultiStartOrchestrator(**kwargs)
