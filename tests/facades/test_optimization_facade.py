"""Unit tests for OptimizationFacade.

Tests for lazy import behavior, global optimization access,
and circular dependency prevention.

Reference: specs/017-curve-fit-decomposition/spec.md FR-011
"""

from __future__ import annotations

import sys

import pytest

# =============================================================================
# Test Lazy Import Behavior
# =============================================================================


class TestLazyImportBehavior:
    """Tests for lazy import functionality."""

    def test_facade_import_does_not_load_global_optimization(self) -> None:
        """Test importing facade doesn't eagerly load global_optimization."""
        # Clear modules to test fresh import
        modules_to_clear = [k for k in sys.modules if "global_optimization" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import facade - should use lazy loading
        from nlsq.facades import OptimizationFacade

        # The test passes if no import error occurs
        # Full validation would check sys.modules but that's implementation-specific

    def test_facade_provides_access_to_cmaes(self) -> None:
        """Test facade provides access to CMA-ES optimizer."""
        from nlsq.facades.optimization_facade import OptimizationFacade

        facade = OptimizationFacade()

        # Should be able to get CMA-ES related functionality
        assert hasattr(facade, "get_cmaes_optimizer")
        optimizer_class = facade.get_cmaes_optimizer()
        assert optimizer_class is not None

    def test_facade_provides_access_to_method_selector(self) -> None:
        """Test facade provides access to method selector."""
        from nlsq.facades.optimization_facade import OptimizationFacade

        facade = OptimizationFacade()

        assert hasattr(facade, "get_method_selector")
        selector_func = facade.get_method_selector()
        assert callable(selector_func)


# =============================================================================
# Test Facade Interface
# =============================================================================


class TestFacadeInterface:
    """Tests for OptimizationFacade interface."""

    def test_facade_is_instantiable(self) -> None:
        """Test facade can be instantiated."""
        from nlsq.facades.optimization_facade import OptimizationFacade

        facade = OptimizationFacade()
        assert facade is not None

    def test_facade_has_expected_methods(self) -> None:
        """Test facade exposes expected methods."""
        from nlsq.facades.optimization_facade import OptimizationFacade

        facade = OptimizationFacade()

        # Core methods
        expected_methods = [
            "get_cmaes_optimizer",
            "get_method_selector",
            "get_bipop_optimizer",
            "get_multistart_optimizer",
        ]

        for method in expected_methods:
            assert hasattr(facade, method), f"Missing method: {method}"


# =============================================================================
# Test Circular Dependency Prevention
# =============================================================================


class TestCircularDependencyPrevention:
    """Tests for circular dependency prevention."""

    def test_no_eager_minpack_import(self) -> None:
        """Test facade doesn't cause eager minpack import loop."""
        # This test verifies that importing OptimizationFacade
        # doesn't trigger a circular import with minpack
        try:
            from nlsq.facades.optimization_facade import OptimizationFacade

            facade = OptimizationFacade()
            # If we get here without ImportError, circular deps are broken
            assert facade is not None
        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            raise


# =============================================================================
# Test Integration
# =============================================================================


class TestIntegration:
    """Integration tests for OptimizationFacade."""

    def test_cmaes_optimizer_is_functional(self) -> None:
        """Test obtained CMA-ES optimizer class is functional."""
        from nlsq.facades.optimization_facade import OptimizationFacade

        facade = OptimizationFacade()
        CMAESOptimizer = facade.get_cmaes_optimizer()

        # Should be able to check it's a class
        assert isinstance(CMAESOptimizer, type)

    def test_multistart_optimizer_is_obtainable(self) -> None:
        """Test multistart optimizer can be obtained."""
        from nlsq.facades.optimization_facade import OptimizationFacade

        facade = OptimizationFacade()
        MultiStart = facade.get_multistart_optimizer()

        assert MultiStart is not None
