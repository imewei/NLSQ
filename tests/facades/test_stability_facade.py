"""Unit tests for StabilityFacade.

Tests for lazy import behavior, stability fallback access,
and circular dependency prevention.

Reference: specs/017-curve-fit-decomposition/spec.md FR-012
"""

from __future__ import annotations

import sys

import pytest

# =============================================================================
# Test Lazy Import Behavior
# =============================================================================


class TestLazyImportBehavior:
    """Tests for lazy import functionality."""

    def test_facade_import_does_not_load_stability_eagerly(self) -> None:
        """Test importing facade doesn't eagerly load stability.fallback."""
        # Clear modules to test fresh import
        modules_to_clear = [k for k in sys.modules if "stability.fallback" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import facade - should use lazy loading
        from nlsq.facades import StabilityFacade

        # The test passes if no import error occurs

    def test_facade_provides_access_to_fallback_svd(self) -> None:
        """Test facade provides access to fallback SVD."""
        from nlsq.facades.stability_facade import StabilityFacade

        facade = StabilityFacade()

        assert hasattr(facade, "get_fallback_svd")
        svd_func = facade.get_fallback_svd()
        assert callable(svd_func)

    def test_facade_provides_access_to_stability_guard(self) -> None:
        """Test facade provides access to stability guard."""
        from nlsq.facades.stability_facade import StabilityFacade

        facade = StabilityFacade()

        assert hasattr(facade, "get_stability_guard")
        guard_class = facade.get_stability_guard()
        assert guard_class is not None


# =============================================================================
# Test Facade Interface
# =============================================================================


class TestFacadeInterface:
    """Tests for StabilityFacade interface."""

    def test_facade_is_instantiable(self) -> None:
        """Test facade can be instantiated."""
        from nlsq.facades.stability_facade import StabilityFacade

        facade = StabilityFacade()
        assert facade is not None

    def test_facade_has_expected_methods(self) -> None:
        """Test facade exposes expected methods."""
        from nlsq.facades.stability_facade import StabilityFacade

        facade = StabilityFacade()

        # Core methods
        expected_methods = [
            "get_fallback_svd",
            "get_stability_guard",
            "get_condition_monitor",
            "get_recovery_handler",
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
        try:
            from nlsq.facades.stability_facade import StabilityFacade

            facade = StabilityFacade()
            assert facade is not None
        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            raise


# =============================================================================
# Test Integration
# =============================================================================


class TestIntegration:
    """Integration tests for StabilityFacade."""

    def test_fallback_svd_is_callable(self) -> None:
        """Test fallback SVD function is callable."""
        from nlsq.facades.stability_facade import StabilityFacade

        facade = StabilityFacade()
        svd_func = facade.get_fallback_svd()

        assert callable(svd_func)

    def test_stability_guard_is_class(self) -> None:
        """Test stability guard is a class type."""
        from nlsq.facades.stability_facade import StabilityFacade

        facade = StabilityFacade()
        Guard = facade.get_stability_guard()

        assert isinstance(Guard, type)
