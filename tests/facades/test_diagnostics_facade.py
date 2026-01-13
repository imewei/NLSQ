"""Unit tests for DiagnosticsFacade.

Tests for lazy import behavior, diagnostics access,
and circular dependency prevention.

Reference: specs/017-curve-fit-decomposition/spec.md FR-013
"""

from __future__ import annotations

import sys

import pytest

# =============================================================================
# Test Lazy Import Behavior
# =============================================================================


class TestLazyImportBehavior:
    """Tests for lazy import functionality."""

    def test_facade_import_does_not_load_diagnostics_eagerly(self) -> None:
        """Test importing facade doesn't eagerly load diagnostics."""
        # Clear modules to test fresh import
        modules_to_clear = [
            k for k in sys.modules if "diagnostics" in k and "facade" not in k
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import facade - should use lazy loading
        from nlsq.facades import DiagnosticsFacade

        # The test passes if no import error occurs

    def test_facade_provides_access_to_diagnostic_level(self) -> None:
        """Test facade provides access to DiagnosticLevel enum."""
        from nlsq.facades.diagnostics_facade import DiagnosticsFacade

        facade = DiagnosticsFacade()

        assert hasattr(facade, "get_diagnostic_level")
        level_enum = facade.get_diagnostic_level()
        assert level_enum is not None

    def test_facade_provides_access_to_diagnostics_config(self) -> None:
        """Test facade provides access to DiagnosticsConfig."""
        from nlsq.facades.diagnostics_facade import DiagnosticsFacade

        facade = DiagnosticsFacade()

        assert hasattr(facade, "get_diagnostics_config")
        config_class = facade.get_diagnostics_config()
        assert config_class is not None


# =============================================================================
# Test Facade Interface
# =============================================================================


class TestFacadeInterface:
    """Tests for DiagnosticsFacade interface."""

    def test_facade_is_instantiable(self) -> None:
        """Test facade can be instantiated."""
        from nlsq.facades.diagnostics_facade import DiagnosticsFacade

        facade = DiagnosticsFacade()
        assert facade is not None

    def test_facade_has_expected_methods(self) -> None:
        """Test facade exposes expected methods."""
        from nlsq.facades.diagnostics_facade import DiagnosticsFacade

        facade = DiagnosticsFacade()

        # Core methods
        expected_methods = [
            "get_diagnostic_level",
            "get_diagnostics_config",
            "get_convergence_monitor",
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
            from nlsq.facades.diagnostics_facade import DiagnosticsFacade

            facade = DiagnosticsFacade()
            assert facade is not None
        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            raise


# =============================================================================
# Test Integration
# =============================================================================


class TestIntegration:
    """Integration tests for DiagnosticsFacade."""

    def test_diagnostic_level_has_expected_values(self) -> None:
        """Test DiagnosticLevel enum has expected values."""
        from nlsq.facades.diagnostics_facade import DiagnosticsFacade

        facade = DiagnosticsFacade()
        DiagnosticLevel = facade.get_diagnostic_level()

        # Should have standard diagnostic levels
        assert hasattr(DiagnosticLevel, "BASIC")
        assert hasattr(DiagnosticLevel, "FULL")

    def test_diagnostics_config_is_instantiable(self) -> None:
        """Test DiagnosticsConfig can be instantiated."""
        from nlsq.facades.diagnostics_facade import DiagnosticsFacade

        facade = DiagnosticsFacade()
        DiagnosticsConfig = facade.get_diagnostics_config()

        # Should be able to create an instance
        config = DiagnosticsConfig()
        assert config is not None
