"""
Tests for Stability Integration with curve_fit
===============================================

Tests the integration of stability checks into the curve_fit API.
"""

import logging

import numpy as np
import pytest

import jax.numpy as jnp
from nlsq import curve_fit


class TestStabilityCheckMode:
    """Test stability='check' mode (check but don't fix)."""

    def exponential_decay(self, x, a, b, c):
        """Exponential decay model."""
        return a * jnp.exp(-b * x) + c

    def test_check_mode_with_warnings(self, caplog):
        """Test that stability='check' mode warns about issues."""
        # Create ill-conditioned problem
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0
        p0 = [1e-6, 1e6]  # Large scale mismatch

        with caplog.at_level(logging.WARNING):
            result = curve_fit(
                lambda x, a, b: a * x + b, x, y, p0=p0, stability="check"
            )

        # Should have warnings logged
        assert len(caplog.records) > 0
        assert any("stability" in record.message.lower() for record in caplog.records)

    def test_check_mode_no_warnings_for_good_data(self, caplog):
        """Test that stability='check' mode doesn't warn for well-conditioned data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(100)
        p0 = [2.0, 1.0]

        with caplog.at_level(logging.WARNING):
            result = curve_fit(
                lambda x, a, b: a * x + b, x, y, p0=p0, stability="check"
            )

        # Should not have stability warnings
        stability_warnings = [
            r for r in caplog.records if "stability" in r.message.lower()
        ]
        assert len(stability_warnings) == 0


class TestStabilityAutoMode:
    """Test stability='auto' mode (check and fix)."""

    def test_auto_mode_fixes_ill_conditioned(self, caplog):
        """Test that stability='auto' mode fixes ill-conditioned data."""
        # Create ill-conditioned problem
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0
        p0 = [2.0, 1.0]

        with caplog.at_level(logging.INFO):
            result = curve_fit(
                lambda x, a, b: a * x + b, x, y, p0=p0, stability="auto"
            )

        # Should have info logs about applied fixes
        info_messages = [
            r.message for r in caplog.records if r.levelno == logging.INFO
        ]
        assert any("fix" in msg.lower() for msg in info_messages)

        # Should converge successfully
        assert result.x is not None
        assert result.success

    def test_auto_mode_fixes_nan_data(self):
        """Test that stability='auto' mode fixes NaN data."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        y[50] = np.nan  # Introduce NaN

        # Should succeed with auto fix
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability="auto"
        )

        assert result.x is not None
        assert result.success

    def test_auto_mode_fixes_inf_data(self):
        """Test that stability='auto' mode fixes Inf data."""
        x = np.linspace(0, 10, 100)
        x[10] = np.inf  # Introduce Inf
        y = 2.0 * np.where(np.isfinite(x), x, 5.0) + 1.0

        # Should succeed with auto fix
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability="auto"
        )

        assert result.x is not None

    def test_auto_mode_fixes_parameter_scales(self):
        """Test that stability='auto' mode fixes parameter scale mismatches."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        p0 = [1e-6, 1e6]  # Huge scale mismatch

        # Should succeed with auto fix
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=p0, stability="auto"
        )

        assert result.x is not None

    def test_auto_mode_preserves_good_data(self):
        """Test that stability='auto' mode doesn't break good data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)
        p0 = [2.5, 0.5, 1.0]

        def exponential_decay(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # Should work just as well with auto mode
        result = curve_fit(exponential_decay, x, y, p0=p0, stability="auto")

        assert result.x is not None
        assert abs(result.x[0] - 2.5) < 0.5
        assert abs(result.x[1] - 0.5) < 0.2
        assert abs(result.x[2] - 1.0) < 0.3


class TestStabilityDisabled:
    """Test that stability=False works (default)."""

    def test_disabled_by_default(self):
        """Test that stability checks are disabled by default."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0

        # Should work without stability parameter
        result = curve_fit(lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0])

        assert result.x is not None

    def test_explicit_false(self):
        """Test that stability=False explicitly disables checks."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0

        # Should work with explicit False
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability=False
        )

        assert result.x is not None


class TestStabilityCombinedFeatures:
    """Test stability combined with other features."""

    def exponential_decay(self, x, a, b, c):
        """Exponential decay model."""
        return a * jnp.exp(-b * x) + c

    def test_stability_with_auto_bounds(self):
        """Test stability='auto' combined with auto_bounds."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 100.0 * np.exp(-0.01 * x) + 0.01 + 0.1 * np.random.randn(100)

        # Both stability and auto_bounds enabled
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[1, 1, 1],  # Poor initial guess
            auto_bounds=True,
            stability="auto",
        )

        assert result.x is not None
        assert result.x[0] > 10  # Large amplitude
        assert result.x[1] > 0  # Positive decay rate

    def test_stability_with_fallback(self):
        """Test stability='auto' combined with fallback."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)

        # Both stability and fallback enabled
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[10, 5, 5],  # Poor initial guess
            stability="auto",
            fallback=True,
        )

        assert result.x is not None

    def test_all_features_combined(self):
        """Test stability + auto_bounds + fallback together."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 100.0 * np.exp(-0.01 * x) + 0.01 + 0.1 * np.random.randn(100)

        # All advanced features enabled
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[1, 1, 1],  # Poor initial guess
            auto_bounds=True,
            stability="auto",
            fallback=True,
        )

        assert result.x is not None
        assert result.success


class TestEdgeCases:
    """Test edge cases for stability integration."""

    def test_stability_without_p0(self):
        """Test stability checks when p0 is not provided."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0

        # Should work even without p0
        # (p0 will be auto-estimated by curve_fit)
        result = curve_fit(lambda x, a, b: a * x + b, x, y, stability="auto")

        assert result.x is not None

    def test_invalid_stability_value(self):
        """Test that invalid stability values are handled."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0

        # Invalid value should be treated as False (no checks)
        # This should just work without error
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability="invalid"
        )

        assert result.x is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
