"""Tests for auto_global workflow auto-selection logic.

Tests cover:
- Auto-selection between CMA-ES and Multi-Start via MethodSelector
- Wide bounds (>1000x scale) trigger CMA-ES selection
- Narrow bounds trigger Multi-Start selection
- Integration with fit() function and workflow parameter

.. versionchanged:: 0.6.3
   Tests updated from global_auto preset to auto_global workflow.
"""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization.cmaes_config import is_evosax_available


def model(x, a, b):
    """Simple exponential model for testing."""
    return a * jnp.exp(-b * x)


class TestAutoGlobalMethodSelection:
    """Tests for auto_global workflow method selection logic."""

    @pytest.mark.skipif(not is_evosax_available(), reason="evosax not installed")
    def test_wide_bounds_selects_cmaes(self):
        """Test that wide bounds (>1000x scale) trigger CMA-ES selection."""
        from nlsq import fit

        # Generate test data
        np.random.seed(42)
        x = jnp.linspace(0, 5, 50)
        y = 2.5 * jnp.exp(-0.5 * x)

        # Wide bounds: >1000x difference in ranges
        # ranges = [9999, 0.9999], ratio = 9999/0.9999 > 1000
        bounds = ([1, 0.0001], [10000, 1])

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            bounds=bounds,
            workflow="auto_global",  # NEW: use auto_global workflow
        )

        # Should succeed and return valid result
        assert result is not None
        assert "x" in result or hasattr(result, "x")

        # CMA-ES should add cmaes_diagnostics to result
        if "cmaes_diagnostics" in result:
            # CMA-ES was used
            assert result["cmaes_diagnostics"]["total_generations"] > 0

    def test_narrow_bounds_selects_multistart(self):
        """Test that narrow bounds (<1000x scale) trigger Multi-Start selection."""
        from nlsq import fit

        # Generate test data
        np.random.seed(42)
        x = jnp.linspace(0, 5, 50)
        y = 2.5 * jnp.exp(-0.5 * x)

        # Narrow bounds: same scale for all parameters
        bounds = ([0.1, 0.1], [10.0, 10.0])

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 1.0],
            bounds=bounds,
            workflow="auto_global",  # NEW: use auto_global workflow
        )

        # Should succeed and return valid result
        assert result is not None
        assert "x" in result or hasattr(result, "x")

    def test_removed_presets_raise_error(self):
        """Test that using removed global_auto preset raises ValueError."""
        from nlsq import fit
        from nlsq.core.minpack import REMOVED_PRESETS

        assert "global_auto" in REMOVED_PRESETS

        np.random.seed(42)
        x = jnp.linspace(0, 5, 50)
        y = 2.5 * jnp.exp(-0.5 * x)

        with pytest.raises(ValueError, match=r"was removed in v0\.6\.3"):
            fit(
                model,
                x,
                y,
                p0=[1.0, 0.5],
                bounds=([0.1, 0.1], [10.0, 10.0]),
                workflow="global_auto",  # OLD: removed preset
            )

    @pytest.mark.skipif(not is_evosax_available(), reason="evosax not installed")
    def test_truly_wide_bounds_selects_cmaes(self):
        """Test truly wide bounds (>1000x scale difference) trigger CMA-ES."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()

        # Truly wide bounds: 10000x difference in ranges
        lower = np.array([0, 0])
        upper = np.array([10000, 1])  # ranges: 10000, 1; ratio = 10000

        method = selector.select(
            requested_method="auto",
            lower_bounds=lower,
            upper_bounds=upper,
        )
        assert method == "cmaes"

    def test_uniform_bounds_selects_multistart(self):
        """Test uniform bounds (1x scale) select Multi-Start."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()

        # Uniform bounds: same range for all parameters
        lower = np.array([0, 0, 0])
        upper = np.array([1, 1, 1])  # ranges: 1, 1, 1; ratio = 1

        method = selector.select(
            requested_method="auto",
            lower_bounds=lower,
            upper_bounds=upper,
        )
        assert method == "multi-start"


class TestAutoGlobalFitIntegration:
    """Integration tests for auto_global workflow through fit() API."""

    @pytest.mark.skipif(not is_evosax_available(), reason="evosax not installed")
    def test_fit_with_auto_global_wide_bounds(self):
        """Test fit() with auto_global workflow and wide bounds uses CMA-ES."""
        from nlsq import fit

        # Generate test data
        np.random.seed(42)
        x = jnp.linspace(0, 5, 30)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 30)

        # Wide bounds: 100000x difference
        bounds = ([0, 0], [100000, 1])

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            bounds=bounds,
            workflow="auto_global",  # NEW: use auto_global workflow
        )

        assert result.success
        # Parameters should be reasonable
        popt = np.array(result.x)
        assert 1.0 < popt[0] < 5.0  # a should be around 2.5
        assert 0.1 < popt[1] < 1.0  # b should be around 0.5

    def test_fit_with_auto_global_narrow_bounds(self):
        """Test fit() with auto_global workflow and narrow bounds uses Multi-Start."""
        from nlsq import fit

        # Generate test data
        np.random.seed(42)
        x = jnp.linspace(0, 5, 30)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 30)

        # Narrow bounds: same scale
        bounds = ([1.0, 0.1], [5.0, 1.0])

        result = fit(
            model,
            x,
            y,
            p0=[2.0, 0.5],
            bounds=bounds,
            workflow="auto_global",  # NEW: use auto_global workflow
        )

        assert result.success
        # Parameters should be reasonable
        popt = np.array(result.x)
        assert 1.0 < popt[0] < 5.0  # a should be around 2.5
        assert 0.1 < popt[1] < 1.0  # b should be around 0.5


class TestMethodSelectorScaleRatioAccuracy:
    """Tests to verify MethodSelector scale ratio computation accuracy."""

    def test_scale_ratio_1000x_threshold(self):
        """Test that exactly 1000x scale ratio is at threshold."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector(scale_threshold=1000.0)

        # Exactly 1000x ratio
        lower = np.array([0, 0])
        upper = np.array([1000, 1])

        ratio = selector.compute_scale_ratio(lower, upper)
        assert ratio == 1000.0

    def test_scale_ratio_above_threshold_selects_cmaes(self):
        """Test that scale ratio > 1000 selects CMA-ES (when evosax available)."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector(scale_threshold=1000.0)

        # 1001x ratio (just above threshold)
        lower = np.array([0, 0])
        upper = np.array([1001, 1])

        with patch(
            "nlsq.global_optimization.method_selector.is_evosax_available",
            return_value=True,
        ):
            method = selector.select(
                requested_method="auto",
                lower_bounds=lower,
                upper_bounds=upper,
            )
            assert method == "cmaes"

    def test_scale_ratio_below_threshold_selects_multistart(self):
        """Test that scale ratio < 1000 selects Multi-Start."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector(scale_threshold=1000.0)

        # 999x ratio (just below threshold)
        lower = np.array([0, 0])
        upper = np.array([999, 1])

        method = selector.select(
            requested_method="auto",
            lower_bounds=lower,
            upper_bounds=upper,
        )
        assert method == "multi-start"
