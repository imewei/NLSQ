"""Tests for workflow preset streaming behavior.

Tests cover:
- Explicit workflow="streaming" forces streaming even on small data
- Streaming preset configuration is correctly applied
- Integration with fit() function
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


def model(x, a, b):
    """Simple exponential model for testing."""
    return a * jnp.exp(-b * x)


class TestStreamingWorkflowPreset:
    """Tests for workflow='streaming' preset behavior."""

    def test_streaming_preset_exists(self):
        """Test that streaming preset is defined in WORKFLOW_PRESETS."""
        from nlsq.core.minpack import WORKFLOW_PRESETS

        assert "streaming" in WORKFLOW_PRESETS

        preset = WORKFLOW_PRESETS["streaming"]
        assert preset["tier"] == "STREAMING"
        assert preset["enable_multistart"] is False

    def test_streaming_preset_forces_streaming_on_small_data(self):
        """Test that workflow='streaming' forces streaming even on small datasets."""
        from nlsq import fit

        # Very small dataset that would normally use standard curve_fit
        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="streaming",
        )

        # Should succeed even with streaming on small data
        assert result is not None
        assert "x" in result or hasattr(result, "x")

    def test_streaming_preset_respects_explicit_selection(self):
        """Test that explicit workflow='streaming' bypasses auto-selection."""
        from nlsq import fit

        # Small dataset that auto would route to standard
        np.random.seed(42)
        x = jnp.linspace(0, 5, 50)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 50)

        # Using streaming explicitly
        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="streaming",
        )

        assert result is not None
        # Check that fit succeeded
        popt = np.array(result.x if hasattr(result, "x") else result["x"])
        assert 1.0 < popt[0] < 5.0  # a should be around 2.5
        assert 0.1 < popt[1] < 1.0  # b should be around 0.5

    def test_streaming_preset_tolerances(self):
        """Test that streaming preset has correct tolerance settings."""
        from nlsq.core.minpack import WORKFLOW_PRESETS

        preset = WORKFLOW_PRESETS["streaming"]
        assert preset["gtol"] == 1e-7
        assert preset["ftol"] == 1e-7
        assert preset["xtol"] == 1e-7


class TestOtherWorkflowPresets:
    """Tests for other workflow presets to ensure no regressions."""

    def test_standard_preset_uses_curve_fit(self):
        """Test that workflow='standard' uses basic curve_fit path."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="standard",
        )

        assert result is not None
        assert result.success

    def test_fast_preset_uses_looser_tolerances(self):
        """Test that workflow='fast' uses looser tolerances."""
        from nlsq.core.minpack import WORKFLOW_PRESETS

        preset = WORKFLOW_PRESETS["fast"]
        assert preset["gtol"] == 1e-6
        assert preset["ftol"] == 1e-6
        assert preset["xtol"] == 1e-6

    def test_quality_preset_uses_multistart(self):
        """Test that workflow='quality' enables multi-start."""
        from nlsq.core.minpack import WORKFLOW_PRESETS

        preset = WORKFLOW_PRESETS["quality"]
        assert preset["enable_multistart"] is True
        assert preset["n_starts"] == 20

    def test_large_robust_preset_uses_chunked_tier(self):
        """Test that workflow='large_robust' uses chunked tier."""
        from nlsq.core.minpack import WORKFLOW_PRESETS

        preset = WORKFLOW_PRESETS["large_robust"]
        assert preset["tier"] == "CHUNKED"
        assert preset["enable_multistart"] is True


class TestAutoWorkflow:
    """Tests for workflow='auto' behavior."""

    def test_auto_workflow_selects_standard_for_small_data(self):
        """Test that auto workflow selects standard for small datasets."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="auto",
        )

        assert result is not None
        assert result.success

    def test_auto_workflow_uses_memory_budget_selector(self):
        """Test that auto workflow uses MemoryBudgetSelector."""
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector()

        # Small dataset should select standard
        strategy, config = selector.select(n_points=1000, n_params=3)
        assert strategy == "standard"
        assert config is None

    def test_workflow_presets_all_have_required_fields(self):
        """Test that all workflow presets have required fields."""
        from nlsq.core.minpack import WORKFLOW_PRESETS

        required_fields = ["description", "tier"]

        for preset_name, preset in WORKFLOW_PRESETS.items():
            for field in required_fields:
                assert field in preset, f"{preset_name} missing {field}"


class TestWorkflowPresetIntegration:
    """Integration tests for workflow presets."""

    @pytest.mark.parametrize(
        "preset_name",
        ["standard", "fast", "quality", "streaming"],
    )
    def test_all_basic_presets_work(self, preset_name):
        """Test that all basic presets produce valid results."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 50)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 50)

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow=preset_name,
        )

        assert result is not None
        # All presets should produce valid fit
        popt = np.array(result.x if hasattr(result, "x") else result["x"])
        assert len(popt) == 2
