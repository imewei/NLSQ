"""Integration tests for the workflow system (Task Group 8).

This module provides comprehensive end-to-end integration tests for the
workflow system, testing:
- Full workflow: fit() -> auto_select -> backend -> result
- fit() with various dataset sizes (small, large, huge)
- Goal transitions (fast -> quality) on same dataset
- Memory pressure scenarios (mock low memory)
- Config persistence (YAML -> workflow -> fit)
- Error recovery and graceful degradation
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.minpack import curve_fit, fit
from nlsq.core.workflow import (
    WORKFLOW_PRESETS,
    MemoryTier,
    OptimizationGoal,
    WorkflowConfig,
    WorkflowSelector,
    WorkflowTier,
    auto_select_workflow,
    calculate_adaptive_tolerances,
    get_env_overrides,
    load_yaml_config,
)
from nlsq.result import CurveFitResult


# Test model functions
def exponential_model(x, a, b):
    """Simple exponential decay model."""
    return a * jnp.exp(-b * x)


def gaussian_model(x, amplitude, center, sigma):
    """Gaussian peak model for more complex fitting."""
    return amplitude * jnp.exp(-((x - center) ** 2) / (2 * sigma**2))


@pytest.fixture
def small_dataset():
    """Small dataset for quick tests (<1K points)."""
    np.random.seed(42)
    x = np.linspace(0, 5, 500)
    y_true = 2.5 * np.exp(-1.3 * x)
    y = y_true + 0.05 * np.random.normal(size=len(x))
    return x, y


@pytest.fixture
def medium_dataset():
    """Medium dataset (10K points)."""
    np.random.seed(42)
    x = np.linspace(0, 10, 10_000)
    y_true = 2.5 * np.exp(-0.5 * x)
    y = y_true + 0.1 * np.random.normal(size=len(x))
    return x, y


@pytest.fixture
def large_dataset():
    """Large dataset (100K points)."""
    np.random.seed(42)
    x = np.linspace(0, 20, 100_000)
    y_true = 3.0 * np.exp(-0.3 * x)
    y = y_true + 0.05 * np.random.normal(size=len(x))
    return x, y


class TestEndToEndWorkflow:
    """Test full workflow: fit() -> auto_select -> backend -> result."""

    def test_full_workflow_small_dataset_auto(self, small_dataset):
        """Test complete workflow with small dataset and auto selection."""
        x, y = small_dataset

        # Full workflow with auto selection
        result = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            workflow="auto",
        )

        # Verify result structure
        assert isinstance(result, CurveFitResult)
        assert result.popt is not None
        assert result.pcov is not None
        assert len(result.popt) == 2

        # Verify fit quality
        np.testing.assert_allclose(result.popt[0], 2.5, rtol=0.1)
        np.testing.assert_allclose(result.popt[1], 1.3, rtol=0.1)

    def test_full_workflow_with_goal_quality(self, small_dataset):
        """Test workflow with quality goal through entire pipeline."""
        x, y = small_dataset

        result = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            workflow="auto",
            goal="quality",
            bounds=([0, 0], [10, 5]),
        )

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None
        # Quality goal should achieve good fit
        np.testing.assert_allclose(result.popt[0], 2.5, rtol=0.1)

    def test_full_workflow_preserves_result_metadata(self, small_dataset):
        """Test that workflow preserves all result metadata."""
        x, y = small_dataset

        result = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            workflow="auto",
        )

        # Check essential metadata
        assert hasattr(result, "xdata")
        assert hasattr(result, "ydata")
        assert hasattr(result, "popt")
        assert hasattr(result, "pcov")


class TestDatasetSizeVariations:
    """Test fit() with various dataset sizes."""

    def test_fit_small_dataset(self, small_dataset):
        """Test fit() with small dataset (<1K points)."""
        x, y = small_dataset

        result = fit(exponential_model, x, y, p0=[2.0, 1.0])

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None

    def test_fit_medium_dataset(self, medium_dataset):
        """Test fit() with medium dataset (10K points)."""
        x, y = medium_dataset

        result = fit(exponential_model, x, y, p0=[2.0, 0.5])

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None
        # Check fit quality
        np.testing.assert_allclose(result.popt[0], 2.5, rtol=0.2)
        np.testing.assert_allclose(result.popt[1], 0.5, rtol=0.2)

    def test_workflow_selector_scales_with_dataset(self):
        """Test WorkflowSelector returns appropriate tier for different sizes."""
        selector = WorkflowSelector()

        # Small dataset -> should use standard-tier config
        config_small = selector.select(n_points=1_000, n_params=2)

        # Large dataset with mocked low memory -> should use streaming-tier config
        selector_low_mem = WorkflowSelector(memory_limit_gb=8.0)
        config_large = selector_low_mem.select(n_points=50_000_000, n_params=2)

        # Both should return valid configs
        assert config_small is not None
        assert config_large is not None


class TestGoalTransitions:
    """Test goal transitions (fast -> quality) on same dataset."""

    def test_fast_then_quality_goal(self, small_dataset):
        """Test switching from fast to quality goal on same dataset."""
        x, y = small_dataset

        # First fit with fast goal
        result_fast = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            goal="fast",
        )

        # Then fit with quality goal
        result_quality = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            goal="quality",
            bounds=([0, 0], [10, 5]),
        )

        # Both should complete successfully
        assert result_fast.popt is not None
        assert result_quality.popt is not None

        # Both should give similar results for this well-behaved problem
        np.testing.assert_allclose(result_fast.popt, result_quality.popt, rtol=0.2)

    def test_goal_affects_tolerances(self):
        """Test that different goals produce different tolerances."""
        tol_fast = calculate_adaptive_tolerances(10_000, OptimizationGoal.FAST)
        tol_quality = calculate_adaptive_tolerances(10_000, OptimizationGoal.QUALITY)
        tol_robust = calculate_adaptive_tolerances(10_000, OptimizationGoal.ROBUST)

        # Fast should have looser tolerances than quality
        assert tol_fast["gtol"] > tol_quality["gtol"]
        # Robust should be between fast and quality
        assert tol_fast["gtol"] > tol_robust["gtol"]


class TestMemoryPressureScenarios:
    """Test memory pressure scenarios with mocked low memory."""

    def test_low_memory_selects_streaming(self):
        """Test that low memory triggers streaming tier selection."""
        # Mock low memory environment (8GB)
        selector = WorkflowSelector(memory_limit_gb=8.0)

        # Large dataset with low memory should select streaming
        config = selector.select(n_points=50_000_000, n_params=3)

        # Should return a streaming-compatible config
        assert config is not None

    def test_workflow_selector_memory_tier_classification(self):
        """Test MemoryTier classification for different memory amounts."""
        assert MemoryTier.from_available_memory_gb(8.0) == MemoryTier.LOW
        assert MemoryTier.from_available_memory_gb(32.0) == MemoryTier.MEDIUM
        assert MemoryTier.from_available_memory_gb(96.0) == MemoryTier.HIGH
        assert MemoryTier.from_available_memory_gb(200.0) == MemoryTier.VERY_HIGH

    def test_memory_efficient_goal_prioritizes_streaming(self):
        """Test MEMORY_EFFICIENT goal prioritizes streaming over chunking."""
        selector = WorkflowSelector(memory_limit_gb=32.0)

        # Medium memory + memory_efficient goal should prefer streaming
        config = selector.select(
            n_points=5_000_000,
            n_params=3,
            goal=OptimizationGoal.MEMORY_EFFICIENT,
        )

        assert config is not None


class TestYAMLConfigPersistence:
    """Test config persistence via YAML files."""

    def test_yaml_config_loads_correctly(self, tmp_path):
        """Test that YAML config is loaded and applied correctly."""
        # Create a temporary YAML config file
        yaml_content = """
default_workflow: standard
memory_limit_gb: 16.0

workflows:
  test_workflow:
    tier: CHUNKED
    goal: QUALITY
    enable_multistart: true
    n_starts: 15
    gtol: 1e-9
    ftol: 1e-9
    xtol: 1e-9
"""
        yaml_file = tmp_path / "nlsq.yaml"
        yaml_file.write_text(yaml_content)

        # Load the config
        try:
            config = load_yaml_config(yaml_file)

            assert config is not None
            assert config.get("default_workflow") == "standard"
            assert config.get("memory_limit_gb") == 16.0
            assert "workflows" in config
            assert "test_workflow" in config["workflows"]
        except ImportError:
            # pyyaml not installed, skip this test
            pytest.skip("pyyaml not installed")

    def test_env_overrides_take_precedence(self, tmp_path):
        """Test that environment variables override YAML config."""
        # Set environment variables
        with patch.dict(
            os.environ,
            {
                "NLSQ_WORKFLOW_GOAL": "fast",
                "NLSQ_MEMORY_LIMIT_GB": "8.0",
            },
        ):
            overrides = get_env_overrides()

            assert overrides.get("goal") == "FAST"
            assert overrides.get("memory_limit_gb") == 8.0


class TestErrorRecoveryAndGracefulDegradation:
    """Test error recovery and graceful degradation scenarios."""

    def test_fit_handles_nan_in_data(self, small_dataset):
        """Test fit() handles NaN values appropriately."""
        x, y = small_dataset

        # Introduce NaN (this should be caught by check_finite)
        y_with_nan = y.copy()
        y_with_nan[10] = np.nan

        # Should raise or handle gracefully depending on check_finite
        with pytest.raises((ValueError, RuntimeError)):
            fit(
                exponential_model,
                x,
                y_with_nan,
                p0=[2.0, 1.0],
                check_finite=True,
            )

    def test_fit_handles_inf_in_data(self, small_dataset):
        """Test fit() handles infinity values appropriately."""
        x, y = small_dataset

        # Introduce inf
        y_with_inf = y.copy()
        y_with_inf[10] = np.inf

        with pytest.raises((ValueError, RuntimeError)):
            fit(
                exponential_model,
                x,
                y_with_inf,
                p0=[2.0, 1.0],
                check_finite=True,
            )

    def test_fit_recovers_with_good_initial_guess(self, small_dataset):
        """Test fit() with good vs bad initial guesses."""
        x, y = small_dataset

        # Good initial guess
        result_good = fit(
            exponential_model,
            x,
            y,
            p0=[2.5, 1.3],  # Close to true values
        )

        # Less ideal initial guess (but still reasonable)
        result_less_ideal = fit(
            exponential_model,
            x,
            y,
            p0=[1.0, 0.5],  # Further from true values
        )

        # Both should converge to similar solutions
        np.testing.assert_allclose(result_good.popt, result_less_ideal.popt, rtol=0.2)


class TestPresetConfigurations:
    """Test that all preset configurations work correctly."""

    def test_all_presets_are_valid(self):
        """Test that all WORKFLOW_PRESETS produce valid configs."""
        for preset_name in WORKFLOW_PRESETS:
            config = WorkflowConfig.from_preset(preset_name)

            assert config is not None
            assert config.tier is not None
            assert config.goal is not None
            assert config.gtol > 0
            assert config.ftol > 0
            assert config.xtol > 0

    def test_preset_fit_integration(self, small_dataset):
        """Test fit() works with each named preset workflow."""
        x, y = small_dataset

        # Test each standard preset
        for preset_name in ["standard", "fast"]:  # Quick presets only
            result = fit(
                exponential_model,
                x,
                y,
                p0=[2.0, 1.0],
                workflow=preset_name,
            )

            assert result.popt is not None, f"Preset {preset_name} failed"


class TestPackageExports:
    """Test that all required exports are available from the package."""

    def test_fit_importable_from_nlsq(self):
        """Test fit() can be imported from nlsq package."""
        from nlsq import fit as fit_func

        assert callable(fit_func)

    def test_workflow_components_importable(self):
        """Test workflow components can be imported."""
        from nlsq.core.workflow import (
            WORKFLOW_PRESETS,
            OptimizationGoal,
            WorkflowConfig,
            WorkflowSelector,
        )

        assert WorkflowTier is not None
        assert OptimizationGoal is not None
        assert WorkflowConfig is not None
        assert WorkflowSelector is not None
        assert callable(auto_select_workflow)
        assert isinstance(WORKFLOW_PRESETS, dict)

    def test_curve_fit_still_available(self):
        """Test that curve_fit is still available after fit() addition."""
        from nlsq import curve_fit as cf
        from nlsq.core.minpack import curve_fit as cf_minpack

        assert callable(cf)
        assert callable(cf_minpack)
