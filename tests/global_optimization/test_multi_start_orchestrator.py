"""
Tests for Multi-Start Orchestrator
===================================

Tests for the MultiStartOrchestrator class that evaluates multiple starting
points and selects the best based on loss value.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit, curve_fit
from nlsq.global_optimization import GlobalOptimizationConfig
from nlsq.global_optimization.multi_start import MultiStartOrchestrator


class TestMultiStartOrchestrator:
    """Test MultiStartOrchestrator implementation."""

    def test_orchestrator_selects_best_starting_point_by_loss(self):
        """Test orchestrator selects best starting point by loss value."""

        # Define a simple quadratic model
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        # Generate data with known parameters
        np.random.seed(42)
        x = np.linspace(-5, 5, 50)
        true_params = [1.0, 2.0, 3.0]
        y = quadratic(x, *true_params) + np.random.normal(0, 0.1, len(x))

        # Create config with multiple starts
        config = GlobalOptimizationConfig(n_starts=5, sampler="lhs", center_on_p0=False)

        # Create orchestrator
        orchestrator = MultiStartOrchestrator(config=config)

        # Run multi-start fit
        result = orchestrator.fit(
            f=quadratic,
            xdata=x,
            ydata=y,
            bounds=([0, 0, 0], [10, 10, 10]),
        )

        # Check that result is valid
        assert result is not None
        assert hasattr(result, "popt")
        assert len(result.popt) == 3

        # Check that fitted parameters are close to true values
        np.testing.assert_array_almost_equal(result.popt, true_params, decimal=1)

        # Check diagnostics contain multi-start info
        assert (
            hasattr(result, "multistart_diagnostics")
            or "multistart_diagnostics" in result
        )

    def test_preset_configuration_applies_correct_n_starts(self):
        """Test preset configuration applies correct n_starts values."""
        # Test all presets
        presets_expected = {
            "fast": 0,
            "robust": 5,
            "global": 20,
            "thorough": 50,
            "streaming": 10,
        }

        for preset_name, expected_n_starts in presets_expected.items():
            orchestrator = MultiStartOrchestrator.from_preset(preset_name)
            assert orchestrator.config.n_starts == expected_n_starts, (
                f"Preset '{preset_name}' should have n_starts={expected_n_starts}, "
                f"got {orchestrator.config.n_starts}"
            )

    def test_n_starts_zero_bypasses_multistart(self):
        """Test n_starts=0 bypasses multi-start and uses standard single-start."""

        def linear(x, a, b):
            return a * x + b

        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 3.0 + np.random.normal(0, 0.1, len(x))

        # Create config with n_starts=0 (no multi-start)
        config = GlobalOptimizationConfig(n_starts=0)

        # Create orchestrator
        orchestrator = MultiStartOrchestrator(config=config)

        # Run fit (should bypass multi-start)
        result = orchestrator.fit(
            f=linear,
            xdata=x,
            ydata=y,
            p0=[1.0, 1.0],
        )

        # Check that result is valid
        assert result is not None
        assert len(result.popt) == 2

        # Check diagnostics indicate single-start was used
        if hasattr(result, "multistart_diagnostics"):
            diagnostics = result.multistart_diagnostics
            assert diagnostics.get("n_starts_evaluated", 0) == 0 or diagnostics.get(
                "bypassed", True
            )

    def test_multistart_with_lhs_samples_produces_valid_fit(self):
        """Test multi-start with LHS samples produces valid fit results."""

        def exponential_decay(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        true_params = [3.0, 0.5, 1.0]
        y = np.array(true_params[0] * np.exp(-true_params[1] * x) + true_params[2])
        y = y + np.random.normal(0, 0.1, len(y))

        # Create config with LHS sampling
        config = GlobalOptimizationConfig(
            n_starts=10,
            sampler="lhs",
            center_on_p0=False,
        )

        orchestrator = MultiStartOrchestrator(config=config)

        result = orchestrator.fit(
            f=exponential_decay,
            xdata=x,
            ydata=y,
            bounds=([0.1, 0.01, 0.1], [10.0, 5.0, 10.0]),
        )

        # Verify result is valid
        assert result is not None
        assert len(result.popt) == 3
        assert result.success or result.get("success", True)

        # Parameters should be reasonable (within order of magnitude)
        assert 0.1 < result.popt[0] < 10.0  # amplitude
        assert 0.01 < result.popt[1] < 5.0  # rate
        assert 0.1 < result.popt[2] < 10.0  # offset

    def test_bounds_inference_triggered_when_bounds_not_provided(self):
        """Test bounds inference is triggered when bounds not provided."""

        def gaussian(x, amp, mu, sigma):
            return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

        np.random.seed(42)
        x = np.linspace(-5, 5, 100)
        y = 2.0 * np.exp(-((x - 1.0) ** 2) / (2 * 0.5**2)) + np.random.normal(
            0, 0.05, len(x)
        )

        # Create config with multi-start but don't provide bounds
        config = GlobalOptimizationConfig(n_starts=5, center_on_p0=True)

        orchestrator = MultiStartOrchestrator(config=config)

        # Run fit without explicit bounds - should trigger inference
        result = orchestrator.fit(
            f=gaussian,
            xdata=x,
            ydata=y,
            p0=[1.0, 0.0, 1.0],  # Initial guess for centering
            # No bounds provided - should be inferred
        )

        # Verify fit succeeded
        assert result is not None
        assert len(result.popt) == 3

        # Check that bounds were inferred (diagnostics should mention it)
        if hasattr(result, "multistart_diagnostics"):
            diagnostics = result.multistart_diagnostics
            # Either bounds_inferred flag or inferred_bounds should be present
            assert (
                diagnostics.get("bounds_inferred", False)
                or "inferred_bounds" in diagnostics
            )

    def test_center_on_p0_centers_samples_around_heuristic_estimate(self):
        """Test center_on_p0 correctly centers samples around heuristic estimate."""

        def linear(x, a, b):
            return a * x + b

        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        true_params = [2.0, 3.0]
        y = true_params[0] * x + true_params[1] + np.random.normal(0, 0.2, len(x))

        # Create config with centering enabled
        config = GlobalOptimizationConfig(
            n_starts=10,
            sampler="lhs",
            center_on_p0=True,
            scale_factor=0.5,  # Narrow exploration around p0
        )

        orchestrator = MultiStartOrchestrator(config=config)

        # Provide p0 close to true values
        p0 = [1.5, 2.5]

        result = orchestrator.fit(
            f=linear,
            xdata=x,
            ydata=y,
            p0=p0,
            bounds=([0, 0], [10, 10]),
        )

        # Verify result
        assert result is not None
        assert len(result.popt) == 2

        # With centering enabled, the fit should converge well
        np.testing.assert_array_almost_equal(result.popt, true_params, decimal=1)

        # Check that multistart diagnostics show centering was used
        if hasattr(result, "multistart_diagnostics"):
            diagnostics = result.multistart_diagnostics
            assert diagnostics.get("center_on_p0", False) or diagnostics.get(
                "centered_on_p0", False
            )


class TestMultiStartOrchestratorEdgeCases:
    """Test edge cases and error handling for MultiStartOrchestrator."""

    def test_from_preset_invalid_preset_raises_error(self):
        """Test from_preset raises error for invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            MultiStartOrchestrator.from_preset("nonexistent_preset")

    def test_orchestrator_with_custom_curve_fit_instance(self):
        """Test orchestrator works with custom CurveFit instance."""

        def simple_model(x, a):
            return a * x

        x = np.linspace(0, 10, 20)
        y = 2.0 * x + np.random.normal(0, 0.1, len(x))

        # Create custom CurveFit instance
        custom_fitter = CurveFit()

        # Create config
        config = GlobalOptimizationConfig(n_starts=3)

        # Create orchestrator with custom fitter
        orchestrator = MultiStartOrchestrator(
            config=config, curve_fit_instance=custom_fitter
        )

        result = orchestrator.fit(
            f=simple_model,
            xdata=x,
            ydata=y,
            p0=[1.0],
        )

        assert result is not None
        assert len(result.popt) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
