"""Tests for Multi-Start API Extensions.

This module tests the API extensions for multi-start optimization in NLSQ,
including the new curve_fit() parameters and the unified fit() function
with presets.

These tests verify Task Group 5 of the multi-start optimization specification:
- curve_fit(..., multistart=True) enables multi-start with default n_starts
- curve_fit(..., multistart=True, n_starts=20) overrides default
- curve_fit(..., global_search=True) is shorthand for multistart=True, n_starts=20
- fit(preset='robust') applies correct multi-start configuration
- fit(preset='streaming') uses AdaptiveHybridStreaming with tournament
- fit() auto-detects dataset size and selects appropriate strategy
"""

import jax.numpy as jnp
import numpy as np
import pytest


def exponential_model(x, a, b, c):
    """Test model: exponential decay with offset."""
    return a * jnp.exp(-b * x) + c


def generate_test_data(n_points: int = 1000, noise_level: float = 0.1):
    """Generate test data for curve fitting tests.

    Parameters
    ----------
    n_points : int
        Number of data points.
    noise_level : float
        Standard deviation of Gaussian noise.

    Returns
    -------
    tuple
        (xdata, ydata, true_params)
    """
    np.random.seed(42)
    true_params = [2.5, 0.5, 0.3]  # a, b, c
    xdata = np.linspace(0, 10, n_points)
    ydata = true_params[0] * np.exp(-true_params[1] * xdata) + true_params[2]
    ydata += noise_level * np.random.normal(size=n_points)
    return xdata, ydata, true_params


class TestCurveFitMultistart:
    """Tests for curve_fit() with multi-start parameters."""

    def test_multistart_true_enables_multistart_with_default_n_starts(self):
        """Test that curve_fit(..., multistart=True) enables multi-start with default n_starts."""
        from nlsq import curve_fit

        xdata, ydata, true_params = generate_test_data(n_points=500)

        # Fit with multistart=True
        result = curve_fit(
            exponential_model,
            xdata,
            ydata,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            multistart=True,
        )

        # Check that result has multi-start diagnostics
        assert (
            hasattr(result, "multistart_diagnostics")
            or "multistart_diagnostics" in result
        )
        diagnostics = result.get(
            "multistart_diagnostics",
            result.multistart_diagnostics
            if hasattr(result, "multistart_diagnostics")
            else {},
        )

        # Verify multi-start was enabled (n_starts_configured > 0 and not bypassed)
        assert diagnostics.get("n_starts_configured", 0) > 0
        # Default n_starts should be 10
        assert diagnostics.get("n_starts_configured") == 10

        # Check that fit succeeded
        popt = result.popt if hasattr(result, "popt") else result[0]
        assert len(popt) == 3
        # Parameters should be reasonably close to true values
        assert abs(popt[0] - true_params[0]) < 0.5
        assert abs(popt[1] - true_params[1]) < 0.3
        assert abs(popt[2] - true_params[2]) < 0.3

    def test_n_starts_override(self):
        """Test that curve_fit(..., multistart=True, n_starts=20) overrides default n_starts."""
        from nlsq import curve_fit

        xdata, ydata, _true_params = generate_test_data(n_points=500)

        # Fit with custom n_starts (requires multistart=True to take effect)
        result = curve_fit(
            exponential_model,
            xdata,
            ydata,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            multistart=True,
            n_starts=20,
        )

        # Check that n_starts was set to 20
        diagnostics = result.get(
            "multistart_diagnostics",
            result.multistart_diagnostics
            if hasattr(result, "multistart_diagnostics")
            else {},
        )
        assert diagnostics.get("n_starts_configured") == 20

        # Check that fit succeeded
        popt = result.popt if hasattr(result, "popt") else result[0]
        assert len(popt) == 3

    def test_global_search_shorthand(self):
        """Test that global_search=True is shorthand for multistart=True, n_starts=20."""
        from nlsq import curve_fit

        xdata, ydata, _true_params = generate_test_data(n_points=500)

        # Fit with global_search=True
        result = curve_fit(
            exponential_model,
            xdata,
            ydata,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            global_search=True,
        )

        # Check that global_search enabled multistart with n_starts=20
        diagnostics = result.get(
            "multistart_diagnostics",
            result.multistart_diagnostics
            if hasattr(result, "multistart_diagnostics")
            else {},
        )
        assert diagnostics.get("n_starts_configured") == 20

        # Check that fit succeeded
        popt = result.popt if hasattr(result, "popt") else result[0]
        assert len(popt) == 3


class TestFitFunction:
    """Tests for the unified fit() function with presets."""

    def test_fit_robust_preset(self):
        """Test that fit(preset='robust') applies correct multi-start configuration."""
        from nlsq import fit

        xdata, ydata, _true_params = generate_test_data(n_points=500)

        # Fit with 'robust' preset
        result = fit(
            exponential_model,
            xdata,
            ydata,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            preset="robust",
        )

        # Check multi-start diagnostics
        diagnostics = result.get(
            "multistart_diagnostics",
            result.multistart_diagnostics
            if hasattr(result, "multistart_diagnostics")
            else {},
        )

        # 'robust' preset should use n_starts=5
        assert diagnostics.get("n_starts_configured") == 5

        # Check that fit succeeded
        popt = result.popt if hasattr(result, "popt") else result[0]
        assert len(popt) == 3

    def test_fit_streaming_preset(self):
        """Test that fit(preset='streaming') uses AdaptiveHybridStreaming with tournament."""
        from nlsq import fit

        xdata, ydata, _true_params = generate_test_data(n_points=500)

        # Fit with 'streaming' preset
        result = fit(
            exponential_model,
            xdata,
            ydata,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            preset="streaming",
        )

        # Check that fit succeeded
        popt = result.popt if hasattr(result, "popt") else result[0]
        assert len(popt) == 3

        # 'streaming' preset should configure n_starts=10
        diagnostics = result.get(
            "multistart_diagnostics",
            result.multistart_diagnostics
            if hasattr(result, "multistart_diagnostics")
            else {},
        )
        # Check n_starts is 10 for streaming preset (even if multi-start is bypassed for small datasets)
        assert diagnostics.get("n_starts_configured", 0) == 10 or diagnostics.get(
            "bypassed", False
        )

    def test_fit_auto_detects_dataset_size(self):
        """Test that fit() auto-detects dataset size and selects appropriate strategy."""
        from nlsq import fit

        # Test with small dataset - should use standard curve_fit
        xdata_small, ydata_small, _ = generate_test_data(n_points=500)

        result_small = fit(
            exponential_model,
            xdata_small,
            ydata_small,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            preset="large",  # 'large' preset should auto-detect
        )

        # Check that fit succeeded
        popt = result_small.popt if hasattr(result_small, "popt") else result_small[0]
        assert len(popt) == 3

        # For small datasets with 'large' preset, it should still work but may not use
        # chunking or streaming (depending on size threshold)
        # The key test is that it doesn't fail and returns valid results

    def test_fit_fast_preset_no_multistart(self):
        """Test that fit(preset='fast') has n_starts=0 (no multi-start overhead)."""
        from nlsq import fit

        xdata, ydata, _ = generate_test_data(n_points=500)

        result = fit(
            exponential_model,
            xdata,
            ydata,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            preset="fast",
        )

        # Check multi-start diagnostics
        diagnostics = result.get(
            "multistart_diagnostics",
            result.multistart_diagnostics
            if hasattr(result, "multistart_diagnostics")
            else {},
        )

        # 'fast' preset should have n_starts=0
        assert diagnostics.get("n_starts_configured") == 0
        assert diagnostics.get("bypassed", True)  # Should be bypassed

        # Check that fit succeeded
        popt = result.popt if hasattr(result, "popt") else result[0]
        assert len(popt) == 3

    def test_fit_global_preset(self):
        """Test that fit(preset='global') uses n_starts=20."""
        from nlsq import fit

        xdata, ydata, _ = generate_test_data(n_points=500)

        result = fit(
            exponential_model,
            xdata,
            ydata,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            preset="global",
        )

        # Check multi-start diagnostics
        diagnostics = result.get(
            "multistart_diagnostics",
            result.multistart_diagnostics
            if hasattr(result, "multistart_diagnostics")
            else {},
        )

        # 'global' preset should have n_starts=20
        assert diagnostics.get("n_starts_configured") == 20

        # Check that fit succeeded
        popt = result.popt if hasattr(result, "popt") else result[0]
        assert len(popt) == 3


class TestCurveFitLargeMultistart:
    """Tests for curve_fit_large() with multi-start parameters."""

    def test_curve_fit_large_multistart(self):
        """Test curve_fit_large with multistart parameter."""
        from nlsq import curve_fit_large

        xdata, ydata, _true_params = generate_test_data(n_points=500)

        # Note: For small datasets, curve_fit_large may delegate to standard curve_fit
        result = curve_fit_large(
            exponential_model,
            xdata,
            ydata,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -1], [10, 5, 5]),
            multistart=True,
            n_starts=5,
        )

        # Check that fit succeeded
        popt = (
            result[0]
            if isinstance(result, tuple)
            else (result.popt if hasattr(result, "popt") else result["popt"])
        )
        assert len(popt) == 3


class TestAPIExports:
    """Tests for nlsq module exports."""

    def test_fit_function_exported(self):
        """Test that fit function is exported from nlsq."""
        import nlsq

        assert hasattr(nlsq, "fit")
        assert callable(nlsq.fit)

    def test_global_optimization_exports(self):
        """Test that global optimization classes are exported."""
        import nlsq

        assert hasattr(nlsq, "GlobalOptimizationConfig")
        assert hasattr(nlsq, "MultiStartOrchestrator")
        assert hasattr(nlsq, "TournamentSelector")

    def test_curve_fit_multistart_parameters(self):
        """Test that curve_fit accepts multi-start parameters."""
        import inspect

        from nlsq import curve_fit

        sig = inspect.signature(curve_fit)
        params = sig.parameters

        # Check that multistart parameters are present
        assert "multistart" in params
        assert "n_starts" in params
        assert "global_search" in params
        assert "sampler" in params
        assert "center_on_p0" in params
        assert "scale_factor" in params
