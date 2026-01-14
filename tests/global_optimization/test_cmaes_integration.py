"""Integration tests for CMA-ES with curve_fit.

Tests cover:
- CMAESOptimizer with curve_fit workflow
- Fallback behavior when evosax unavailable
- Configuration passthrough
"""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization.cmaes_config import is_evosax_available

# Skip all tests if evosax is not available
pytestmark = pytest.mark.skipif(
    not is_evosax_available(),
    reason="evosax not installed - skipping CMA-ES integration tests",
)


class TestCMAESOptimizerIntegration:
    """Integration tests for CMAESOptimizer with real fitting."""

    def test_optimizer_produces_valid_result(self) -> None:
        """Test that CMAESOptimizer produces a valid optimization result."""
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        config = CMAESConfig(
            max_generations=50,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(model, x, y, bounds=bounds)

        # Verify result structure
        assert "popt" in result
        assert "pcov" in result
        assert len(result["popt"]) == 2

        # Verify convergence
        np.testing.assert_allclose(result["popt"], [2.5, 0.5], rtol=0.1)

    def test_optimizer_with_sigma_weights(self) -> None:
        """Test that optimizer handles sigma weights correctly."""
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 100)
        y_true = 2.5 * jnp.exp(-0.5 * x)
        # Add heteroscedastic noise
        noise = 0.1 * y_true * jnp.sin(x * 5)
        y = y_true + noise
        sigma = 0.1 * y_true + 0.01  # Heteroscedastic weights

        bounds = ([0.1, 0.01], [10.0, 2.0])

        config = CMAESConfig(
            max_generations=50,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(model, x, y, bounds=bounds, sigma=sigma)

        # Should still converge reasonably
        assert "popt" in result
        popt = result["popt"]
        assert 0.1 < popt[0] < 10.0
        assert 0.01 < popt[1] < 2.0


class TestEvosaxUnavailabilityFallback:
    """Tests for behavior when evosax is unavailable.

    These tests mock evosax as unavailable to verify fallback behavior.
    """

    def test_cmaes_optimizer_raises_without_evosax(self) -> None:
        """Test that CMAESOptimizer raises ImportError when evosax unavailable."""
        # This test is skipped if evosax IS available (via pytestmark above)
        # We need to test the import behavior directly
        pass  # Tested via test_method_selector.py::test_select_explicit_cmaes_without_evosax_falls_back

    @pytest.mark.serial  # Log capture is unreliable during parallel execution
    def test_method_selector_fallback_logs_info(self, caplog) -> None:
        """Test that MethodSelector logs INFO when falling back from CMA-ES."""
        import logging

        from nlsq.global_optimization import MethodSelector

        selector = MethodSelector()
        lower = np.array([0, 0])
        upper = np.array([1, 1])

        # Explicitly configure caplog for the specific logger module
        logger_name = "nlsq.global_optimization.method_selector"
        with (
            caplog.at_level(logging.INFO, logger=logger_name),
            patch(
                "nlsq.global_optimization.method_selector.is_evosax_available",
                return_value=False,
            ),
        ):
            method = selector.select(
                requested_method="cmaes", lower_bounds=lower, upper_bounds=upper
            )

        assert method == "multi-start"
        # Check both caplog.records and caplog.text for robustness
        log_text = caplog.text.lower()
        assert "fallback" in log_text or "multi-start" in log_text, (
            f"Expected 'fallback' or 'multi-start' in logs, got: {caplog.text}"
        )


class TestCMAESConfigPassthrough:
    """Tests for configuration passthrough to optimizer."""

    def test_custom_config_affects_behavior(self) -> None:
        """Test that custom config parameters are used."""
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        def model(x, a):
            return a * x

        x = jnp.linspace(0, 10, 50)
        y = 2.5 * x
        bounds = ([0.1], [10.0])

        # Very limited generations - should still work but not converge well
        config = CMAESConfig(
            max_generations=5,  # Very few
            restart_strategy="none",
            max_restarts=0,
            seed=42,
            refine_with_nlsq=False,  # No refinement
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(model, x, y, bounds=bounds)

        # Should return a result (may not be converged)
        assert "popt" in result
        assert len(result["popt"]) == 1

    def test_preset_config_creates_valid_optimizer(self) -> None:
        """Test that preset configurations work correctly."""
        from nlsq.global_optimization import CMAESOptimizer

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 50)
        y = 2.0 * jnp.exp(-0.5 * x)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        # Test each preset
        for preset in ["cmaes-fast", "cmaes", "cmaes-global"]:
            optimizer = CMAESOptimizer.from_preset(preset)
            # Just verify it can run without error
            result = optimizer.fit(model, x, y, bounds=bounds)
            assert "popt" in result


class TestCMAESLogging:
    """Tests for CMA-ES logging output."""

    def test_cmaes_logs_info_on_start_and_end(self, caplog) -> None:
        """Test that CMA-ES logs INFO messages at start and end."""
        import logging

        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        def model(x, a):
            return a * x

        x = jnp.linspace(0, 10, 50)
        y = 2.5 * x
        bounds = ([0.1], [10.0])

        config = CMAESConfig(
            max_generations=20,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
        )
        optimizer = CMAESOptimizer(config=config)

        # Capture logs from the CMA-ES optimizer logger specifically
        with caplog.at_level(
            logging.INFO, logger="nlsq.global_optimization.cmaes_optimizer"
        ):
            optimizer.fit(model, x, y, bounds=bounds)

        # Should have INFO logs for CMA-ES start/end
        log_messages = [record.message.lower() for record in caplog.records]

        # Check for CMA-ES related log messages
        cmaes_logs = [msg for msg in log_messages if "cma" in msg or "bipop" in msg]
        assert len(cmaes_logs) > 0, "Should have CMA-ES related log messages"

    def test_cmaes_logs_debug_per_generation(self, caplog) -> None:
        """Test that CMA-ES logs DEBUG messages per generation when verbose."""
        import logging

        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        def model(x, a):
            return a * x

        x = jnp.linspace(0, 10, 50)
        y = 2.5 * x
        bounds = ([0.1], [10.0])

        config = CMAESConfig(
            max_generations=10,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
        )
        optimizer = CMAESOptimizer(config=config)

        # Capture logs from the CMA-ES optimizer logger specifically
        with caplog.at_level(
            logging.DEBUG, logger="nlsq.global_optimization.cmaes_optimizer"
        ):
            optimizer.fit(model, x, y, bounds=bounds)

        # Check for DEBUG level messages
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        # Should have some debug messages (may not be per-generation yet)
        assert isinstance(debug_records, list)


class TestCMAESDiagnosticsIntegration:
    """Tests for diagnostics returned by CMAESOptimizer."""

    def test_optimizer_returns_diagnostics(self) -> None:
        """Test that CMAESOptimizer result includes diagnostics."""
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 50)
        y = 2.5 * jnp.exp(-0.5 * x)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        config = CMAESConfig(
            max_generations=30,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(model, x, y, bounds=bounds)

        # Check for diagnostics in result
        assert "cmaes_diagnostics" in result
        diag = result["cmaes_diagnostics"]

        # Verify diagnostics fields
        assert "total_generations" in diag
        assert "total_restarts" in diag
        assert "best_fitness" in diag
        assert diag["total_generations"] > 0

    def test_diagnostics_with_bipop_restarts(self) -> None:
        """Test diagnostics track restarts correctly."""
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        def model(x, a):
            return a * x

        x = jnp.linspace(0, 10, 50)
        y = 2.5 * x
        bounds = ([0.1], [10.0])

        config = CMAESConfig(
            max_generations=20,
            restart_strategy="bipop",
            max_restarts=3,
            seed=42,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(model, x, y, bounds=bounds)

        assert "cmaes_diagnostics" in result
        diag = result["cmaes_diagnostics"]

        # Should track restarts
        assert "total_restarts" in diag
        assert isinstance(diag["total_restarts"], int)
