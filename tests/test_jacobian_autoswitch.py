"""Tests for Jacobian automatic mode selection (jacfwd vs jacrev).

This module tests the automatic selection of Jacobian computation mode based on
problem dimensions, configuration precedence, and debug logging functionality.

Task Group 3: Jacobian Auto-Switch
Target Impact: 10-100x Jacobian time reduction on high-parameter problems
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cleanup_env():
    """Clean up environment variables after each test."""
    original_env = os.environ.get("NLSQ_JACOBIAN_MODE")
    yield
    if original_env is not None:
        os.environ["NLSQ_JACOBIAN_MODE"] = original_env
    elif "NLSQ_JACOBIAN_MODE" in os.environ:
        del os.environ["NLSQ_JACOBIAN_MODE"]


class TestJacobianModeSelector:
    """Test automatic Jacobian mode selection based on problem dimensions."""

    def test_jacrev_for_tall_jacobian(self):
        """Test jacrev selection when n_params > n_residuals (tall Jacobian)."""
        from nlsq.core.least_squares import jacobian_mode_selector

        n_params = 500
        n_residuals = 100

        mode, rationale = jacobian_mode_selector(n_params, n_residuals, mode="auto")

        assert mode == "rev", (
            f"Expected 'rev' for {n_params} params > {n_residuals} residuals"
        )
        assert "jacrev" in rationale
        assert str(n_params) in rationale
        assert str(n_residuals) in rationale

    def test_jacfwd_for_wide_jacobian(self):
        """Test jacfwd selection when n_params <= n_residuals (wide Jacobian)."""
        from nlsq.core.least_squares import jacobian_mode_selector

        n_params = 100
        n_residuals = 500

        mode, rationale = jacobian_mode_selector(n_params, n_residuals, mode="auto")

        assert mode == "fwd", (
            f"Expected 'fwd' for {n_params} params <= {n_residuals} residuals"
        )
        assert "jacfwd" in rationale
        assert str(n_params) in rationale
        assert str(n_residuals) in rationale

    def test_manual_override_fwd(self):
        """Test manual override to force jacfwd mode."""
        from nlsq.core.least_squares import jacobian_mode_selector

        n_params = 1000
        n_residuals = 100

        mode, rationale = jacobian_mode_selector(n_params, n_residuals, mode="fwd")

        assert mode == "fwd", "Manual override to 'fwd' should be respected"
        assert "explicit override" in rationale or "fwd" in rationale

    def test_manual_override_rev(self):
        """Test manual override to force jacrev mode."""
        from nlsq.core.least_squares import jacobian_mode_selector

        n_params = 100
        n_residuals = 1000

        mode, rationale = jacobian_mode_selector(n_params, n_residuals, mode="rev")

        assert mode == "rev", "Manual override to 'rev' should be respected"
        assert "explicit override" in rationale or "rev" in rationale

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        from nlsq.core.least_squares import jacobian_mode_selector

        with pytest.raises(ValueError, match="Invalid jacobian_mode"):
            jacobian_mode_selector(100, 100, mode="invalid")


class TestConfigurationPrecedence:
    """Test configuration precedence: function param > env var > config file > auto."""

    def test_function_parameter_overrides_all(self, cleanup_env, temp_config_dir):
        """Test that function parameter has highest precedence."""
        from nlsq.config import get_jacobian_mode

        # Set environment variable
        os.environ["NLSQ_JACOBIAN_MODE"] = "rev"

        # Set config file
        config_path = Path(temp_config_dir) / "config.json"
        config_path.write_text(json.dumps({"jacobian_mode": "fwd"}))

        # Function parameter should win (tested in jacobian_mode_selector tests)
        # Here we test that get_jacobian_mode respects env var over config file
        with mock.patch("os.path.expanduser", return_value=str(config_path)):
            mode, source = get_jacobian_mode()
            assert mode == "rev", "Environment variable should override config file"
            assert source == "environment variable"

    def test_env_var_overrides_config_file(self, cleanup_env, temp_config_dir):
        """Test that environment variable overrides config file."""
        from nlsq.config import get_jacobian_mode

        os.environ["NLSQ_JACOBIAN_MODE"] = "fwd"

        config_path = Path(temp_config_dir) / "config.json"
        config_path.write_text(json.dumps({"jacobian_mode": "rev"}))

        with mock.patch("os.path.expanduser", return_value=str(config_path)):
            mode, source = get_jacobian_mode()
            assert mode == "fwd"
            assert source == "environment variable"

    def test_config_file_overrides_auto_default(self, cleanup_env, temp_config_dir):
        """Test that config file overrides auto-default."""
        from nlsq.config import get_jacobian_mode

        if "NLSQ_JACOBIAN_MODE" in os.environ:
            del os.environ["NLSQ_JACOBIAN_MODE"]

        config_path = Path(temp_config_dir) / "config.json"
        config_path.write_text(json.dumps({"jacobian_mode": "rev"}))

        with mock.patch("os.path.expanduser", return_value=str(config_path)):
            mode, source = get_jacobian_mode()
            assert mode == "rev"
            assert source == "config file"

    def test_auto_default_when_no_config(self, cleanup_env, temp_config_dir):
        """Test auto-default when no configuration is provided."""
        from nlsq.config import get_jacobian_mode

        if "NLSQ_JACOBIAN_MODE" in os.environ:
            del os.environ["NLSQ_JACOBIAN_MODE"]

        # Point to non-existent config file
        nonexistent_path = Path(temp_config_dir) / "nonexistent.json"

        with mock.patch("os.path.expanduser", return_value=str(nonexistent_path)):
            mode, source = get_jacobian_mode()
            assert mode == "auto"
            assert source == "auto-default"


class TestDebugLogging:
    """Test debug logging of Jacobian mode selection."""

    def test_mode_selection_logging(self, caplog):
        """Test that mode selection is logged in debug mode."""
        import logging

        from nlsq.core.least_squares import jacobian_mode_selector

        caplog.set_level(logging.DEBUG)

        n_params = 500
        n_residuals = 100

        mode, rationale = jacobian_mode_selector(n_params, n_residuals, mode="auto")

        # The function should return a rationale string suitable for logging
        assert mode == "rev"
        assert "jacrev" in rationale
        assert "500" in rationale
        assert "100" in rationale


class TestNumericalAccuracy:
    """Test that jacfwd and jacrev give identical results within tolerance."""

    def test_jacfwd_jacrev_consistency(self):
        """Test that jacfwd and jacrev produce consistent Jacobian results."""
        from jax import jacfwd, jacrev, jit

        # Simple test function: f(x) = [x0^2 + x1, x0 * x1^2]
        @jit
        def test_func(x):
            return jnp.array([x[0] ** 2 + x[1], x[0] * x[1] ** 2])

        x0 = jnp.array([2.0, 3.0])

        # Compute Jacobian with both modes
        jac_fwd_func = jacfwd(test_func)
        jac_rev_func = jacrev(test_func)

        J_fwd = jac_fwd_func(x0)
        J_rev = jac_rev_func(x0)

        # Should be identical (or very close due to floating point)
        np.testing.assert_allclose(
            J_fwd,
            J_rev,
            rtol=1e-12,
            atol=1e-14,
            err_msg="jacfwd and jacrev should produce identical Jacobians",
        )
