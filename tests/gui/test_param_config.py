"""Tests for the parameter configuration component.

This module tests the parameter configuration component for the NLSQ GUI,
which provides dynamic input fields for model parameters, p0 estimation,
bounds configuration, and parameter transforms.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from nlsq.gui.components.param_config import (
    TRANSFORM_OPTIONS,
    create_param_config_dict,
    estimate_p0_for_model,
    format_p0_display,
    get_default_p0_value,
    get_param_names_from_model,
    validate_bounds,
    validate_p0_input,
)
from nlsq.gui.state import SessionState, initialize_state


class TestGetParamNamesFromModel:
    """Tests for extracting parameter names from models."""

    def test_builtin_model_gaussian(self) -> None:
        """Gaussian model should have amp, mu, sigma parameters."""
        from nlsq.gui.adapters.model_adapter import get_model

        model = get_model("builtin", {"name": "gaussian"})
        param_names = get_param_names_from_model(model)

        assert isinstance(param_names, list)
        assert len(param_names) == 3
        assert "amp" in param_names
        assert "mu" in param_names
        assert "sigma" in param_names

    def test_builtin_model_linear(self) -> None:
        """Linear model should have 2 parameters."""
        from nlsq.gui.adapters.model_adapter import get_model

        model = get_model("builtin", {"name": "linear"})
        param_names = get_param_names_from_model(model)

        assert isinstance(param_names, list)
        assert len(param_names) == 2

    def test_polynomial_model(self) -> None:
        """Polynomial model should have degree+1 parameters."""
        from nlsq.gui.adapters.model_adapter import get_model

        model = get_model("polynomial", {"degree": 3})
        param_names = get_param_names_from_model(model)

        assert isinstance(param_names, list)
        assert len(param_names) == 4  # degree 3 = 4 coefficients


class TestValidateBounds:
    """Tests for bounds validation."""

    def test_valid_bounds(self) -> None:
        """Valid bounds where lower < upper should pass."""
        is_valid, message = validate_bounds(0.0, 10.0)
        assert is_valid is True
        assert message == ""

    def test_invalid_bounds_lower_greater(self) -> None:
        """Lower bound > upper bound should fail."""
        is_valid, message = validate_bounds(10.0, 0.0)
        assert is_valid is False
        assert "lower" in message.lower() or "upper" in message.lower()

    def test_equal_bounds_valid(self) -> None:
        """Equal bounds should be valid (fixed parameter)."""
        is_valid, _message = validate_bounds(5.0, 5.0)
        assert is_valid is True

    def test_none_bounds_valid(self) -> None:
        """None bounds (no constraint) should be valid."""
        is_valid, _message = validate_bounds(None, None)
        assert is_valid is True

    def test_partial_bounds_valid(self) -> None:
        """Only lower or upper bound should be valid."""
        is_valid_lower, _ = validate_bounds(0.0, None)
        is_valid_upper, _ = validate_bounds(None, 10.0)
        assert is_valid_lower is True
        assert is_valid_upper is True

    def test_infinite_bounds_valid(self) -> None:
        """Infinite bounds should be valid."""
        is_valid, _message = validate_bounds(float("-inf"), float("inf"))
        assert is_valid is True


class TestAutoP0Estimation:
    """Tests for automatic p0 estimation."""

    def test_estimate_p0_with_gaussian(self) -> None:
        """Gaussian model should provide p0 estimation."""
        import numpy as np

        from nlsq.gui.adapters.model_adapter import get_model

        model = get_model("builtin", {"name": "gaussian"})
        xdata = np.linspace(-5, 5, 50)
        ydata = 2.0 * np.exp(-((xdata - 1.0) ** 2) / (2 * 0.5**2))

        p0 = estimate_p0_for_model(model, xdata, ydata)

        assert p0 is not None
        assert isinstance(p0, (list, tuple))
        assert len(p0) == 3

    def test_estimate_p0_with_linear(self) -> None:
        """Linear model should provide p0 estimation."""
        import numpy as np

        from nlsq.gui.adapters.model_adapter import get_model

        model = get_model("builtin", {"name": "linear"})
        xdata = np.array([0, 1, 2, 3, 4])
        ydata = np.array([1, 3, 5, 7, 9])

        p0 = estimate_p0_for_model(model, xdata, ydata)

        assert p0 is not None
        assert len(p0) == 2
        # Should estimate slope ~2, intercept ~1
        assert abs(p0[0] - 2.0) < 0.5  # slope
        assert abs(p0[1] - 1.0) < 0.5  # intercept

    def test_estimate_p0_returns_none_for_custom(self) -> None:
        """Custom model without estimate_p0 should return None."""

        def custom_model(x, a, b):
            return a * x + b

        p0 = estimate_p0_for_model(custom_model, [1, 2, 3], [1, 2, 3])

        assert p0 is None


class TestTransformOptions:
    """Tests for parameter transform options."""

    def test_transform_options_available(self) -> None:
        """Transform options should include standard transforms."""
        assert "none" in TRANSFORM_OPTIONS
        assert "log" in TRANSFORM_OPTIONS
        assert "logit" in TRANSFORM_OPTIONS
        assert "exp" in TRANSFORM_OPTIONS

    def test_transform_options_count(self) -> None:
        """Should have at least 4 transform options."""
        assert len(TRANSFORM_OPTIONS) >= 4


class TestCreateParamConfigDict:
    """Tests for creating parameter configuration dictionaries."""

    def test_create_empty_config(self) -> None:
        """Empty param list should return empty config."""
        config = create_param_config_dict([])
        assert config == {}

    def test_create_config_with_params(self) -> None:
        """Should create config entry for each parameter."""
        param_names = ["a", "b", "c"]
        config = create_param_config_dict(param_names)

        assert len(config) == 3
        for name in param_names:
            assert name in config
            assert "p0" in config[name]
            assert "lower" in config[name]
            assert "upper" in config[name]
            assert "transform" in config[name]
            assert "auto" in config[name]

    def test_config_defaults(self) -> None:
        """Config should have correct default values."""
        config = create_param_config_dict(["param"])

        assert config["param"]["p0"] is None
        assert config["param"]["lower"] is None
        assert config["param"]["upper"] is None
        assert config["param"]["transform"] == "none"
        assert config["param"]["auto"] is True


class TestGetDefaultP0Value:
    """Tests for getting default p0 values."""

    def test_default_value(self) -> None:
        """Default p0 should be 1.0."""
        value = get_default_p0_value()
        assert value == 1.0


class TestFormatP0Display:
    """Tests for formatting p0 values for display."""

    def test_format_none(self) -> None:
        """None should display as 'Auto'."""
        display = format_p0_display(None)
        assert display == "Auto"

    def test_format_float(self) -> None:
        """Float should be formatted appropriately."""
        display = format_p0_display(1.5)
        assert "1.5" in display

    def test_format_scientific(self) -> None:
        """Small values should use scientific notation."""
        display = format_p0_display(1e-10)
        assert "e" in display.lower() or "E" in display


class TestValidateP0Input:
    """Tests for validating p0 input values."""

    def test_valid_float(self) -> None:
        """Valid float should pass."""
        is_valid, value, message = validate_p0_input("1.5")
        assert is_valid is True
        assert value == 1.5
        assert message == ""

    def test_valid_scientific(self) -> None:
        """Scientific notation should be valid."""
        is_valid, value, _message = validate_p0_input("1e-5")
        assert is_valid is True
        assert value == 1e-5

    def test_invalid_string(self) -> None:
        """Non-numeric string should fail."""
        is_valid, value, message = validate_p0_input("abc")
        assert is_valid is False
        assert value is None
        assert len(message) > 0

    def test_empty_string(self) -> None:
        """Empty string should return None (auto mode)."""
        is_valid, value, _message = validate_p0_input("")
        assert is_valid is True
        assert value is None


class TestStateSynchronization:
    """Tests for synchronizing param config with session state."""

    def test_state_preserves_p0(self) -> None:
        """p0 values should be preserved in state."""
        state = initialize_state()
        state.p0 = [1.0, 2.0, 3.0]

        assert state.p0 == [1.0, 2.0, 3.0]

    def test_state_preserves_bounds(self) -> None:
        """Bounds should be preserved in state."""
        state = initialize_state()
        state.bounds = ([0.0, 0.0], [10.0, 10.0])

        assert state.bounds[0] == [0.0, 0.0]
        assert state.bounds[1] == [10.0, 10.0]

    def test_state_preserves_transforms(self) -> None:
        """Transforms should be preserved in state."""
        state = initialize_state()
        state.transforms = {"a": "log", "b": "none"}

        assert state.transforms["a"] == "log"
        assert state.transforms["b"] == "none"
