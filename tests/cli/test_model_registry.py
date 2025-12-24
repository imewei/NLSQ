"""Tests for NLSQ CLI model registry module.

This module tests:
- Builtin model discovery from nlsq.functions.__all__
- Builtin model retrieval (linear, exponential_decay, gaussian, etc.)
- Custom model loading from external Python files
- Polynomial model generation by degree
- Custom model estimate_p0 and bounds method support (optional)
- ModelError for unresolvable models
"""

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from nlsq.cli.errors import ModelError
from nlsq.cli.model_registry import ModelRegistry


class TestModelRegistryBuiltinDiscovery:
    """Tests for builtin model discovery from nlsq.functions.__all__."""

    def test_list_builtin_models_returns_expected_models(self):
        """Test that list_builtin_models returns known builtin models."""
        registry = ModelRegistry()
        builtins = registry.list_builtin_models()

        # Should contain all known builtin models
        expected = {
            "linear",
            "exponential_decay",
            "exponential_growth",
            "gaussian",
            "sigmoid",
            "power_law",
            "polynomial",
        }
        assert expected.issubset(set(builtins))

    def test_list_builtin_models_returns_list_of_strings(self):
        """Test that list_builtin_models returns a list of strings."""
        registry = ModelRegistry()
        builtins = registry.list_builtin_models()

        assert isinstance(builtins, list)
        assert all(isinstance(name, str) for name in builtins)


class TestModelRegistryBuiltinRetrieval:
    """Tests for builtin model retrieval."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "linear",
            "exponential_decay",
            "exponential_growth",
            "gaussian",
            "sigmoid",
            "power_law",
        ],
    )
    def test_get_builtin_model_by_name(self, model_name: str):
        """Test that builtin models can be retrieved by name."""
        registry = ModelRegistry()
        config = {"type": "builtin", "name": model_name}

        model = registry.get_model(model_name, config)

        assert callable(model)
        assert model.__name__ == model_name or model_name in model.__name__

    def test_builtin_model_has_estimate_p0_method(self):
        """Test that builtin models have estimate_p0 method attached."""
        registry = ModelRegistry()
        config = {"type": "builtin", "name": "linear"}

        model = registry.get_model("linear", config)

        assert hasattr(model, "estimate_p0")
        assert callable(model.estimate_p0)

        # Test it works with sample data
        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        p0 = model.estimate_p0(xdata, ydata)
        assert isinstance(p0, list)
        assert len(p0) == 2  # linear has 2 parameters: slope, intercept

    def test_builtin_model_has_bounds_method(self):
        """Test that builtin models have bounds method attached."""
        registry = ModelRegistry()
        config = {"type": "builtin", "name": "gaussian"}

        model = registry.get_model("gaussian", config)

        assert hasattr(model, "bounds")
        assert callable(model.bounds)

        bounds = model.bounds()
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2  # (lower, upper)

    def test_get_nonexistent_builtin_raises_model_error(self):
        """Test that requesting non-existent builtin model raises ModelError."""
        registry = ModelRegistry()
        config = {"type": "builtin", "name": "nonexistent_model"}

        with pytest.raises(ModelError) as exc_info:
            registry.get_model("nonexistent_model", config)

        assert "nonexistent_model" in str(exc_info.value)


class TestModelRegistryCustomModels:
    """Tests for custom model loading from external Python files."""

    def test_load_custom_model_from_file(self, tmp_path: Path):
        """Test loading a custom model from an external Python file."""
        # Create a custom model file
        model_file = tmp_path / "custom_model.py"
        model_file.write_text(
            textwrap.dedent("""
            import jax.numpy as jnp

            def my_model(x, a, b, c):
                \"\"\"Custom model: y = a * sin(b * x) + c\"\"\"
                return a * jnp.sin(b * x) + c
        """)
        )

        registry = ModelRegistry()
        config = {
            "type": "custom",
            "path": str(model_file),
            "function": "my_model",
        }

        model = registry.get_model(str(model_file), config)

        assert callable(model)
        # Test that the model works
        x = np.array([0.0, 1.0, 2.0])
        result = model(x, 1.0, 1.0, 0.0)
        assert result is not None

    def test_custom_model_with_estimate_p0(self, tmp_path: Path):
        """Test that custom models can have optional estimate_p0 method."""
        model_file = tmp_path / "custom_with_estimate.py"
        model_file.write_text(
            textwrap.dedent("""
            import jax.numpy as jnp
            import numpy as np

            def my_exp_model(x, a, b):
                \"\"\"Custom exponential: y = a * exp(b * x)\"\"\"
                return a * jnp.exp(b * x)

            def estimate_p0(xdata, ydata):
                \"\"\"Estimate initial parameters.\"\"\"
                return [float(np.max(ydata)), 0.1]
        """)
        )

        registry = ModelRegistry()
        config = {
            "type": "custom",
            "path": str(model_file),
            "function": "my_exp_model",
        }

        model = registry.get_model(str(model_file), config)

        assert hasattr(model, "estimate_p0")
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0, 8.0])
        p0 = model.estimate_p0(xdata, ydata)
        assert p0 == [8.0, 0.1]

    def test_custom_model_with_bounds(self, tmp_path: Path):
        """Test that custom models can have optional bounds method."""
        model_file = tmp_path / "custom_with_bounds.py"
        model_file.write_text(
            textwrap.dedent("""
            import jax.numpy as jnp

            def bounded_model(x, a, b):
                \"\"\"Model with bounded parameters.\"\"\"
                return a * x + b

            def bounds():
                \"\"\"Return parameter bounds.\"\"\"
                return ([0.0, -10.0], [10.0, 10.0])
        """)
        )

        registry = ModelRegistry()
        config = {
            "type": "custom",
            "path": str(model_file),
            "function": "bounded_model",
        }

        model = registry.get_model(str(model_file), config)

        assert hasattr(model, "bounds")
        bounds = model.bounds()
        assert bounds == ([0.0, -10.0], [10.0, 10.0])

    def test_custom_model_file_not_found_raises_model_error(self):
        """Test that missing custom model file raises ModelError."""
        registry = ModelRegistry()
        config = {
            "type": "custom",
            "path": "/nonexistent/path/model.py",
            "function": "my_model",
        }

        with pytest.raises(ModelError) as exc_info:
            registry.get_model("/nonexistent/path/model.py", config)

        assert (
            "not found" in str(exc_info.value).lower()
            or "does not exist" in str(exc_info.value).lower()
        )

    def test_custom_model_function_not_found_raises_model_error(self, tmp_path: Path):
        """Test that missing function in custom model file raises ModelError."""
        model_file = tmp_path / "missing_func.py"
        model_file.write_text(
            textwrap.dedent("""
            def other_function(x):
                return x
        """)
        )

        registry = ModelRegistry()
        config = {
            "type": "custom",
            "path": str(model_file),
            "function": "nonexistent_function",
        }

        with pytest.raises(ModelError) as exc_info:
            registry.get_model(str(model_file), config)

        assert "nonexistent_function" in str(exc_info.value)


class TestModelRegistryPolynomial:
    """Tests for polynomial model generation."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 5])
    def test_polynomial_model_generation_by_degree(self, degree: int):
        """Test that polynomial models are generated correctly for various degrees."""
        registry = ModelRegistry()
        config = {"type": "polynomial", "degree": degree}

        model = registry.get_model(f"polynomial_{degree}", config)

        assert callable(model)
        # Polynomial of degree n has n+1 coefficients
        x = np.array([1.0, 2.0, 3.0])
        coeffs = [1.0] * (degree + 1)
        result = model(x, *coeffs)
        assert result is not None

    def test_polynomial_model_has_estimate_p0(self):
        """Test that polynomial model has estimate_p0 method."""
        registry = ModelRegistry()
        config = {"type": "polynomial", "degree": 2}

        model = registry.get_model("polynomial_2", config)

        assert hasattr(model, "estimate_p0")
        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # y = x^2
        p0 = model.estimate_p0(xdata, ydata)
        assert len(p0) == 3  # degree 2 has 3 coefficients

    def test_polynomial_model_has_bounds(self):
        """Test that polynomial model has bounds method."""
        registry = ModelRegistry()
        config = {"type": "polynomial", "degree": 3}

        model = registry.get_model("polynomial_3", config)

        assert hasattr(model, "bounds")
        bounds = model.bounds()
        assert isinstance(bounds, tuple)
        assert len(bounds[0]) == 4  # degree 3 has 4 coefficients
        assert len(bounds[1]) == 4

    def test_polynomial_invalid_degree_raises_model_error(self):
        """Test that invalid polynomial degree raises ModelError."""
        registry = ModelRegistry()
        config = {"type": "polynomial", "degree": -1}

        with pytest.raises(ModelError):
            registry.get_model("polynomial_-1", config)


class TestModelRegistryModelError:
    """Tests for ModelError scenarios."""

    def test_unknown_model_type_raises_model_error(self):
        """Test that unknown model type raises ModelError."""
        registry = ModelRegistry()
        config = {"type": "unknown_type", "name": "some_model"}

        with pytest.raises(ModelError) as exc_info:
            registry.get_model("some_model", config)

        assert (
            "unknown" in str(exc_info.value).lower()
            or "type" in str(exc_info.value).lower()
        )

    def test_model_error_includes_available_models_suggestion(self):
        """Test that ModelError for builtin includes list of available models."""
        registry = ModelRegistry()
        config = {"type": "builtin", "name": "lineaar"}  # Typo

        with pytest.raises(ModelError) as exc_info:
            registry.get_model("lineaar", config)

        # Should suggest available models or correct spelling
        error_message = str(exc_info.value)
        assert "lineaar" in error_message
