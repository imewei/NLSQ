"""Tests for the model adapter module.

This module tests the GUI adapter that wraps nlsq.cli.model_registry.ModelRegistry
to provide model listing, loading, and introspection for the GUI.
"""

import inspect

import pytest

from nlsq.gui.adapters.model_adapter import (
    get_latex_equation,
    get_model,
    get_model_info,
    get_polynomial_latex,
    list_builtin_models,
    list_functions_in_module,
    load_custom_model_file,
    parse_custom_model_string,
    validate_jit_compatibility,
)


class TestListBuiltinModels:
    """Tests for list_builtin_models function."""

    def test_returns_list_of_dicts(self) -> None:
        """list_builtin_models should return a list of model info dictionaries."""
        models = list_builtin_models()

        assert isinstance(models, list)
        assert len(models) > 0
        for model_info in models:
            assert isinstance(model_info, dict)

    def test_model_info_has_required_fields(self) -> None:
        """Each model info dict should have name, n_params, has_estimate_p0, has_bounds."""
        models = list_builtin_models()

        for model_info in models:
            assert "name" in model_info
            assert "n_params" in model_info
            assert "has_estimate_p0" in model_info
            assert "has_bounds" in model_info
            assert isinstance(model_info["name"], str)
            assert isinstance(model_info["n_params"], int)
            assert isinstance(model_info["has_estimate_p0"], bool)
            assert isinstance(model_info["has_bounds"], bool)

    def test_contains_expected_models(self) -> None:
        """Should include all 7 built-in models."""
        models = list_builtin_models()
        model_names = [m["name"] for m in models]

        expected = [
            "linear",
            "exponential_decay",
            "exponential_growth",
            "gaussian",
            "sigmoid",
            "power_law",
            "polynomial",
        ]

        for name in expected:
            assert name in model_names, f"Missing expected model: {name}"

    def test_linear_model_info(self) -> None:
        """Linear model should have 2 parameters and support auto p0."""
        models = list_builtin_models()
        linear_info = next((m for m in models if m["name"] == "linear"), None)

        assert linear_info is not None
        assert linear_info["n_params"] == 2
        assert linear_info["has_estimate_p0"] is True
        assert linear_info["has_bounds"] is True


class TestGetModel:
    """Tests for get_model function."""

    def test_get_builtin_model_linear(self) -> None:
        """Should load the linear builtin model correctly."""
        model = get_model("builtin", {"name": "linear"})

        assert callable(model)
        assert hasattr(model, "estimate_p0")
        assert hasattr(model, "bounds")

    def test_get_builtin_model_gaussian(self) -> None:
        """Should load the gaussian builtin model correctly."""
        model = get_model("builtin", {"name": "gaussian"})

        assert callable(model)
        # Test that the model works
        import jax.numpy as jnp

        result = model(jnp.array([0.0]), 1.0, 0.0, 1.0)
        assert float(result[0]) == pytest.approx(1.0, rel=1e-6)

    def test_get_polynomial_model(self) -> None:
        """Should generate polynomial model of given degree."""
        model = get_model("polynomial", {"degree": 2})

        assert callable(model)
        assert hasattr(model, "estimate_p0")
        assert hasattr(model, "bounds")

        # Test quadratic: 1*x^2 + 0*x + 0 = x^2
        import jax.numpy as jnp

        result = model(jnp.array([2.0]), 1.0, 0.0, 0.0)
        assert float(result[0]) == pytest.approx(4.0, rel=1e-6)

    def test_get_polynomial_different_degrees(self) -> None:
        """Polynomial generation should work for various degrees."""
        for degree in [0, 1, 3, 5]:
            model = get_model("polynomial", {"degree": degree})
            assert callable(model)
            # Number of coefficients should be degree + 1
            sig = inspect.signature(model)
            # First param is x, rest are coefficients
            n_params = len(sig.parameters) - 1  # Subtract x
            # For *args, we check estimate_p0 output
            p0 = model.estimate_p0([1, 2, 3], [1, 4, 9])
            assert len(p0) == degree + 1

    def test_invalid_model_type_raises(self) -> None:
        """Should raise ValueError for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model("invalid_type", {})


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_extracts_linear_info(self) -> None:
        """Should extract correct info from linear model."""
        model = get_model("builtin", {"name": "linear"})
        info = get_model_info(model)

        assert "param_names" in info
        assert "param_count" in info
        assert info["param_count"] == 2
        # Linear has parameters: a (slope), b (intercept)
        assert len(info["param_names"]) == 2

    def test_extracts_gaussian_info(self) -> None:
        """Should extract correct info from gaussian model."""
        model = get_model("builtin", {"name": "gaussian"})
        info = get_model_info(model)

        assert info["param_count"] == 3
        assert len(info["param_names"]) == 3
        # Gaussian has parameters: amp, mu, sigma
        assert "amp" in info["param_names"]
        assert "mu" in info["param_names"]
        assert "sigma" in info["param_names"]

    def test_extracts_polynomial_info(self) -> None:
        """Should extract info from dynamically generated polynomial."""
        model = get_model("polynomial", {"degree": 3})
        info = get_model_info(model)

        # Polynomial of degree 3 has 4 coefficients
        assert info["param_count"] == 4


class TestParseCustomModelString:
    """Tests for parse_custom_model_string function."""

    def test_parse_simple_function(self) -> None:
        """Should parse inline Python code defining a model function."""
        code = """
def my_model(x, a, b):
    return a * x + b
"""
        func, param_names = parse_custom_model_string(code, "my_model")

        assert callable(func)
        assert param_names == ["a", "b"]

    def test_parse_with_jax_import(self) -> None:
        """Should parse code using jax.numpy."""
        code = """
import jax.numpy as jnp

def exp_model(x, a, b):
    return a * jnp.exp(-b * x)
"""
        func, param_names = parse_custom_model_string(code, "exp_model")

        assert callable(func)
        assert param_names == ["a", "b"]

    def test_parse_extracts_param_names(self) -> None:
        """Should correctly extract parameter names from function signature."""
        code = """
def complex_model(x, amplitude, frequency, phase, offset):
    import jax.numpy as jnp
    return amplitude * jnp.sin(frequency * x + phase) + offset
"""
        _func, param_names = parse_custom_model_string(code, "complex_model")

        assert param_names == ["amplitude", "frequency", "phase", "offset"]

    def test_parse_function_not_found_raises(self) -> None:
        """Should raise ValueError if function name not found in code."""
        code = """
def some_function(x, a):
    return a * x
"""
        with pytest.raises(ValueError, match="not found"):
            parse_custom_model_string(code, "nonexistent_function")

    def test_parse_syntax_error_raises(self) -> None:
        """Should raise SyntaxError for invalid Python code."""
        code = """
def broken(x, a
    return a * x
"""
        with pytest.raises(SyntaxError):
            parse_custom_model_string(code, "broken")


class TestListFunctionsInModule:
    """Tests for list_functions_in_module function."""

    def test_lists_all_functions(self) -> None:
        """Should list all public function names in code."""
        code = """
def func_a(x, a):
    return a * x

def func_b(x, a, b):
    return a * x + b

def _private_func(x):
    return x
"""
        functions = list_functions_in_module(code)

        assert "func_a" in functions
        assert "func_b" in functions
        # Private functions may or may not be included based on implementation
        # but we test that public ones are present

    def test_empty_code_returns_empty_list(self) -> None:
        """Should return empty list for code with no functions."""
        code = """
x = 1
y = 2
"""
        functions = list_functions_in_module(code)
        assert functions == []


class TestValidateJitCompatibility:
    """Tests for validate_jit_compatibility function."""

    def test_valid_jax_code(self) -> None:
        """Code using jax.numpy should be JIT compatible."""
        code = """
import jax.numpy as jnp

def model(x, a):
    return a * jnp.exp(x)
"""
        assert validate_jit_compatibility(code) is True

    def test_numpy_warning(self) -> None:
        """Code using numpy instead of jax.numpy may not be JIT compatible."""
        code = """
import numpy as np

def model(x, a):
    return a * np.exp(x)
"""
        # This should still validate but may return False or a warning
        # Implementation may choose to warn rather than fail
        result = validate_jit_compatibility(code)
        assert isinstance(result, bool)


class TestGetLatexEquation:
    """Tests for get_latex_equation function."""

    def test_linear_latex(self) -> None:
        """Linear model should return valid LaTeX."""
        latex = get_latex_equation("linear")

        assert isinstance(latex, str)
        assert len(latex) > 0
        assert "a" in latex or "x" in latex

    def test_exponential_decay_latex(self) -> None:
        """Exponential decay should return valid LaTeX with exp."""
        latex = get_latex_equation("exponential_decay")

        assert isinstance(latex, str)
        assert "exp" in latex.lower() or "e^" in latex

    def test_gaussian_latex(self) -> None:
        """Gaussian should return valid LaTeX with exp and sigma."""
        latex = get_latex_equation("gaussian")

        assert isinstance(latex, str)
        assert "exp" in latex.lower() or "e^" in latex
        assert "sigma" in latex.lower() or "\\sigma" in latex

    def test_sigmoid_latex(self) -> None:
        """Sigmoid should return valid LaTeX."""
        latex = get_latex_equation("sigmoid")

        assert isinstance(latex, str)
        assert "1" in latex  # 1/(1+exp(...))

    def test_power_law_latex(self) -> None:
        """Power law should return valid LaTeX with exponent."""
        latex = get_latex_equation("power_law")

        assert isinstance(latex, str)
        assert "^" in latex or "**" in latex  # Exponentiation

    def test_unknown_model_returns_placeholder(self) -> None:
        """Unknown model should return a placeholder or generic equation."""
        latex = get_latex_equation("unknown_model_xyz")

        assert isinstance(latex, str)
        # Should return something, even if generic


class TestGetPolynomialLatex:
    """Tests for get_polynomial_latex function."""

    def test_degree_0(self) -> None:
        """Degree 0 polynomial is a constant."""
        latex = get_polynomial_latex(0)

        assert isinstance(latex, str)
        assert "c" in latex.lower() or "a" in latex.lower()

    def test_degree_1(self) -> None:
        """Degree 1 polynomial is linear."""
        latex = get_polynomial_latex(1)

        assert isinstance(latex, str)
        assert "x" in latex

    def test_degree_2(self) -> None:
        """Degree 2 polynomial is quadratic."""
        latex = get_polynomial_latex(2)

        assert isinstance(latex, str)
        assert "x^2" in latex or "x^{2}" in latex

    def test_degree_5(self) -> None:
        """Higher degree polynomial should have correct powers."""
        latex = get_polynomial_latex(5)

        assert isinstance(latex, str)
        assert "x^5" in latex or "x^{5}" in latex


class TestLoadCustomModelFile:
    """Tests for load_custom_model_file function."""

    def test_load_nonexistent_file_raises(self, tmp_path) -> None:
        """Should raise FileNotFoundError for missing file."""
        fake_path = tmp_path / "nonexistent.py"

        with pytest.raises(FileNotFoundError):
            load_custom_model_file(str(fake_path), "model")

    def test_load_valid_file(self, tmp_path) -> None:
        """Should load model function from valid Python file."""
        model_file = tmp_path / "my_model.py"
        model_file.write_text("""
import jax.numpy as jnp

def custom_exponential(x, a, tau):
    return a * jnp.exp(-x / tau)
""")

        func, param_names = load_custom_model_file(
            str(model_file), "custom_exponential"
        )

        assert callable(func)
        assert param_names == ["a", "tau"]

    def test_load_file_function_not_found(self, tmp_path) -> None:
        """Should raise ValueError if function not in file."""
        model_file = tmp_path / "other_model.py"
        model_file.write_text("""
def some_func(x, a):
    return a * x
""")

        with pytest.raises(ValueError, match="not found"):
            load_custom_model_file(str(model_file), "nonexistent_func")
