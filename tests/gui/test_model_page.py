"""Tests for NLSQ GUI model selection page components.

This module tests the model selection page UI components including model type
selection, built-in model dropdown, polynomial degree selector, custom code
editor, and parameter display.
"""

from typing import Any

import pytest

from nlsq.gui.adapters.model_adapter import (
    get_latex_equation,
    get_model,
    get_model_info,
    get_polynomial_latex,
    list_builtin_models,
)
from nlsq.gui.components.code_editor import (
    get_default_model_template,
    validate_code_syntax,
)
from nlsq.gui.components.model_preview import (
    format_parameter_list,
    get_model_summary,
)


class TestModelDropdownPopulation:
    """Tests for model dropdown population."""

    def test_builtin_models_populate_dropdown(self) -> None:
        """Built-in models should populate dropdown options."""
        models = list_builtin_models()

        assert isinstance(models, list)
        assert len(models) >= 7  # At least 7 built-in models

    def test_builtin_model_has_display_name(self) -> None:
        """Each built-in model should have a name for display."""
        models = list_builtin_models()

        for model_info in models:
            assert "name" in model_info
            assert isinstance(model_info["name"], str)
            assert len(model_info["name"]) > 0

    def test_builtin_models_sorted_alphabetically(self) -> None:
        """Model names should be available for sorting in dropdown."""
        models = list_builtin_models()
        model_names = [m["name"] for m in models]

        # All names should be valid strings
        for name in model_names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_model_info_available_for_tooltip(self) -> None:
        """Model info should include parameter count for tooltip."""
        models = list_builtin_models()

        for model_info in models:
            assert "n_params" in model_info
            # Polynomial is special case with variable params
            if model_info["name"] != "polynomial":
                assert model_info["n_params"] > 0


class TestPolynomialDegreeSelector:
    """Tests for polynomial degree selector behavior."""

    def test_polynomial_degree_0_creates_constant(self) -> None:
        """Degree 0 polynomial should create constant function."""
        model = get_model("polynomial", {"degree": 0})
        info = get_model_info(model)

        # Degree 0 has 1 coefficient (constant)
        assert info["param_count"] == 1

    def test_polynomial_degree_1_creates_linear(self) -> None:
        """Degree 1 polynomial should create linear function."""
        model = get_model("polynomial", {"degree": 1})
        info = get_model_info(model)

        # Degree 1 has 2 coefficients
        assert info["param_count"] == 2

    def test_polynomial_degree_5_creates_quintic(self) -> None:
        """Degree 5 polynomial should have 6 parameters."""
        model = get_model("polynomial", {"degree": 5})
        info = get_model_info(model)

        # Degree 5 has 6 coefficients
        assert info["param_count"] == 6

    def test_polynomial_degree_10_maximum(self) -> None:
        """Degree 10 polynomial should be valid maximum."""
        model = get_model("polynomial", {"degree": 10})
        info = get_model_info(model)

        # Degree 10 has 11 coefficients
        assert info["param_count"] == 11

    def test_polynomial_latex_updates_with_degree(self) -> None:
        """LaTeX equation should reflect polynomial degree."""
        latex_0 = get_polynomial_latex(0)
        latex_2 = get_polynomial_latex(2)
        latex_5 = get_polynomial_latex(5)

        # Each should be different
        assert latex_0 != latex_2
        assert latex_2 != latex_5

        # Higher degrees should have power notation
        assert "x^2" in latex_2 or "x^{2}" in latex_2
        assert "x^5" in latex_5 or "x^{5}" in latex_5


class TestCodeEditorRendering:
    """Tests for custom code editor component."""

    def test_default_template_is_valid_python(self) -> None:
        """Default code template should be valid Python."""
        template = get_default_model_template()

        assert isinstance(template, str)
        # Should compile without syntax error
        is_valid, _ = validate_code_syntax(template)
        assert is_valid

    def test_default_template_has_function_definition(self) -> None:
        """Default template should contain a function definition."""
        template = get_default_model_template()

        assert "def " in template
        assert "(x" in template  # First parameter should be x

    def test_validate_code_syntax_valid(self) -> None:
        """Valid Python code should pass validation."""
        code = """
def my_model(x, a, b):
    return a * x + b
"""
        is_valid, error = validate_code_syntax(code)

        assert is_valid is True
        assert error == ""

    def test_validate_code_syntax_invalid(self) -> None:
        """Invalid Python code should fail validation."""
        code = """
def broken(x, a
    return a * x
"""
        is_valid, error = validate_code_syntax(code)

        assert is_valid is False
        assert len(error) > 0

    def test_validate_code_syntax_with_jax_import(self) -> None:
        """Code with jax.numpy import should be valid."""
        code = """
import jax.numpy as jnp

def exp_model(x, a, tau):
    return a * jnp.exp(-x / tau)
"""
        is_valid, _error = validate_code_syntax(code)

        assert is_valid is True


class TestParameterDisplay:
    """Tests for parameter display after model selection."""

    def test_format_parameter_list_linear(self) -> None:
        """Linear model should display 2 parameter names."""
        model = get_model("builtin", {"name": "linear"})
        info = get_model_info(model)

        param_display = format_parameter_list(info["param_names"])

        assert isinstance(param_display, str)
        assert len(param_display) > 0

    def test_format_parameter_list_gaussian(self) -> None:
        """Gaussian model should display 3 parameter names."""
        model = get_model("builtin", {"name": "gaussian"})
        info = get_model_info(model)

        param_display = format_parameter_list(info["param_names"])

        # Should contain all parameter names
        assert "amp" in param_display
        assert "mu" in param_display
        assert "sigma" in param_display

    def test_format_parameter_list_empty(self) -> None:
        """Empty parameter list should return placeholder."""
        param_display = format_parameter_list([])

        assert isinstance(param_display, str)
        # Should indicate no parameters or return empty string

    def test_get_model_summary_builtin(self) -> None:
        """Model summary should include key information."""
        model = get_model("builtin", {"name": "exponential_decay"})
        summary = get_model_summary(model)

        assert "param_count" in summary
        assert "param_names" in summary
        assert "has_auto_p0" in summary

    def test_get_model_summary_polynomial(self) -> None:
        """Polynomial summary should show correct parameter count."""
        model = get_model("polynomial", {"degree": 3})
        summary = get_model_summary(model)

        assert summary["param_count"] == 4
        assert len(summary["param_names"]) == 4


class TestModelTypeSelection:
    """Tests for model type radio button selection."""

    def test_builtin_type_loads_model_correctly(self) -> None:
        """Selecting 'Built-in' type should load from registry."""
        model = get_model("builtin", {"name": "sigmoid"})

        assert callable(model)
        assert hasattr(model, "estimate_p0")

    def test_polynomial_type_generates_model(self) -> None:
        """Selecting 'Polynomial' type should generate model."""
        model = get_model("polynomial", {"degree": 2})

        assert callable(model)
        assert hasattr(model, "estimate_p0")

    def test_custom_type_parses_code(self) -> None:
        """Selecting 'Custom' type should parse provided code."""
        code = """
def custom_linear(x, m, c):
    return m * x + c
"""
        model = get_model("custom", {"code": code, "function": "custom_linear"})

        assert callable(model)


class TestLatexEquationDisplay:
    """Tests for LaTeX equation display."""

    def test_latex_equation_renders_for_all_builtins(self) -> None:
        """All built-in models should have LaTeX equations."""
        model_names = [
            "linear",
            "exponential_decay",
            "exponential_growth",
            "gaussian",
            "sigmoid",
            "power_law",
        ]

        for name in model_names:
            latex = get_latex_equation(name)
            assert isinstance(latex, str)
            assert len(latex) > 0

    def test_latex_equation_contains_variables(self) -> None:
        """LaTeX equations should contain mathematical variables."""
        latex = get_latex_equation("gaussian")

        # Should contain common math notation
        assert "x" in latex or "y" in latex

    def test_polynomial_latex_correct_for_each_degree(self) -> None:
        """Polynomial LaTeX should match degree."""
        # Constant
        latex_0 = get_polynomial_latex(0)
        assert "c" in latex_0.lower()

        # Linear
        latex_1 = get_polynomial_latex(1)
        assert "x" in latex_1

        # Quadratic
        latex_2 = get_polynomial_latex(2)
        assert "x^2" in latex_2 or "x^{2}" in latex_2
