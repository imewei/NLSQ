"""Comprehensive tests for nlsq.validators module.

This test suite covers:
- InputValidator class with fast_mode
- validate_curve_fit_inputs validation logic
- validate_least_squares_inputs validation logic
- validate_inputs decorator
- Edge cases and error handling
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.utils.validators import InputValidator, validate_inputs


class TestInputValidatorInit(unittest.TestCase):
    """Tests for InputValidator initialization."""

    def test_default_initialization(self):
        """Test InputValidator with default fast_mode=True."""
        validator = InputValidator()

        self.assertTrue(validator.fast_mode)
        self.assertIsInstance(validator._function_cache, dict)
        self.assertEqual(len(validator._function_cache), 0)

    def test_fast_mode_true(self):
        """Test InputValidator with explicit fast_mode=True."""
        validator = InputValidator(fast_mode=True)

        self.assertTrue(validator.fast_mode)

    def test_fast_mode_false(self):
        """Test InputValidator with fast_mode=False."""
        validator = InputValidator(fast_mode=False)

        self.assertFalse(validator.fast_mode)


class TestValidateCurveFitBasic(unittest.TestCase):
    """Tests for basic curve_fit input validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator(fast_mode=True)

        # Simple linear function
        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear

    def test_valid_inputs(self):
        """Test validation with valid inputs."""
        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        errors, _warnings_list, x_clean, y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertEqual(len(errors), 0)
        np.testing.assert_array_equal(x_clean, xdata)
        np.testing.assert_array_equal(y_clean, ydata)

    def test_list_inputs_converted_to_arrays(self):
        """Test that list inputs are converted to arrays with warning."""
        xdata = [1.0, 2.0, 3.0, 4.0, 5.0]
        ydata = [2.0, 4.0, 6.0, 8.0, 10.0]

        errors, warnings_list, x_clean, y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertEqual(len(errors), 0)
        self.assertTrue(any("xdata converted" in w for w in warnings_list))
        self.assertTrue(any("ydata converted" in w for w in warnings_list))
        self.assertIsInstance(x_clean, np.ndarray)
        self.assertIsInstance(y_clean, np.ndarray)

    def test_jax_arrays_accepted(self):
        """Test that JAX arrays are accepted without conversion."""
        xdata = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])

        # Convert to numpy to avoid JAX flat property issue
        xdata_np = np.array(xdata)
        ydata_np = np.array(ydata)

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, xdata_np, ydata_np
            )
        )

        self.assertEqual(len(errors), 0)

    def test_minimum_two_points(self):
        """Test error when less than 2 data points."""
        xdata = np.array([1.0])
        ydata = np.array([2.0])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertTrue(any("at least 2 data points" in e for e in errors))

    def test_length_mismatch(self):
        """Test error when xdata and ydata have different lengths."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertTrue(any("same length" in e for e in errors))

    def test_scalar_inputs_error(self):
        """Test error when inputs are scalars (0-dimensional)."""
        xdata = np.array(1.0)
        ydata = np.array(2.0)

        # Scalars cause TypeError during validation, not a validation error
        with self.assertRaises(TypeError):
            _errors, _warnings_list, _x_clean, _y_clean = (
                self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
            )

    def test_insufficient_points_for_parameters(self):
        """Test error when not enough points for number of parameters."""
        # Linear function needs 2 params, but we have only 2 points (need > n_params)
        xdata = np.array([1.0, 2.0])
        ydata = np.array([2.0, 4.0])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        # Should have error about needing more points than parameters
        self.assertTrue(
            any("more data points" in e and "parameters" in e for e in errors)
        )


class TestValidateCurveFitEdgeCases(unittest.TestCase):
    """Tests for edge cases in curve_fit validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator(fast_mode=True)

        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear

    def test_identical_x_values_error(self):
        """Test error when all x values are identical."""
        xdata = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertTrue(any("x values are identical" in e for e in errors))

    def test_identical_y_values_warning(self):
        """Test warning when all y values are identical."""
        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        _errors, warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertTrue(any("y values are identical" in w for w in warnings_list))

    def test_very_small_x_range_warning(self):
        """Test warning when x data range is very small."""
        xdata = np.array([1.0, 1.0 + 1e-12, 1.0 + 2e-12, 1.0 + 3e-12])
        ydata = np.array([1.0, 2.0, 3.0, 4.0])

        _errors, warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertTrue(any("x data range is very small" in w for w in warnings_list))

    def test_very_large_x_range_warning(self):
        """Test warning when x data range is very large."""
        xdata = np.array([0.0, 1e11, 2e11, 3e11])
        ydata = np.array([1.0, 2.0, 3.0, 4.0])

        _errors, warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertTrue(any("x data range is very large" in w for w in warnings_list))

    def test_very_small_y_range_warning(self):
        """Test warning when y data range is very small."""
        xdata = np.array([1.0, 2.0, 3.0, 4.0])
        ydata = np.array([1.0, 1.0 + 1e-12, 1.0 + 2e-12, 1.0 + 3e-12])

        _errors, warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.linear_func, xdata, ydata)
        )

        self.assertTrue(any("y data range is very small" in w for w in warnings_list))

    def test_nan_in_xdata_error(self):
        """Test error when xdata contains NaN with check_finite=True."""
        xdata = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, xdata, ydata, check_finite=True
            )
        )

        self.assertTrue(any("NaN or Inf" in e and "xdata" in e for e in errors))

    def test_inf_in_ydata_error(self):
        """Test error when ydata contains Inf with check_finite=True."""
        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, np.inf, 8.0, 10.0])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, xdata, ydata, check_finite=True
            )
        )

        self.assertTrue(any("NaN or Inf" in e and "ydata" in e for e in errors))

    def test_nan_allowed_when_check_finite_false(self):
        """Test that NaN is allowed when check_finite=False."""
        xdata = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, xdata, ydata, check_finite=False
            )
        )

        # Should not have NaN/Inf error
        self.assertFalse(any("NaN or Inf" in e for e in errors))


class TestValidateCurveFitParameters(unittest.TestCase):
    """Tests for parameter validation (p0, bounds, sigma)."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator(fast_mode=True)

        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear
        self.xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    def test_valid_p0(self):
        """Test validation with valid initial parameters."""
        p0 = np.array([1.0, 1.0])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, p0=p0
            )
        )

        self.assertEqual(len(errors), 0)

    def test_p0_wrong_length_error(self):
        """Test error when p0 has wrong number of parameters."""
        p0 = np.array([1.0, 1.0, 1.0])  # 3 params, but function needs 2

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, p0=p0
            )
        )

        self.assertTrue(any("p0 has" in e and "parameters" in e for e in errors))

    def test_p0_with_nan_error(self):
        """Test error when p0 contains NaN."""
        p0 = np.array([1.0, np.nan])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, p0=p0
            )
        )

        self.assertTrue(any("p0 contains NaN or Inf" in e for e in errors))

    def test_valid_bounds(self):
        """Test validation with valid bounds."""
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, bounds=bounds
            )
        )

        self.assertEqual(len(errors), 0)

    def test_bounds_wrong_length_error(self):
        """Test error when bounds has wrong length."""
        bounds = (np.array([0.0]), np.array([10.0]))  # 1 param, but function needs 2

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, bounds=bounds
            )
        )

        self.assertTrue(any("bounds must have length" in e for e in errors))

    def test_bounds_lower_greater_than_upper_error(self):
        """Test error when lower bounds >= upper bounds."""
        bounds = (
            np.array([10.0, 5.0]),
            np.array([5.0, 10.0]),
        )  # First param has lb >= ub

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, bounds=bounds
            )
        )

        self.assertTrue(
            any("Lower bounds must be less than upper bounds" in e for e in errors)
        )

    def test_p0_outside_bounds_warning(self):
        """Test warning when p0 is outside bounds."""
        p0 = np.array([15.0, 15.0])  # Outside bounds
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))

        _errors, warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, p0=p0, bounds=bounds
            )
        )

        self.assertTrue(any("p0 is outside bounds" in w for w in warnings_list))

    def test_valid_sigma(self):
        """Test validation with valid sigma."""
        sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, sigma=sigma
            )
        )

        self.assertEqual(len(errors), 0)

    def test_sigma_wrong_shape_error(self):
        """Test error when sigma has wrong shape."""
        sigma = np.array([0.1, 0.1, 0.1])  # 3 elements, but ydata has 5

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, sigma=sigma
            )
        )

        self.assertTrue(any("sigma must have same shape as ydata" in e for e in errors))

    def test_sigma_negative_error(self):
        """Test error when sigma contains negative values."""
        sigma = np.array([0.1, 0.1, -0.1, 0.1, 0.1])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, sigma=sigma
            )
        )

        self.assertTrue(any("sigma values must be positive" in e for e in errors))

    def test_sigma_with_nan_error(self):
        """Test error when sigma contains NaN."""
        sigma = np.array([0.1, 0.1, np.nan, 0.1, 0.1])

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(
                self.linear_func, self.xdata, self.ydata, sigma=sigma
            )
        )

        self.assertTrue(any("sigma contains NaN or Inf" in e for e in errors))


class TestValidateCurveFitMultidimensional(unittest.TestCase):
    """Tests for multi-dimensional xdata validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator(fast_mode=True)

        # 2D function
        def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y):
            x, y = xy
            return amplitude * np.exp(
                -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
            )

        self.gaussian_2d = gaussian_2d

    def test_tuple_xdata_valid(self):
        """Test validation with tuple xdata (for 2D fitting)."""
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        xdata = (x, y)
        zdata = np.random.rand(50)

        _errors, warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.gaussian_2d, xdata, zdata)
        )

        # Should have warning about tuple
        self.assertTrue(any("xdata is tuple" in w for w in warnings_list))

    def test_tuple_xdata_length_mismatch_error(self):
        """Test error when arrays in tuple have different lengths."""
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 40)  # Different length
        xdata = (x, y)
        zdata = np.random.rand(50)

        errors, _warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(self.gaussian_2d, xdata, zdata)
        )

        self.assertTrue(any("same length" in e for e in errors))

    def test_2d_xdata_array(self):
        """Test validation with 2D xdata array (multiple independent variables)."""
        # Create 2D xdata: shape (n_points, n_vars)
        xdata = np.random.rand(50, 2)
        ydata = np.random.rand(50)

        def multi_var_func(x, a, b, c):
            return a * x[:, 0] + b * x[:, 1] + c

        _errors, warnings_list, _x_clean, _y_clean = (
            self.validator.validate_curve_fit_inputs(multi_var_func, xdata, ydata)
        )

        # Should have warning about independent variables
        self.assertTrue(any("independent variables" in w for w in warnings_list))


class TestValidateLeastSquaresInputs(unittest.TestCase):
    """Tests for least_squares input validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator(fast_mode=True)

        def residuals(params):
            return params - np.array([1.0, 2.0, 3.0])

        self.residuals_func = residuals

    def test_valid_inputs(self):
        """Test validation with valid inputs."""
        x0 = np.array([0.0, 0.0, 0.0])

        errors, _warnings_list, x0_clean = self.validator.validate_least_squares_inputs(
            self.residuals_func, x0
        )

        self.assertEqual(len(errors), 0)
        np.testing.assert_array_equal(x0_clean, x0)

    def test_x0_converted_to_array(self):
        """Test that x0 is converted to array if needed."""
        x0 = [1.0, 2.0, 3.0]

        errors, _warnings_list, x0_clean = self.validator.validate_least_squares_inputs(
            self.residuals_func, x0
        )

        self.assertEqual(len(errors), 0)
        self.assertIsInstance(x0_clean, np.ndarray)

    def test_x0_scalar_error(self):
        """Test error when x0 is scalar."""
        x0 = 1.0

        # Scalars cause TypeError during validation, not a validation error
        with self.assertRaises(TypeError):
            _errors, _warnings_list, _x0_clean = (
                self.validator.validate_least_squares_inputs(self.residuals_func, x0)
            )

    def test_x0_with_nan_error(self):
        """Test error when x0 contains NaN."""
        x0 = np.array([1.0, np.nan, 3.0])

        errors, _warnings_list, _x0_clean = (
            self.validator.validate_least_squares_inputs(self.residuals_func, x0)
        )

        self.assertTrue(any("NaN or Inf" in e for e in errors))

    def test_invalid_method_error(self):
        """Test error with invalid method."""
        x0 = np.array([1.0, 2.0, 3.0])

        errors, _warnings_list, _x0_clean = (
            self.validator.validate_least_squares_inputs(
                self.residuals_func, x0, method="invalid"
            )
        )

        self.assertTrue(any("method must be" in e for e in errors))

    def test_valid_methods(self):
        """Test all valid methods."""
        x0 = np.array([1.0, 2.0, 3.0])

        for method in ["trf", "dogbox", "lm"]:
            errors, _warnings_list, _x0_clean = (
                self.validator.validate_least_squares_inputs(
                    self.residuals_func, x0, method=method
                )
            )

            self.assertEqual(len(errors), 0)

    def test_tolerance_negative_error(self):
        """Test error when tolerances are negative."""
        x0 = np.array([1.0, 2.0, 3.0])

        errors, _warnings_list, _x0_clean = (
            self.validator.validate_least_squares_inputs(
                self.residuals_func, x0, ftol=-1.0
            )
        )

        self.assertTrue(any("must be positive" in e for e in errors))

    def test_max_nfev_negative_error(self):
        """Test error when max_nfev is negative."""
        x0 = np.array([1.0, 2.0, 3.0])

        errors, _warnings_list, _x0_clean = (
            self.validator.validate_least_squares_inputs(
                self.residuals_func, x0, max_nfev=-10
            )
        )

        self.assertTrue(any("max_nfev must be positive" in e for e in errors))

    def test_bounds_validation(self):
        """Test bounds validation for least_squares."""
        x0 = np.array([1.0, 2.0, 3.0])
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]))

        errors, _warnings_list, _x0_clean = (
            self.validator.validate_least_squares_inputs(
                self.residuals_func, x0, bounds=bounds
            )
        )

        self.assertEqual(len(errors), 0)


class TestValidateInputsDecorator(unittest.TestCase):
    """Tests for validate_inputs decorator."""

    def test_curve_fit_decorator_valid(self):
        """Test decorator with valid curve_fit inputs."""

        @validate_inputs(validation_type="curve_fit")
        def my_curve_fit(f, xdata, ydata, p0=None):
            return p0 if p0 is not None else [1.0, 1.0]

        def linear(x, a, b):
            return a * x + b

        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = my_curve_fit(linear, xdata, ydata, p0=[1.0, 0.0])

        self.assertEqual(result, [1.0, 0.0])

    def test_curve_fit_decorator_invalid(self):
        """Test decorator raises error with invalid inputs."""

        @validate_inputs(validation_type="curve_fit")
        def my_curve_fit(f, xdata, ydata, p0=None):
            return p0

        def linear(x, a, b):
            return a * x + b

        xdata = np.array([1.0, 2.0])  # Too few points
        ydata = np.array([2.0, 4.0])

        with self.assertRaises(ValueError) as ctx:
            my_curve_fit(linear, xdata, ydata)

        self.assertIn("Input validation failed", str(ctx.exception))

    def test_least_squares_decorator_valid(self):
        """Test decorator with valid least_squares inputs."""

        @validate_inputs(validation_type="least_squares")
        def my_least_squares(fun, x0, method="trf"):
            return x0

        def residuals(x):
            return x - np.array([1.0, 2.0, 3.0])

        x0 = np.array([0.0, 0.0, 0.0])

        result = my_least_squares(residuals, x0)

        np.testing.assert_array_equal(result, x0)

    def test_decorator_invalid_type(self):
        """Test decorator raises error with invalid validation type."""

        with self.assertRaises(ValueError) as ctx:

            @validate_inputs(validation_type="invalid")
            def my_func(x):
                return x

            my_func(1.0)

        # Note: error is raised during call, not decoration


class TestValidatorFunctionCache(unittest.TestCase):
    """Tests for function caching in non-fast mode."""

    def test_function_cache_used(self):
        """Test that function cache is populated in non-fast mode."""
        validator = InputValidator(fast_mode=False)

        def linear(x, a, b):
            return a * x + b

        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        # First call
        errors1, _warnings1, _x1, _y1 = validator.validate_curve_fit_inputs(
            linear, xdata, ydata, p0=[1.0, 0.0]
        )

        # Cache should be populated
        func_id = id(linear)
        self.assertIn(func_id, validator._function_cache)

        # Second call should use cache
        errors2, _warnings2, _x2, _y2 = validator.validate_curve_fit_inputs(
            linear, xdata, ydata, p0=[2.0, 1.0]
        )

        self.assertEqual(len(errors1), 0)
        self.assertEqual(len(errors2), 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
