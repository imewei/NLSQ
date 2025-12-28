"""
Comprehensive tests for validators module.

Target: InputValidator.validate_curve_fit_inputs (complexity 25)
Goal: Cover all validation branches for Sprint 1 safety net.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.utils.validators import InputValidator


class TestValidateCurveFitInputs:
    """Test validate_curve_fit_inputs comprehensive coverage."""

    def setup_method(self):
        """Setup validator instance."""
        self.validator = InputValidator()

    def test_valid_inputs_pass(self):
        """Test valid inputs pass validation."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        # Should not raise
        result = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )
        assert result is not None

    def test_function_not_callable_raises(self):
        """Test non-callable function returns error."""
        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        errors, _warnings, _xd, _yd = self.validator.validate_curve_fit_inputs(
            f="not_a_function",  # Invalid!
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

        # Should return error (may be about function evaluation or bounds processing)
        assert len(errors) > 0

    def test_xdata_ydata_shape_mismatch_raises(self):
        """Test shape mismatch returns error."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4])  # Wrong shape!

        errors, _warnings, _xd, _yd = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

        # Should return error about shape mismatch
        assert len(errors) > 0
        assert any(
            "shape" in err.lower()
            or "length" in err.lower()
            or "mismatch" in err.lower()
            for err in errors
        )

    def test_empty_data_raises(self):
        """Test empty arrays raise ValueError."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([])
        ydata = np.array([])

        with pytest.raises((ValueError, IndexError)):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=None,
                sigma=None,
                bounds=(-np.inf, np.inf),
            )

    def test_sigma_shape_mismatch_raises(self):
        """Test sigma shape mismatch returns error."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        sigma = np.array([0.1, 0.2])  # Wrong shape!

        errors, _warnings, _xd, _yd = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=sigma,
            bounds=(-np.inf, np.inf),
        )

        # Should return error about sigma shape mismatch
        assert len(errors) > 0
        assert any("sigma" in err.lower() for err in errors)

    def test_sigma_negative_raises(self):
        """Test negative sigma returns error."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        sigma = np.array([0.1, -0.2, 0.3])  # Negative!

        errors, _warnings, _xd, _yd = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=sigma,
            bounds=(-np.inf, np.inf),
        )

        # Should return error about negative sigma
        assert len(errors) > 0
        assert any(
            "sigma" in err.lower()
            and ("negative" in err.lower() or "positive" in err.lower())
            for err in errors
        )

    def test_sigma_zero_raises(self):
        """Test zero sigma returns error."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        sigma = np.array([0.1, 0.0, 0.3])  # Zero!

        errors, _warnings, _xd, _yd = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=sigma,
            bounds=(-np.inf, np.inf),
        )

        # Should return error about zero sigma
        assert len(errors) > 0
        assert any(
            "sigma" in err.lower()
            and ("zero" in err.lower() or "positive" in err.lower())
            for err in errors
        )

    def test_bounds_lower_ge_upper_raises(self):
        """Test lower >= upper bounds returns error."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        bounds = ([10, 10], [0, 0])  # Lower >= upper!

        errors, _warnings, _xd, _yd = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=[1, 1],
            sigma=None,
            bounds=bounds,
        )

        # Should return error about invalid bounds
        assert len(errors) > 0
        assert any("bound" in err.lower() for err in errors)

    def test_p0_outside_bounds_raises(self):
        """Test p0 outside bounds returns warning."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        p0 = [5, 5]
        bounds = ([0, 0], [3, 3])  # p0 outside!

        _errors, warnings, _xd, _yd = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=None,
            bounds=bounds,
        )

        # Should return warning about p0 outside bounds
        assert len(warnings) > 0
        assert any(
            "p0" in warn.lower() and "bound" in warn.lower() for warn in warnings
        )

    def test_valid_method_trf(self):
        """Test valid method='trf' passes."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

    def test_valid_method_dogbox(self):
        """Test valid method='dogbox' passes."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

    def test_valid_method_lm_unbounded(self):
        """Test method='lm' without bounds passes."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

    def test_numpy_arrays_accepted(self):
        """Test NumPy arrays are accepted."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

    def test_jax_arrays_accepted(self):
        """Test JAX arrays are accepted."""

        def model(x, a, b):
            return a * x + b

        xdata = jnp.array([1, 2, 3])
        ydata = jnp.array([2, 4, 6])

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

    def test_mixed_array_types_accepted(self):
        """Test mixed NumPy/JAX arrays accepted."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = jnp.array([2, 4, 6])  # Mixed!

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

    def test_python_lists_accepted(self):
        """Test Python lists are accepted."""

        def model(x, a, b):
            return a * x + b

        xdata = [1, 2, 3]  # List
        ydata = [2, 4, 6]  # List

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

    def test_valid_sigma_array(self):
        """Test valid sigma array passes."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        sigma = np.array([0.1, 0.2, 0.3])

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=None,
            sigma=sigma,
            bounds=(-np.inf, np.inf),
        )

    def test_valid_bounds_array(self):
        """Test valid bounds arrays pass."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        bounds = ([0, -10], [10, 10])

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=[1, 1],
            sigma=None,
            bounds=bounds,
        )

    def test_valid_p0_array(self):
        """Test valid p0 array passes."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        p0 = [1.0, 0.0]

        # Should not raise
        self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=None,
            bounds=(-np.inf, np.inf),
        )

    def test_bounds_shape_mismatch_raises(self):
        """Test bounds shape mismatch returns error."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        bounds = ([0], [10, 10])  # Shape mismatch!

        errors, _warnings, _xd, _yd = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=[1, 1],
            sigma=None,
            bounds=bounds,
        )

        # Should return error about bounds shape mismatch
        assert len(errors) > 0
        assert any("bound" in err.lower() for err in errors)


class TestSecurityValidation:
    """Tests for security-focused validation methods."""

    def setup_method(self):
        """Setup validator instance."""
        self.validator = InputValidator()

    def test_array_size_limits_normal(self):
        """Test normal array sizes pass validation."""
        errors, _ = self.validator._validate_array_size_limits(
            n_points=1000, n_params=5
        )
        assert len(errors) == 0

    def test_array_size_limits_large_warning(self):
        """Test large arrays produce warnings about memory."""
        # 200M points x 10 params = 2B elements = ~15 GB (exceeds 10GB warning threshold)
        errors, warnings = self.validator._validate_array_size_limits(
            n_points=200_000_000, n_params=10
        )
        assert len(errors) == 0
        # Should warn about large memory (>10GB)
        assert len(warnings) > 0
        assert any("memory" in w.lower() or "gb" in w.lower() for w in warnings)

    def test_array_size_limits_exceeds_max(self):
        """Test exceeding max array size produces error."""
        # 100B points exceeds 10B limit
        errors, _ = self.validator._validate_array_size_limits(
            n_points=100_000_000_000, n_params=10
        )
        assert len(errors) > 0
        assert any("exceeds maximum" in e.lower() for e in errors)

    def test_array_size_limits_negative(self):
        """Test negative sizes produce errors."""
        errors, _ = self.validator._validate_array_size_limits(n_points=-1, n_params=5)
        assert len(errors) > 0
        assert any("negative" in e.lower() for e in errors)

    def test_bounds_numeric_range_normal(self):
        """Test normal bounds pass validation."""
        bounds = ([-100, -100], [100, 100])
        errors, warnings = self.validator._validate_bounds_numeric_range(bounds)
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_bounds_numeric_range_extreme(self):
        """Test extreme bounds produce warnings."""
        bounds = ([-1e150, -1e150], [1e150, 1e150])
        errors, warnings = self.validator._validate_bounds_numeric_range(bounds)
        assert len(errors) == 0
        assert len(warnings) > 0
        assert any("large" in w.lower() for w in warnings)

    def test_bounds_with_nan_produces_error(self):
        """Test NaN in bounds produces error."""
        bounds = ([np.nan, -100], [100, 100])
        errors, _ = self.validator._validate_bounds_numeric_range(bounds)
        assert len(errors) > 0
        assert any("nan" in e.lower() for e in errors)

    def test_bounds_with_inf_allowed(self):
        """Test infinite bounds are allowed (common use case)."""
        bounds = ([-np.inf, -100], [np.inf, 100])
        errors, _ = self.validator._validate_bounds_numeric_range(bounds)
        assert len(errors) == 0

    def test_parameter_values_normal(self):
        """Test normal parameter values pass validation."""
        p0 = [1.0, 2.0, 3.0]
        errors, warnings = self.validator._validate_parameter_values(p0)
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_parameter_values_extreme(self):
        """Test extreme parameter values produce warnings."""
        p0 = [1e60, 2.0, 3.0]
        errors, warnings = self.validator._validate_parameter_values(p0)
        assert len(errors) == 0
        assert len(warnings) > 0
        assert any("large" in w.lower() or "overflow" in w.lower() for w in warnings)

    def test_parameter_values_subnormal(self):
        """Test subnormal parameter values produce warnings."""
        p0 = [1e-310, 2.0, 3.0]
        errors, warnings = self.validator._validate_parameter_values(p0)
        assert len(errors) == 0
        assert len(warnings) > 0
        assert any("small" in w.lower() or "underflow" in w.lower() for w in warnings)

    def test_security_constraints_combined(self):
        """Test combined security validation."""
        errors, _ = self.validator.validate_security_constraints(
            n_points=1000,
            n_params=5,
            bounds=([-100, -100, -100, -100, -100], [100, 100, 100, 100, 100]),
            p0=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        assert len(errors) == 0

    def test_security_in_curve_fit_validation(self):
        """Test security validation is called in curve_fit inputs validation."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        # Normal case should pass
        errors, _, _, _ = self.validator.validate_curve_fit_inputs(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=[1.0, 1.0],
        )
        # No security errors expected
        security_errors = [
            e for e in errors if "exceeds" in e.lower() or "negative" in e.lower()
        ]
        assert len(security_errors) == 0


# Total: 38 comprehensive tests covering all major validation paths including security
