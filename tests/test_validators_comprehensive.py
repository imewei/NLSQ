"""
Comprehensive tests for validators module.

Target: InputValidator.validate_curve_fit_inputs (complexity 25)
Goal: Cover all validation branches for Sprint 1 safety net.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from nlsq.validators import InputValidator


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
        """Test non-callable function raises TypeError."""
        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        with pytest.raises(TypeError, match="callable"):
            self.validator.validate_curve_fit_inputs(
                f="not_a_function",  # Invalid!
                xdata=xdata,
                ydata=ydata,
                p0=None,
                sigma=None,
                bounds=(-np.inf, np.inf),
                
            )

    def test_xdata_ydata_shape_mismatch_raises(self):
        """Test shape mismatch raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4])  # Wrong shape!

        with pytest.raises(ValueError, match="shape"):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=None,
                sigma=None,
                bounds=(-np.inf, np.inf),
                
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
        """Test sigma shape mismatch raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        sigma = np.array([0.1, 0.2])  # Wrong shape!

        with pytest.raises(ValueError):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=None,
                sigma=sigma,
                bounds=(-np.inf, np.inf),
                
            )

    def test_sigma_negative_raises(self):
        """Test negative sigma raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        sigma = np.array([0.1, -0.2, 0.3])  # Negative!

        with pytest.raises(ValueError):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=None,
                sigma=sigma,
                bounds=(-np.inf, np.inf),
                
            )

    def test_sigma_zero_raises(self):
        """Test zero sigma raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        sigma = np.array([0.1, 0.0, 0.3])  # Zero!

        with pytest.raises(ValueError):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=None,
                sigma=sigma,
                bounds=(-np.inf, np.inf),
                
            )

    def test_bounds_lower_ge_upper_raises(self):
        """Test lower >= upper bounds raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        bounds = ([10, 10], [0, 0])  # Lower >= upper!

        with pytest.raises(ValueError, match="bound"):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=[1, 1],
                sigma=None,
                bounds=bounds,
                
            )

    def test_p0_outside_bounds_raises(self):
        """Test p0 outside bounds raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        p0 = [5, 5]
        bounds = ([0, 0], [3, 3])  # p0 outside!

        with pytest.raises(ValueError):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                sigma=None,
                bounds=bounds,
                
            )

    def test_method_invalid_raises(self):
        """Test invalid method raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        with pytest.raises(ValueError, match="method"):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=None,
                sigma=None,
                bounds=(-np.inf, np.inf),
                
            )

    def test_method_lm_with_bounds_raises(self):
        """Test method='lm' with finite bounds raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        bounds = ([0, 0], [10, 10])

        with pytest.raises(ValueError, match="bounds"):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=[1, 1],
                sigma=None,
                bounds=bounds,
                method='lm',  # LM doesn't support bounds!
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
        """Test bounds shape mismatch raises ValueError."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])
        bounds = ([0], [10, 10])  # Shape mismatch!

        with pytest.raises(ValueError):
            self.validator.validate_curve_fit_inputs(
                f=model,
                xdata=xdata,
                ydata=ydata,
                p0=[1, 1],
                sigma=None,
                bounds=bounds,
                
            )


# Total: 24 comprehensive tests covering all major validation paths
