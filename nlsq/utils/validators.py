"""Input validation for NLSQ optimization functions.

This module provides comprehensive input validation to catch errors early
and provide helpful error messages to users.
"""

import logging
import warnings
from collections.abc import Callable
from contextlib import suppress
from functools import wraps
from inspect import signature
from typing import Any

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp

from nlsq.constants import DEFAULT_FTOL, DEFAULT_GTOL, DEFAULT_XTOL

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation for curve fitting functions."""

    def __init__(self, fast_mode: bool = True) -> None:
        """Initialize the input validator.

        Parameters
        ----------
        fast_mode : bool, default True
            If True, skip expensive validation checks for better performance.
            If False, perform all validation checks.
        """
        self.fast_mode = fast_mode
        self._function_cache: dict[int, bool] = {}  # Cache function test results

    def _validate_and_convert_arrays(
        self, xdata: Any, ydata: Any
    ) -> tuple[list[str], list[str], Any, Any, int]:
        """Validate and convert xdata/ydata to arrays.

        Parameters
        ----------
        xdata : Any
            Independent variable data
        ydata : Any
            Dependent variable data

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        xdata_converted : Any
            Converted xdata (array or tuple)
        ydata_converted : np.ndarray
            Converted ydata array
        n_points : int
            Number of data points
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        # Handle tuple xdata (for multi-dimensional fitting)
        if isinstance(xdata, tuple):
            try:
                n_points = len(xdata[0]) if len(xdata) > 0 else 0
                # Check all arrays in tuple have same length
                for i, x_arr in enumerate(xdata):
                    if len(x_arr) != n_points:
                        errors.append("All arrays in xdata tuple must have same length")
                        break
                warnings_list.append(f"xdata is tuple with {len(xdata)} arrays")
            except Exception as e:
                errors.append(f"Invalid xdata tuple: {e}")
                return errors, warnings_list, xdata, ydata, 0
        else:
            # Convert to numpy arrays and check types
            try:
                if not isinstance(xdata, (np.ndarray, jnp.ndarray)):
                    xdata = np.asarray(xdata)
                    warnings_list.append("xdata converted to numpy array")
            except Exception as e:
                errors.append(f"Cannot convert xdata to array: {e}")
                return errors, warnings_list, xdata, ydata, 0

            # Check dimensions
            if xdata.ndim == 0:
                errors.append("xdata must be at least 1-dimensional")

            # Handle 2D xdata (multiple independent variables)
            if xdata.ndim == 2:
                n_points = xdata.shape[0]
                n_vars = xdata.shape[1]
                warnings_list.append(f"xdata has {n_vars} independent variables")
            else:
                n_points = len(xdata) if hasattr(xdata, "__len__") else 1

        # Convert and validate ydata
        try:
            if not isinstance(ydata, (np.ndarray, jnp.ndarray)):
                ydata = np.asarray(ydata)
                warnings_list.append("ydata converted to numpy array")
        except Exception as e:
            errors.append(f"Cannot convert ydata to array: {e}")
            return errors, warnings_list, xdata, ydata, n_points

        if ydata.ndim == 0:
            errors.append("ydata must be at least 1-dimensional")

        return errors, warnings_list, xdata, ydata, n_points

    def _estimate_n_params(self, f: Callable, p0: Any | None) -> int:
        """Estimate number of parameters from function signature or p0.

        Parameters
        ----------
        f : Callable
            Model function
        p0 : Any | None
            Initial parameter guess

        Returns
        -------
        n_params : int
            Estimated number of parameters
        """
        n_params = 2  # Default estimate
        try:
            sig = signature(f)
            # Count parameters excluding x
            params = list(sig.parameters.keys())
            if params:
                n_params = len(params) - 1
        except Exception:
            if p0 is not None:
                with suppress(Exception):
                    n_params = len(p0)
        return n_params

    def _validate_data_shapes(
        self, n_points: int, ydata: np.ndarray, n_params: int
    ) -> tuple[list[str], list[str]]:
        """Validate data shapes and minimum requirements.

        Parameters
        ----------
        n_points : int
            Number of data points
        ydata : np.ndarray
            Dependent variable data
        n_params : int
            Number of parameters

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        # Check shapes match
        if len(ydata) != n_points:
            errors.append(
                f"xdata ({n_points} points) and ydata ({len(ydata)} points) must have same length"
            )

        # Check for minimum data points
        if n_points < 2:
            errors.append("Need at least 2 data points for fitting")

        if n_points <= n_params:
            errors.append(
                f"Need more data points ({n_points}) than parameters ({n_params}) for fitting"
            )

        return errors, warnings_list

    def _validate_finite_values(
        self, xdata: Any, ydata: np.ndarray
    ) -> tuple[list[str], list[str]]:
        """Validate that arrays contain only finite values (no NaN/Inf).

        Parameters
        ----------
        xdata : Any
            Independent variable data (array or tuple)
        ydata : np.ndarray
            Dependent variable data

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        # Check xdata for finite values
        if isinstance(xdata, tuple):
            # Check each array in the tuple
            for i, x_arr in enumerate(xdata):
                if not np.all(np.isfinite(x_arr)):
                    n_bad = np.sum(~np.isfinite(x_arr))
                    errors.append(f"xdata[{i}] contains {n_bad} NaN or Inf values")
        elif not np.all(np.isfinite(xdata)):
            n_bad = np.sum(~np.isfinite(xdata))
            errors.append(f"xdata contains {n_bad} NaN or Inf values")

        # Check ydata for finite values
        if not np.all(np.isfinite(ydata)):
            n_bad = np.sum(~np.isfinite(ydata))
            errors.append(f"ydata contains {n_bad} NaN or Inf values")

        return errors, warnings_list

    def _validate_initial_guess(
        self, p0: Any | None, n_params: int
    ) -> tuple[list[str], list[str]]:
        """Validate initial parameter guess.

        Parameters
        ----------
        p0 : Any | None
            Initial parameter guess
        n_params : int
            Expected number of parameters

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        if p0 is None:
            return errors, warnings_list

        try:
            p0 = np.asarray(p0)
            if len(p0) != n_params:
                errors.append(
                    f"Initial guess p0 has {len(p0)} parameters, "
                    f"but function expects {n_params}"
                )

            if not np.all(np.isfinite(p0)):
                errors.append("Initial parameter guess p0 contains NaN or Inf values")

        except Exception as e:
            errors.append(f"Invalid initial parameter guess p0: {e}")

        return errors, warnings_list

    def _validate_bounds(
        self, bounds: tuple | None, n_params: int, p0: Any | None
    ) -> tuple[list[str], list[str]]:
        """Validate parameter bounds.

        Parameters
        ----------
        bounds : tuple | None
            Parameter bounds (lower, upper)
        n_params : int
            Number of parameters
        p0 : Any | None
            Initial parameter guess (to check if within bounds)

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        if bounds is None:
            return errors, warnings_list

        try:
            if len(bounds) != 2:
                errors.append("bounds must be a 2-tuple of (lower, upper)")
            else:
                lb, ub = bounds
                if lb is not None and ub is not None:
                    lb = np.asarray(lb)
                    ub = np.asarray(ub)

                    if len(lb) != n_params or len(ub) != n_params:
                        errors.append(
                            f"bounds must have length {n_params} to match parameters"
                        )

                    if np.any(lb >= ub):
                        errors.append("Lower bounds must be less than upper bounds")

                    # Check if p0 is within bounds
                    if p0 is not None:
                        p0_array = np.asarray(p0)
                        if np.any(p0_array < lb) or np.any(p0_array > ub):
                            warnings_list.append("Initial guess p0 is outside bounds")

        except Exception as e:
            errors.append(f"Invalid bounds: {e}")

        return errors, warnings_list

    def _validate_sigma(
        self, sigma: Any | None, ydata: np.ndarray
    ) -> tuple[list[str], list[str]]:
        """Validate uncertainty (sigma) parameters.

        Parameters
        ----------
        sigma : Any | None
            Uncertainties in ydata
        ydata : np.ndarray
            Dependent variable data

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        if sigma is None:
            return errors, warnings_list

        try:
            sigma = np.asarray(sigma)
            if sigma.shape != ydata.shape:
                errors.append("sigma must have same shape as ydata")

            if np.any(sigma <= 0):
                errors.append("sigma values must be positive")

            if not np.all(np.isfinite(sigma)):
                errors.append("sigma contains NaN or Inf values")

        except Exception as e:
            errors.append(f"Invalid sigma: {e}")

        return errors, warnings_list

    def _check_degenerate_x_values(self, xdata: Any) -> tuple[list[str], list[str]]:
        """Check for degenerate x data (all identical, very small/large range).

        Parameters
        ----------
        xdata : array_like
            Independent variable data

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Only check for non-tuple xdata
        if isinstance(xdata, tuple):
            return errors, warnings

        if not (hasattr(xdata, "ndim") and xdata.ndim == 1 and len(xdata) > 0):
            return errors, warnings

        # Check for all identical values (handle JAX arrays)
        try:
            xdata_first = (
                xdata.flatten()[0] if hasattr(xdata, "flatten") else xdata.flat[0]
            )
            if np.all(xdata == xdata_first):
                errors.append("All x values are identical - cannot fit")
        except (AttributeError, NotImplementedError):
            # Skip if array type doesn't support .flat or .flatten()
            pass

        # Check for very small range
        x_range = np.ptp(xdata)
        if x_range < 1e-10 and x_range > 0:
            warnings.append(
                f"x data range is very small ({x_range:.2e}) - consider rescaling"
            )

        # Check for very large range
        if x_range > 1e10:
            warnings.append(
                f"x data range is very large ({x_range:.2e}) - consider rescaling"
            )

        return errors, warnings

    def _check_degenerate_y_values(self, ydata: np.ndarray) -> list[str]:
        """Check for degenerate y data (all identical, very small range).

        Parameters
        ----------
        ydata : np.ndarray
            Dependent variable data

        Returns
        -------
        warnings : list
            List of warning messages
        """
        warnings: list[str] = []

        # Check if all y values are identical
        try:
            ydata_first = (
                ydata.flatten()[0] if hasattr(ydata, "flatten") else ydata.flat[0]
            )
            if np.all(ydata == ydata_first):
                warnings.append("All y values are identical - trivial fit")
        except Exception as e:
            # Skip this check if it fails - log for debugging
            logger.debug(f"Y-value uniformity check failed (non-critical): {e}")

        # Check for very small range
        y_range = np.ptp(ydata)
        if y_range < 1e-10 and y_range > 0:
            warnings.append(f"y data range is very small ({y_range:.2e})")

        return warnings

    def _check_function_callable(
        self, f: Callable, xdata: Any, ydata: np.ndarray, p0: Any, n_params: int
    ) -> tuple[list[str], list[str]]:
        """Check if function can be called with test data.

        Parameters
        ----------
        f : Callable
            Model function
        xdata : array_like
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        p0 : array_like
            Initial parameters
        n_params : int
            Number of parameters

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Cache function test results to avoid repeated calls
            func_id = id(f)
            if func_id not in self._function_cache:
                if isinstance(xdata, tuple):
                    # For tuple xdata, sample from each array
                    test_x = tuple(arr[: min(10, len(arr))] for arr in xdata)
                    expected_len = min(10, len(xdata[0]))
                else:
                    if hasattr(xdata, "ndim") and xdata.ndim > 1:
                        test_x = xdata[: min(10, len(xdata))]
                    else:
                        test_x = xdata[: min(10, len(xdata))]
                    expected_len = min(10, len(xdata))

                if p0 is not None:
                    test_result = f(test_x, *p0)
                else:
                    # Try with dummy parameters
                    dummy_params = np.ones(n_params)
                    test_result = f(test_x, *dummy_params)

                # Cache the result
                self._function_cache[func_id] = True

                # Check output shape/length
                if hasattr(test_result, "__len__"):
                    if len(test_result) != expected_len:
                        warnings.append(
                            f"Function output length {len(test_result)} doesn't match "
                            f"expected length {expected_len}"
                        )

        except Exception as e:
            errors.append(f"Cannot evaluate function: {e}")

        return errors, warnings

    def _check_data_quality(self, xdata: Any, ydata: np.ndarray) -> list[str]:
        """Check data quality (duplicates, outliers).

        Parameters
        ----------
        xdata : array_like
            Independent variable data
        ydata : np.ndarray
            Dependent variable data

        Returns
        -------
        warnings : list
            List of warning messages
        """
        warnings: list[str] = []

        # Check for duplicates in x
        if not isinstance(xdata, tuple) and hasattr(xdata, "ndim") and xdata.ndim == 1:
            unique_x = np.unique(xdata)
            if len(unique_x) < len(xdata):
                n_dup = len(xdata) - len(unique_x)
                warnings.append(f"xdata contains {n_dup} duplicate values")

        # Check for outliers in y
        if len(ydata) > 10:
            q1, q3 = np.percentile(ydata, [25, 75])
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            n_outliers = np.sum((ydata < lower) | (ydata > upper))
            if n_outliers > 0:
                warnings.append(
                    f"ydata may contain {n_outliers} outliers - "
                    "consider using robust loss function"
                )

        return warnings

    def validate_curve_fit_inputs(
        self,
        f: Callable,
        xdata: Any,
        ydata: Any,
        p0: Any | None = None,
        bounds: tuple | None = None,
        sigma: Any | None = None,
        absolute_sigma: bool = True,
        check_finite: bool = True,
    ) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
        """Validate inputs for curve_fit function.

        This method orchestrates the validation pipeline by calling focused
        helper methods for each validation step.

        Parameters
        ----------
        f : callable
            Model function to fit
        xdata : array_like
            Independent variable data
        ydata : array_like
            Dependent variable data
        p0 : array_like, optional
            Initial parameter guess
        bounds : tuple, optional
            Parameter bounds
        sigma : array_like, optional
            Uncertainties in ydata
        absolute_sigma : bool
            Whether sigma is absolute or relative
        check_finite : bool
            Whether to check for finite values

        Returns
        -------
        errors : list
            List of error messages (empty if no errors)
        warnings : list
            List of warning messages
        xdata_clean : np.ndarray
            Cleaned and validated xdata
        ydata_clean : np.ndarray
            Cleaned and validated ydata
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        # Step 1: Validate and convert arrays
        arr_errors, arr_warnings, xdata, ydata, n_points = (
            self._validate_and_convert_arrays(xdata, ydata)
        )
        errors.extend(arr_errors)
        warnings_list.extend(arr_warnings)
        if errors:
            return errors, warnings_list, xdata, ydata

        # Step 2: Estimate number of parameters
        n_params = self._estimate_n_params(f, p0)

        # Step 2.5: Security validation (array size limits, bounds ranges)
        security_errors, security_warnings = self.validate_security_constraints(
            n_points, n_params, bounds, p0
        )
        errors.extend(security_errors)
        warnings_list.extend(security_warnings)
        if security_errors:
            # Return early on critical security errors
            return errors, warnings_list, xdata, ydata

        # Step 3: Validate data shapes
        shape_errors, shape_warnings = self._validate_data_shapes(
            n_points, ydata, n_params
        )
        errors.extend(shape_errors)
        warnings_list.extend(shape_warnings)

        # Step 4: Check for degenerate x values
        deg_x_errors, deg_x_warnings = self._check_degenerate_x_values(xdata)
        errors.extend(deg_x_errors)
        warnings_list.extend(deg_x_warnings)

        # Step 5: Check for degenerate y values
        deg_y_warnings = self._check_degenerate_y_values(ydata)
        warnings_list.extend(deg_y_warnings)

        # Step 6: Validate finite values if requested
        if check_finite:
            finite_errors, finite_warnings = self._validate_finite_values(xdata, ydata)
            errors.extend(finite_errors)
            warnings_list.extend(finite_warnings)

        # Step 7: Validate initial parameters
        p0_errors, p0_warnings = self._validate_initial_guess(p0, n_params)
        errors.extend(p0_errors)
        warnings_list.extend(p0_warnings)

        # Step 8: Validate bounds if provided
        bounds_errors, bounds_warnings = self._validate_bounds(bounds, n_params, p0)
        errors.extend(bounds_errors)
        warnings_list.extend(bounds_warnings)

        # Step 9: Validate sigma if provided
        sigma_errors, sigma_warnings = self._validate_sigma(sigma, ydata)
        errors.extend(sigma_errors)
        warnings_list.extend(sigma_warnings)

        # Step 10: Check function can be called (skip in fast mode)
        if not self.fast_mode:
            func_errors, func_warnings = self._check_function_callable(
                f, xdata, ydata, p0, n_params
            )
            errors.extend(func_errors)
            warnings_list.extend(func_warnings)

        # Step 11: Data quality checks (skip in fast mode)
        if not self.fast_mode:
            quality_warnings = self._check_data_quality(xdata, ydata)
            warnings_list.extend(quality_warnings)

        # Return cleaned data
        # Keep tuples as tuples, convert arrays to numpy
        if not isinstance(xdata, tuple):
            xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        return errors, warnings_list, xdata, ydata

    def _validate_x0_array(self, x0: Any) -> tuple[list[str], np.ndarray]:
        """Validate and convert x0 to array.

        Parameters
        ----------
        x0 : array_like
            Initial parameter guess

        Returns
        -------
        errors : list
            List of error messages
        x0 : np.ndarray
            Converted x0 array
        """
        errors: list[str] = []

        # Convert x0
        try:
            x0 = np.asarray(x0)
        except Exception as e:
            errors.append(f"Cannot convert x0 to array: {e}")
            return errors, x0

        # Check x0 dimensions and values
        if x0.ndim != 1:
            errors.append("x0 must be 1-dimensional")

        if len(x0) == 0:
            errors.append("x0 cannot be empty")

        if not np.all(np.isfinite(x0)):
            errors.append("x0 contains NaN or Inf values")

        return errors, x0

    def _validate_method(self, method: str) -> list[str]:
        """Validate optimization method.

        Parameters
        ----------
        method : str
            Optimization method

        Returns
        -------
        errors : list
            List of error messages
        """
        errors: list[str] = []
        valid_methods = ["trf", "dogbox", "lm"]
        if method not in valid_methods:
            errors.append(f"method must be one of {valid_methods}, got {method}")
        return errors

    def _validate_tolerances(
        self, ftol: float, xtol: float, gtol: float
    ) -> tuple[list[str], list[str]]:
        """Validate convergence tolerances.

        Parameters
        ----------
        ftol : float
            Function tolerance
        xtol : float
            Parameter tolerance
        gtol : float
            Gradient tolerance

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check positive
        if ftol <= 0:
            errors.append(f"ftol must be positive, got {ftol}")
        if xtol <= 0:
            errors.append(f"xtol must be positive, got {xtol}")
        if gtol <= 0:
            errors.append(f"gtol must be positive, got {gtol}")

        # Check very small values
        if ftol < 1e-15:
            warnings.append(f"ftol={ftol} is very small, may not converge")
        if xtol < 1e-15:
            warnings.append(f"xtol={xtol} is very small, may not converge")

        return errors, warnings

    def _validate_max_nfev(
        self, max_nfev: int | None, n_params: int
    ) -> tuple[list[str], list[str]]:
        """Validate maximum function evaluations.

        Parameters
        ----------
        max_nfev : int or None
            Maximum function evaluations
        n_params : int
            Number of parameters

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings: list[str] = []

        if max_nfev is not None:
            if max_nfev <= 0:
                errors.append(f"max_nfev must be positive, got {max_nfev}")
            elif max_nfev < n_params:
                warnings.append(
                    f"max_nfev={max_nfev} is less than number of parameters {n_params}"
                )

        return errors, warnings

    def _validate_bounds_and_x0(
        self, bounds: tuple | None, x0: np.ndarray, method: str
    ) -> list[str]:
        """Validate bounds and check x0 within bounds.

        Parameters
        ----------
        bounds : tuple or None
            Parameter bounds as (lb, ub)
        x0 : np.ndarray
            Initial parameter guess
        method : str
            Optimization method

        Returns
        -------
        errors : list
            List of error messages
        """
        errors: list[str] = []

        if bounds is None:
            return errors

        # Check method compatibility
        if method == "lm":
            errors.append("Levenberg-Marquardt method does not support bounds")
            return errors

        # Validate bounds structure
        try:
            lb, ub = bounds
            lb = np.asarray(lb)
            ub = np.asarray(ub)

            if len(lb) != len(x0) or len(ub) != len(x0):
                errors.append("bounds must have same length as x0")

            if np.any(lb >= ub):
                errors.append("Lower bounds must be less than upper bounds")

            # Check x0 within bounds
            if np.any(x0 < lb) or np.any(x0 > ub):
                errors.append("Initial guess x0 is outside bounds")

        except Exception as e:
            errors.append(f"Invalid bounds: {e}")

        return errors

    def _validate_function_at_x0(
        self, fun: Callable, x0: np.ndarray
    ) -> tuple[list[str], list[str]]:
        """Validate function can be evaluated at x0.

        Parameters
        ----------
        fun : callable
            Residual function
        x0 : np.ndarray
            Initial parameter guess

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings: list[str] = []

        try:
            result = fun(x0)
            result = np.asarray(result)

            if result.ndim != 1:
                errors.append("Function must return 1-dimensional residuals")

            if not np.all(np.isfinite(result)):
                warnings.append("Function returns NaN or Inf at initial guess")

        except Exception as e:
            errors.append(f"Cannot evaluate function at x0: {e}")

        return errors, warnings

    # =========================================================================
    # Security-focused validation methods
    # =========================================================================

    def _validate_array_size_limits(
        self,
        n_points: int,
        n_params: int,
        max_data_points: int = 10_000_000_000,  # 10 billion
        max_jacobian_elements: int = 100_000_000_000,  # 100 billion
    ) -> tuple[list[str], list[str]]:
        """Validate array sizes to prevent memory exhaustion and integer overflow.

        This is a security-focused check to prevent denial-of-service via
        excessive memory allocation or integer overflow in Jacobian computation.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters
        max_data_points : int, default 10_000_000_000
            Maximum allowed data points (10 billion)
        max_jacobian_elements : int, default 100_000_000_000
            Maximum allowed Jacobian elements (100 billion)

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        # Check data point limit
        if n_points > max_data_points:
            errors.append(
                f"Dataset size ({n_points:,} points) exceeds maximum allowed "
                f"({max_data_points:,} points). This limit prevents memory exhaustion."
            )

        if n_points < 0:
            errors.append(f"Invalid negative data point count: {n_points}")

        if n_params < 0:
            errors.append(f"Invalid negative parameter count: {n_params}")

        # Check for potential integer overflow in Jacobian size calculation
        # Jacobian has shape (n_points, n_params)
        try:
            jacobian_elements = n_points * n_params
            if jacobian_elements > max_jacobian_elements:
                errors.append(
                    f"Jacobian size ({n_points:,} x {n_params:,} = {jacobian_elements:,} elements) "
                    f"exceeds maximum allowed ({max_jacobian_elements:,} elements). "
                    "Consider using streaming optimization or reducing dataset size."
                )
        except OverflowError:
            errors.append(
                f"Integer overflow computing Jacobian size: {n_points} x {n_params}. "
                "Dataset is too large."
            )

        # Memory estimation warning (assuming float64)
        estimated_memory_gb = (n_points * n_params * 8) / (1024**3)
        if estimated_memory_gb > 100:
            warnings_list.append(
                f"Estimated Jacobian memory usage: {estimated_memory_gb:.1f} GB. "
                "Consider using streaming optimization."
            )
        elif estimated_memory_gb > 10:
            warnings_list.append(
                f"Large Jacobian estimated at {estimated_memory_gb:.1f} GB memory."
            )

        return errors, warnings_list

    def _validate_bounds_numeric_range(
        self,
        bounds: tuple | None,
        max_bound_magnitude: float = 1e100,
    ) -> tuple[list[str], list[str]]:
        """Validate that bounds are within reasonable numeric ranges.

        This prevents numerical issues from extreme bound values that could
        cause overflow or underflow during optimization.

        Parameters
        ----------
        bounds : tuple | None
            Parameter bounds as (lb, ub)
        max_bound_magnitude : float, default 1e100
            Maximum allowed absolute value for bounds

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        if bounds is None:
            return errors, warnings_list

        try:
            lb, ub = bounds

            # Convert to arrays for checking
            if lb is not None:
                lb_arr = np.asarray(lb)
                # Check for extreme values (excluding inf which is valid)
                finite_lb = lb_arr[np.isfinite(lb_arr)]
                if len(finite_lb) > 0 and np.any(
                    np.abs(finite_lb) > max_bound_magnitude
                ):
                    warnings_list.append(
                        f"Lower bounds contain very large values (|lb| > {max_bound_magnitude:.0e}). "
                        "This may cause numerical issues."
                    )

            if ub is not None:
                ub_arr = np.asarray(ub)
                finite_ub = ub_arr[np.isfinite(ub_arr)]
                if len(finite_ub) > 0 and np.any(
                    np.abs(finite_ub) > max_bound_magnitude
                ):
                    warnings_list.append(
                        f"Upper bounds contain very large values (|ub| > {max_bound_magnitude:.0e}). "
                        "This may cause numerical issues."
                    )

            # Check for NaN in bounds (invalid)
            if lb is not None and np.any(np.isnan(np.asarray(lb))):
                errors.append("Lower bounds contain NaN values")
            if ub is not None and np.any(np.isnan(np.asarray(ub))):
                errors.append("Upper bounds contain NaN values")

        except Exception as e:
            errors.append(f"Error validating bounds numeric range: {e}")

        return errors, warnings_list

    def _validate_parameter_values(
        self,
        p0: Any | None,
        max_param_magnitude: float = 1e50,
    ) -> tuple[list[str], list[str]]:
        """Validate that initial parameters are within reasonable numeric ranges.

        This prevents numerical issues from extreme parameter values that could
        cause overflow during function evaluation.

        Parameters
        ----------
        p0 : array_like | None
            Initial parameter guess
        max_param_magnitude : float, default 1e50
            Maximum allowed absolute value for parameters

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        if p0 is None:
            return errors, warnings_list

        try:
            p0_arr = np.asarray(p0)

            # Check for extreme values
            if np.any(np.abs(p0_arr) > max_param_magnitude):
                max_val = np.max(np.abs(p0_arr))
                warnings_list.append(
                    f"Initial parameters contain very large values (max |p0| = {max_val:.2e}). "
                    "This may cause numerical overflow."
                )

            # Check for subnormal values that might cause underflow
            finite_p0 = p0_arr[np.isfinite(p0_arr) & (p0_arr != 0)]
            if len(finite_p0) > 0:
                min_abs = np.min(np.abs(finite_p0))
                if min_abs < 1e-300:
                    warnings_list.append(
                        f"Initial parameters contain very small values (min |p0| = {min_abs:.2e}). "
                        "This may cause numerical underflow."
                    )

        except Exception as e:
            errors.append(f"Error validating parameter values: {e}")

        return errors, warnings_list

    def validate_security_constraints(
        self,
        n_points: int,
        n_params: int,
        bounds: tuple | None = None,
        p0: Any | None = None,
    ) -> tuple[list[str], list[str]]:
        """Validate security constraints to prevent DoS and numerical issues.

        This method combines all security-focused validation checks.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters
        bounds : tuple | None, optional
            Parameter bounds
        p0 : array_like | None, optional
            Initial parameter guess

        Returns
        -------
        errors : list
            List of critical error messages
        warnings : list
            List of warning messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        # Check array size limits
        size_errors, size_warnings = self._validate_array_size_limits(
            n_points, n_params
        )
        errors.extend(size_errors)
        warnings_list.extend(size_warnings)

        # Check bounds numeric range
        bounds_errors, bounds_warnings = self._validate_bounds_numeric_range(bounds)
        errors.extend(bounds_errors)
        warnings_list.extend(bounds_warnings)

        # Check parameter value range
        param_errors, param_warnings = self._validate_parameter_values(p0)
        errors.extend(param_errors)
        warnings_list.extend(param_warnings)

        return errors, warnings_list

    def validate_least_squares_inputs(
        self,
        fun: Callable,
        x0: Any,
        bounds: tuple | None = None,
        method: str = "trf",
        ftol: float = DEFAULT_FTOL,
        xtol: float = DEFAULT_XTOL,
        gtol: float = DEFAULT_GTOL,
        max_nfev: int | None = None,
    ) -> tuple[list[str], list[str], np.ndarray]:
        """Validate inputs for least_squares function.

        This method orchestrates the validation pipeline by calling focused
        helper methods for each validation step.

        Parameters
        ----------
        fun : callable
            Residual function
        x0 : array_like
            Initial parameter guess
        bounds : tuple, optional
            Parameter bounds
        method : str
            Optimization method
        ftol : float
            Function tolerance
        xtol : float
            Parameter tolerance
        gtol : float
            Gradient tolerance
        max_nfev : int, optional
            Maximum function evaluations

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        x0_clean : np.ndarray
            Cleaned initial guess
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        # Step 1: Validate and convert x0
        x0_errors, x0 = self._validate_x0_array(x0)
        errors.extend(x0_errors)
        if x0_errors:
            return errors, warnings_list, x0

        # Step 2: Validate method
        method_errors = self._validate_method(method)
        errors.extend(method_errors)

        # Step 3: Validate tolerances
        tol_errors, tol_warnings = self._validate_tolerances(ftol, xtol, gtol)
        errors.extend(tol_errors)
        warnings_list.extend(tol_warnings)

        # Step 4: Validate max_nfev
        nfev_errors, nfev_warnings = self._validate_max_nfev(max_nfev, len(x0))
        errors.extend(nfev_errors)
        warnings_list.extend(nfev_warnings)

        # Step 5: Validate bounds and check x0 within bounds
        bounds_errors = self._validate_bounds_and_x0(bounds, x0, method)
        errors.extend(bounds_errors)

        # Step 6: Validate function can be called at x0
        func_errors, func_warnings = self._validate_function_at_x0(fun, x0)
        errors.extend(func_errors)
        warnings_list.extend(func_warnings)

        return errors, warnings_list, x0


def validate_inputs(validation_type: str = "curve_fit") -> Callable:
    """Decorator for automatic input validation.

    Parameters
    ----------
    validation_type : str
        Type of validation to perform ('curve_fit' or 'least_squares')

    Returns
    -------
    decorator : function
        Decorator function that validates inputs
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            validator = InputValidator()

            if validation_type == "curve_fit":
                # Extract arguments
                if len(args) < 3:
                    raise ValueError(
                        "curve_fit requires at least 3 arguments (f, xdata, ydata)"
                    )

                f, xdata, ydata = args[:3]
                remaining_args = args[3:]

                # Get optional arguments
                p0 = kwargs.get("p0")
                bounds = kwargs.get("bounds")
                sigma = kwargs.get("sigma")
                absolute_sigma = kwargs.get("absolute_sigma", True)
                check_finite = kwargs.get("check_finite", True)

                # Validate
                errors, warnings_list, xdata_clean, ydata_clean = (
                    validator.validate_curve_fit_inputs(
                        f, xdata, ydata, p0, bounds, sigma, absolute_sigma, check_finite
                    )
                )

                # Handle errors and warnings
                if errors:
                    raise ValueError(f"Input validation failed: {'; '.join(errors)}")

                for warning in warnings_list:
                    warnings.warn(warning, UserWarning, stacklevel=2)

                # Replace with cleaned data
                args = (f, xdata_clean, ydata_clean, *remaining_args)

            elif validation_type == "least_squares":
                # Extract arguments
                if len(args) < 2:
                    raise ValueError(
                        "least_squares requires at least 2 arguments (fun, x0)"
                    )

                fun, x0 = args[:2]
                remaining_args = args[2:]

                # Get optional arguments
                bounds = kwargs.get("bounds")
                method = kwargs.get("method", "trf")
                ftol = kwargs.get("ftol", 1e-8)
                xtol = kwargs.get("xtol", 1e-8)
                gtol = kwargs.get("gtol", 1e-8)
                max_nfev = kwargs.get("max_nfev")

                # Validate
                errors, warnings_list, x0_clean = (
                    validator.validate_least_squares_inputs(
                        fun, x0, bounds, method, ftol, xtol, gtol, max_nfev
                    )
                )

                # Handle errors and warnings
                if errors:
                    raise ValueError(f"Input validation failed: {'; '.join(errors)}")

                for warning in warnings_list:
                    warnings.warn(warning, UserWarning, stacklevel=2)

                # Replace with cleaned data
                args = (fun, x0_clean, *remaining_args)

            else:
                raise ValueError(f"Unknown validation type: {validation_type}")

            # Call original function
            return func(*args, **kwargs)

        return wrapper

    return decorator
