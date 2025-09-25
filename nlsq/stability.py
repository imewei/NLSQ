"""Numerical stability management for NLSQ optimization.

This module provides comprehensive numerical stability monitoring and correction
capabilities for the NLSQ package, ensuring robust optimization even with
ill-conditioned problems or extreme parameter values.
"""

import warnings

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp
from jax import jit


class NumericalStabilityGuard:
    """Comprehensive numerical stability monitoring and correction.

    This class provides methods to detect and correct numerical issues that
    can arise during optimization, including:
    - NaN/Inf detection and correction
    - Ill-conditioning detection and regularization
    - Overflow/underflow protection
    - Safe mathematical operations

    Attributes
    ----------
    eps : float
        Machine epsilon for float64
    max_float : float
        Maximum representable float64 value
    min_float : float
        Minimum positive float64 value
    condition_threshold : float
        Threshold for detecting ill-conditioned matrices
    regularization_factor : float
        Default regularization factor for ill-conditioned problems
    """

    def __init__(self):
        """Initialize stability guard with numerical constants."""
        self.eps = np.finfo(np.float64).eps
        self.max_float = np.finfo(np.float64).max
        self.min_float = np.finfo(np.float64).tiny
        self.condition_threshold = 1e12
        self.regularization_factor = 1e-10
        self.max_exp_arg = 700  # log(max_float) ≈ 709
        self.min_exp_arg = -700

        # Create JIT-compiled versions of key functions
        self._create_jit_functions()

    def _create_jit_functions(self):
        """Create JIT-compiled versions of numerical operations."""

        @jit
        def _safe_exp_jit(x):
            """JIT-compiled safe exponential."""
            x_clipped = jnp.clip(x, self.min_exp_arg, self.max_exp_arg)
            return jnp.exp(x_clipped)

        @jit
        def _safe_log_jit(x):
            """JIT-compiled safe logarithm."""
            x_safe = jnp.maximum(x, self.min_float)
            return jnp.log(x_safe)

        @jit
        def _safe_divide_jit(numerator, denominator):
            """JIT-compiled safe division."""
            safe_denom = jnp.where(
                jnp.abs(denominator) < self.eps, self.eps, denominator
            )
            return numerator / safe_denom

        @jit
        def _safe_sqrt_jit(x):
            """JIT-compiled safe square root."""
            x_safe = jnp.maximum(x, 0.0)
            return jnp.sqrt(x_safe)

        self._safe_exp_jit = _safe_exp_jit
        self._safe_log_jit = _safe_log_jit
        self._safe_divide_jit = _safe_divide_jit
        self._safe_sqrt_jit = _safe_sqrt_jit

    def check_and_fix_jacobian(self, J: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        """Check Jacobian for numerical issues and fix them.

        This method performs several checks and corrections:
        1. Detects and replaces NaN/Inf values
        2. Computes condition number
        3. Applies regularization if ill-conditioned
        4. Checks for near-zero singular values

        Parameters
        ----------
        J : jnp.ndarray
            Jacobian matrix to check and fix

        Returns
        -------
        J_fixed : jnp.ndarray
            Fixed Jacobian matrix
        condition_number : float
            Condition number of the original matrix
        """
        # Check for NaN/Inf
        has_invalid = jnp.any(~jnp.isfinite(J))
        if has_invalid:
            warnings.warn("Jacobian contains NaN or Inf values, replacing with zeros")
            J = jnp.where(jnp.isfinite(J), J, 0.0)

        # Check if matrix is all zeros
        if jnp.allclose(J, 0.0):
            warnings.warn("Jacobian is all zeros, adding small perturbation")
            m, n = J.shape
            J = J + self.eps * jnp.ones((m, n))
            return J, {"has_nan": False, "has_inf": False, "condition_number": np.inf}

        # Compute singular values for condition number
        try:
            svd_vals = jnp.linalg.svdvals(J)

            # Handle empty or invalid SVD
            if len(svd_vals) == 0:
                return J, {
                    "has_nan": False,
                    "has_inf": False,
                    "condition_number": np.inf,
                }

            max_sv = jnp.max(svd_vals)
            min_sv = jnp.min(svd_vals)

            # Compute condition number safely
            if min_sv < self.eps * max_sv:
                condition_number = np.inf
            else:
                condition_number = float(max_sv / min_sv)

        except Exception as e:
            warnings.warn(f"Could not compute SVD for condition number: {e}")
            condition_number = np.inf

        # Apply fixes based on condition number
        if condition_number > self.condition_threshold:
            warnings.warn(
                f"Ill-conditioned Jacobian (condition number: {condition_number:.2e})"
            )

            # Apply Tikhonov regularization without changing dimensions
            # This adds a small diagonal component to improve conditioning
            m, n = J.shape
            # Create a diagonal regularization term
            reg_term = self.regularization_factor * jnp.eye(m, n)
            # Add regularization to the Jacobian directly
            J = J + reg_term

        # Check for near-zero singular values
        if len(svd_vals) > 0:
            min_sv = jnp.min(svd_vals)
            if min_sv < self.eps * 10:
                # Add small diagonal regularization
                m, n = J.shape
                J = J + self.eps * 10 * jnp.eye(m, n)

        issues = {
            "has_nan": bool(has_invalid),
            "has_inf": bool(has_invalid),
            "is_ill_conditioned": condition_number > self.condition_threshold,
            "condition_number": condition_number,
            "regularized": condition_number > self.condition_threshold,
        }
        return J, issues

    def check_parameters(self, params: jnp.ndarray) -> jnp.ndarray:
        """Check and fix parameter values.

        Parameters
        ----------
        params : jnp.ndarray
            Parameter vector to check

        Returns
        -------
        params_fixed : jnp.ndarray
            Fixed parameter vector
        """
        # Check for NaN/Inf
        has_invalid = jnp.any(~jnp.isfinite(params))
        if has_invalid:
            warnings.warn("Parameters contain NaN or Inf values")
            # Replace with reasonable defaults
            params = jnp.where(jnp.isfinite(params), params, 1.0)

        # Check for extreme values
        max_param = jnp.max(jnp.abs(params))
        if max_param > 1e10:
            warnings.warn(f"Parameters have extreme values (max: {max_param:.2e})")
            # Scale down if needed
            params = params / (max_param / 1e10)

        return params

    def safe_exp(self, x: jnp.ndarray) -> jnp.ndarray:
        """Exponential with overflow/underflow protection.

        Parameters
        ----------
        x : jnp.ndarray
            Input array

        Returns
        -------
        result : jnp.ndarray
            exp(x) with values clipped to prevent overflow
        """
        return self._safe_exp_jit(x)

    def safe_log(self, x: jnp.ndarray) -> jnp.ndarray:
        """Logarithm with domain protection.

        Parameters
        ----------
        x : jnp.ndarray
            Input array (must be positive)

        Returns
        -------
        result : jnp.ndarray
            log(x) with values clipped to ensure positive domain
        """
        return self._safe_log_jit(x)

    def safe_divide(
        self, numerator: jnp.ndarray, denominator: jnp.ndarray
    ) -> jnp.ndarray:
        """Division with zero-protection.

        Parameters
        ----------
        numerator : jnp.ndarray
            Numerator array
        denominator : jnp.ndarray
            Denominator array

        Returns
        -------
        result : jnp.ndarray
            numerator/denominator with small values in denominator replaced
        """
        return self._safe_divide_jit(numerator, denominator)

    def safe_sqrt(self, x: jnp.ndarray) -> jnp.ndarray:
        """Square root with domain protection.

        Parameters
        ----------
        x : jnp.ndarray
            Input array

        Returns
        -------
        result : jnp.ndarray
            sqrt(x) with negative values set to 0
        """
        return self._safe_sqrt_jit(x)

    def safe_power(self, base: jnp.ndarray, exponent: float) -> jnp.ndarray:
        """Safe power operation.

        Parameters
        ----------
        base : jnp.ndarray
            Base array
        exponent : float
            Power exponent

        Returns
        -------
        result : jnp.ndarray
            base^exponent with numerical safety
        """
        # Handle negative base with fractional exponent
        if not float(exponent).is_integer():
            base = jnp.abs(base)

        # Prevent overflow
        max_base = (
            jnp.power(self.max_float, 1.0 / abs(exponent)) if exponent != 0 else np.inf
        )
        base_clipped = jnp.clip(base, -max_base, max_base)

        return jnp.power(base_clipped, exponent)

    def check_gradient(self, gradient: jnp.ndarray) -> jnp.ndarray:
        """Check and fix gradient values.

        Parameters
        ----------
        gradient : jnp.ndarray
            Gradient vector

        Returns
        -------
        gradient_fixed : jnp.ndarray
            Fixed gradient with clipping applied if needed
        """
        # Check for NaN/Inf
        if jnp.any(~jnp.isfinite(gradient)):
            warnings.warn("Gradient contains NaN or Inf values")
            gradient = jnp.where(jnp.isfinite(gradient), gradient, 0.0)

        # Apply gradient clipping if needed
        grad_norm = jnp.linalg.norm(gradient)
        max_grad_norm = 1e6

        if grad_norm > max_grad_norm:
            warnings.warn(f"Gradient norm too large ({grad_norm:.2e}), clipping")
            gradient = gradient * (max_grad_norm / grad_norm)

        return gradient

    def regularize_hessian(
        self, H: jnp.ndarray, min_eigenvalue: float = 1e-8
    ) -> jnp.ndarray:
        """Regularize Hessian to ensure positive definiteness.

        Parameters
        ----------
        H : jnp.ndarray
            Hessian or Hessian approximation matrix
        min_eigenvalue : float
            Minimum eigenvalue to ensure

        Returns
        -------
        H_reg : jnp.ndarray
            Regularized Hessian
        """
        n = H.shape[0]

        # Ensure symmetry
        H = 0.5 * (H + H.T)

        try:
            # Check minimum eigenvalue
            eigenvalues = jnp.linalg.eigvalsh(H)
            min_eig = jnp.min(eigenvalues)

            if min_eig < min_eigenvalue:
                # Add diagonal to ensure positive definiteness
                shift = min_eigenvalue - min_eig + self.eps
                H = H + shift * jnp.eye(n)

        except Exception:
            # Fallback: add small diagonal
            H = H + min_eigenvalue * jnp.eye(n)

        return H

    def check_residuals(self, residuals: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
        """Check residuals for numerical issues and outliers.

        Parameters
        ----------
        residuals : jnp.ndarray
            Residual vector

        Returns
        -------
        residuals_fixed : jnp.ndarray
            Fixed residuals
        has_outliers : bool
            Whether outliers were detected
        """
        # Check for NaN/Inf
        if jnp.any(~jnp.isfinite(residuals)):
            warnings.warn("Residuals contain NaN or Inf values")
            residuals = jnp.where(jnp.isfinite(residuals), residuals, 0.0)

        # Detect outliers using MAD (Median Absolute Deviation)
        median_res = jnp.median(residuals)
        mad = jnp.median(jnp.abs(residuals - median_res))

        # Robust standard deviation estimate
        robust_std = 1.4826 * mad

        # Detect outliers (more than 5 robust std from median)
        outlier_mask = jnp.abs(residuals - median_res) > 5 * robust_std
        has_outliers = jnp.any(outlier_mask)

        if has_outliers:
            n_outliers = jnp.sum(outlier_mask)
            warnings.warn(f"Detected {n_outliers} outliers in residuals")

        return residuals, has_outliers

    def safe_norm(self, x: jnp.ndarray, ord: float = 2) -> float:
        """Compute norm with overflow protection.

        Parameters
        ----------
        x : jnp.ndarray
            Input vector or matrix
        ord : float
            Order of the norm

        Returns
        -------
        norm_value : float
            Norm of x with overflow protection
        """
        # Scale if needed to prevent overflow
        max_val = jnp.max(jnp.abs(x))

        if max_val > 1e100:
            # Scale down
            x_scaled = x / max_val
            norm_scaled = jnp.linalg.norm(x_scaled, ord=ord)
            return float(norm_scaled * max_val)
        elif max_val < 1e-100 and max_val > 0:
            # Scale up
            x_scaled = x / max_val
            norm_scaled = jnp.linalg.norm(x_scaled, ord=ord)
            return float(norm_scaled * max_val)
        else:
            return float(jnp.linalg.norm(x, ord=ord))

    def detect_numerical_issues(self, x: jnp.ndarray) -> dict:
        """Detect numerical issues in array.

        Parameters
        ----------
        x : jnp.ndarray
            Array to check

        Returns
        -------
        issues : dict
            Dictionary with keys 'has_nan', 'has_inf', 'has_negative'
        """
        return {
            "has_nan": bool(jnp.any(jnp.isnan(x))),
            "has_inf": bool(jnp.any(jnp.isinf(x))),
            "has_negative": bool(jnp.any(x < 0)) if x.size > 0 else False,
        }


# Create a global instance for convenience
stability_guard = NumericalStabilityGuard()
