"""Numerical stability management for NLSQ optimization.

This module provides comprehensive numerical stability monitoring and correction
capabilities for the NLSQ package, ensuring robust optimization even with
ill-conditioned problems or extreme parameter values.

Stability Modes
===============

The stability parameter in curve_fit() controls numerical stability behavior:

- ``stability=False`` (default): No stability checks. Maximum performance.
- ``stability='check'``: Check for issues and warn, but don't modify data.
- ``stability='auto'``: Automatically detect and fix numerical issues.

Key Design Decisions (v0.3.0)
=============================

1. **Initialization-only Jacobian checks**: Stability checks on the Jacobian are
   performed only at optimization initialization, not per-iteration. Per-iteration
   Jacobian modification was found to cause optimization divergence due to
   accumulated numerical perturbations. (commit 8028a03)

2. **SVD skip for large Jacobians**: For Jacobians exceeding max_jacobian_elements_for_svd
   (default 10M elements), SVD computation is skipped to avoid O(min(m,n)^2 * max(m,n))
   overhead. Only O(n) NaN/Inf checking is performed.

3. **rescale_data parameter**: Physics applications (XPCS, scattering, etc.) can
   set rescale_data=False to preserve physical units. The default (True) rescales
   data to [0, 1] when ill-conditioning is detected.

Module Constants
================

MAX_JACOBIAN_ELEMENTS_FOR_SVD : int = 10_000_000
    Default maximum Jacobian elements for SVD computation. For Jacobians
    exceeding this threshold, SVD is skipped to avoid excessive computation.

    Performance characteristics:
    - 10M element Jacobian: ~1.5GB RAM, ~5 seconds (GPU)
    - SVD per iteration can exceed total optimization time
    - For XPCS applications (1M points x 3 params = 3M), SVD is computed
    - For large datasets (10M points x 3 params = 30M), SVD is skipped

CONDITION_THRESHOLD : float = 1e12
    Threshold for detecting ill-conditioned matrices. Condition numbers
    above this trigger regularization warnings or automatic fixes.

REGULARIZATION_FACTOR : float = 1e-10
    Default Tikhonov regularization factor for ill-conditioned problems.

See Also
--------
NumericalStabilityGuard : Main class for stability operations
apply_automatic_fixes : Function to apply stability fixes to data
check_problem_stability : Pre-flight stability check for optimization
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax
import jax.numpy as jnp
from jax import jit, lax

# Module-level cached constants (avoid repeated np.finfo lookups)
_FLOAT64_INFO = np.finfo(np.float64)
_EPS = _FLOAT64_INFO.eps
_MAX_FLOAT = _FLOAT64_INFO.max
_MIN_FLOAT = _FLOAT64_INFO.tiny

__all__ = [
    "NumericalStabilityGuard",
    "apply_automatic_fixes",
    "check_problem_stability",
    "detect_collinearity",
    "detect_parameter_scale_mismatch",
    "estimate_condition_number",
    "solve_with_cholesky_fallback",
    "stability_guard",
]


def solve_with_cholesky_fallback(
    A: jnp.ndarray, b: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve linear system Ax = b using Cholesky with eigenvalue fallback.

    This function attempts Cholesky decomposition first (O(n³/3)) and falls back
    to eigenvalue decomposition (O(n³)) only if Cholesky fails. The fallback
    is detected via NaN check, making this function JAX JIT-compatible.

    Parameters
    ----------
    A : jnp.ndarray
        Symmetric matrix (n × n). Should be positive definite for Cholesky to succeed.
    b : jnp.ndarray
        Right-hand side vector (n,).

    Returns
    -------
    x : jnp.ndarray
        Solution vector (n,).
    used_cholesky : jnp.ndarray
        Boolean mask indicating whether Cholesky was used (True) or the
        eigenvalue fallback was needed (False).

    Notes
    -----
    For positive definite matrices (e.g., J^T J + λI with λ > 0), Cholesky is
    approximately 3x faster than eigenvalue decomposition. The JAX-compatible
    fallback pattern uses NaN detection instead of Python try/except.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[4.0, 2.0], [2.0, 3.0]])  # Positive definite
    >>> b = jnp.array([1.0, 2.0])
    >>> x, used_cholesky = solve_with_cholesky_fallback(A, b)
    >>> used_cholesky
    True
    """
    # Ensure symmetry
    A = 0.5 * (A + A.T)

    # Try Cholesky decomposition
    L = jnp.linalg.cholesky(A)
    x_chol = jax.scipy.linalg.cho_solve((L, True), b)

    # Check if Cholesky succeeded (no NaNs)
    cholesky_valid = jnp.all(jnp.isfinite(x_chol))

    # Eigenvalue fallback
    def eigenvalue_solve():
        eigvals, eigvecs = jnp.linalg.eigh(A)
        # Regularize small/negative eigenvalues
        eps = jnp.finfo(A.dtype).eps * jnp.max(jnp.abs(eigvals))
        eigvals_safe = jnp.maximum(eigvals, eps)
        return eigvecs @ (eigvecs.T @ b / eigvals_safe)

    # Use lax.cond for JIT-compatible conditional
    x = lax.cond(cholesky_valid, lambda: x_chol, eigenvalue_solve)

    return x, cholesky_valid


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

    __slots__ = (
        "_check_gradient_jit",
        "_check_jacobian_fast_jit",
        "_safe_divide_jit",
        "_safe_exp_jit",
        "_safe_log_jit",
        "_safe_norm_jit",
        "_safe_sqrt_jit",
        "condition_threshold",
        "eps",
        "max_exp_arg",
        "max_float",
        "max_jacobian_elements_for_svd",
        "min_exp_arg",
        "min_float",
        "regularization_factor",
    )

    def __init__(self, max_jacobian_elements_for_svd: int = 10_000_000):
        """Initialize stability guard with numerical constants.

        Parameters
        ----------
        max_jacobian_elements_for_svd : int, default 10_000_000
            Maximum number of elements in Jacobian (m × n) for which SVD
            will be computed. For larger Jacobians, SVD is skipped to avoid
            excessive computation time. Default is 10M elements, which
            corresponds to e.g., a (1M × 10) or (100K × 100) matrix.
        """
        # Use cached module-level constants for fast initialization
        self.eps = _EPS
        self.max_float = _MAX_FLOAT
        self.min_float = _MIN_FLOAT
        self.condition_threshold = 1e12
        self.regularization_factor = 1e-10
        self.max_exp_arg = 700  # log(max_float) ≈ 709
        self.min_exp_arg = -700
        self.max_jacobian_elements_for_svd = max_jacobian_elements_for_svd

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

        # JIT-compiled fast Jacobian check (for large Jacobians - NaN/Inf only)
        @jit
        def _check_jacobian_fast_jit(J):
            """Fast NaN/Inf check for large Jacobians."""
            has_invalid = jnp.any(~jnp.isfinite(J))
            J_fixed = jnp.where(jnp.isfinite(J), J, 0.0)
            return J_fixed, has_invalid

        # JIT-compiled gradient checking
        @jit
        def _check_gradient_jit(gradient, max_grad_norm):
            """JIT-compiled gradient checking."""
            is_finite_mask = jnp.isfinite(gradient)
            gradient_clean = jnp.where(is_finite_mask, gradient, 0.0)
            grad_norm = jnp.linalg.norm(gradient_clean)
            needs_clipping = grad_norm > max_grad_norm
            gradient_fixed = jnp.where(
                needs_clipping,
                gradient_clean * (max_grad_norm / grad_norm),
                gradient_clean,
            )
            has_invalid = jnp.any(~is_finite_mask)
            return gradient_fixed, has_invalid, needs_clipping, grad_norm

        # JIT-compiled safe norm
        @jit
        def _safe_norm_jit(x, scale_factor):
            """JIT-compiled safe norm with pre-computed scale."""
            x_scaled = x / scale_factor
            return jnp.linalg.norm(x_scaled) * scale_factor

        self._check_jacobian_fast_jit = _check_jacobian_fast_jit
        self._check_gradient_jit = _check_gradient_jit
        self._safe_norm_jit = _safe_norm_jit

    def check_and_fix_jacobian(self, J: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        """Check Jacobian for numerical issues and fix them.

        This method performs several checks and corrections:
        1. Detects and replaces NaN/Inf values
        2. Computes condition number (skipped for large Jacobians)
        3. Applies regularization if ill-conditioned
        4. Checks for near-zero singular values

        For large Jacobians (> max_jacobian_elements_for_svd elements), SVD
        computation is skipped to avoid excessive computation time. Only NaN/Inf
        checking is performed.

        Parameters
        ----------
        J : jnp.ndarray
            Jacobian matrix to check and fix

        Returns
        -------
        J_fixed : jnp.ndarray
            Fixed Jacobian matrix
        issues : dict
            Dictionary with 'has_nan', 'has_inf', 'condition_number', etc.
        """
        m, n = J.shape
        n_elements = m * n

        # Skip SVD for large Jacobians to avoid excessive computation time
        # SVD of (m, n) matrix is O(min(m,n)^2 * max(m,n)) which is very expensive
        # for large matrices (e.g., 10^6 x 7 = 7M elements)
        if n_elements > self.max_jacobian_elements_for_svd:
            # Fast path: JIT-compiled NaN/Inf check only (10-50x faster)
            J_fixed, has_invalid = self._check_jacobian_fast_jit(J)
            if has_invalid:
                warnings.warn(
                    "Jacobian contains NaN or Inf values, replacing with zeros"
                )
            issues: dict[str, object] = {
                "has_nan": bool(has_invalid),
                "has_inf": bool(has_invalid),
                "is_ill_conditioned": False,
                "condition_number": None,
                "regularized": False,
                "svd_skipped": True,
                "reason": f"Jacobian too large ({n_elements:,} > {self.max_jacobian_elements_for_svd:,} elements)",
            }
            return J_fixed, issues

        # Standard path: Check for NaN/Inf
        has_invalid = jnp.any(~jnp.isfinite(J))
        if has_invalid:
            warnings.warn("Jacobian contains NaN or Inf values, replacing with zeros")
            J = jnp.where(jnp.isfinite(J), J, 0.0)

        # Check if matrix is all zeros - use broadcast add (more memory efficient)
        if jnp.allclose(J, 0.0):
            warnings.warn("Jacobian is all zeros, adding small perturbation")
            J = J + self.eps  # Broadcasting is more memory efficient than jnp.ones
            return J, {"has_nan": False, "has_inf": False, "condition_number": np.inf}

        # Compute singular values for condition number
        condition_number = np.inf
        min_sv_cached = None  # Cache min singular value (avoid recomputation)
        try:
            svd_vals = jnp.linalg.svdvals(J)

            # Handle empty or invalid SVD
            if len(svd_vals) == 0:
                return J, {
                    "has_nan": False,
                    "has_inf": False,
                    "condition_number": np.inf,
                }

            # svdvals returns sorted descending, use direct indexing O(1)
            max_sv = svd_vals[0]
            min_sv_cached = svd_vals[-1]

            # Compute condition number safely
            if min_sv_cached < self.eps * max_sv:
                condition_number = np.inf
            else:
                condition_number = float(max_sv / min_sv_cached)

        except Exception as e:
            warnings.warn(f"Could not compute SVD for condition number: {e}")
            condition_number = np.inf

        # Apply fixes based on condition number
        regularized = False
        k = min(m, n)  # Diagonal size
        diag_indices = jnp.arange(k)

        if condition_number > self.condition_threshold:
            warnings.warn(
                f"Ill-conditioned Jacobian (condition number: {condition_number:.2e})"
            )
            # Efficient diagonal regularization - 99% memory reduction vs jnp.eye(m, n)
            J = J.at[diag_indices, diag_indices].add(self.regularization_factor)
            regularized = True

        # Check for near-zero singular values (use cached value)
        if min_sv_cached is not None and min_sv_cached < self.eps * 10:
            # Efficient diagonal regularization
            J = J.at[diag_indices, diag_indices].add(self.eps * 10)
            regularized = True

        issues = {
            "has_nan": bool(has_invalid),
            "has_inf": bool(has_invalid),
            "is_ill_conditioned": condition_number > self.condition_threshold,
            "condition_number": condition_number,
            "regularized": regularized,
            "svd_skipped": False,
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
        max_grad_norm = 1e6

        # Use JIT-compiled gradient checking (5-15x faster)
        gradient_fixed, has_invalid, needs_clipping, grad_norm = (
            self._check_gradient_jit(gradient, max_grad_norm)
        )

        # Warnings outside JIT (side effects not allowed in JIT)
        if has_invalid:
            warnings.warn("Gradient contains NaN or Inf values")
        if needs_clipping:
            warnings.warn(f"Gradient norm too large ({float(grad_norm):.2e}), clipping")

        return gradient_fixed

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

        return residuals, bool(has_outliers)

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

        # Use JIT-compiled safe norm when scaling is needed (2-5x faster)
        if max_val > 1e100 or (max_val < 1e-100 and max_val > 0):
            return float(self._safe_norm_jit(x, max_val))
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


# ==============================================================================
# Pre-flight Stability Checks (Day 18 - Phase 3)
# ==============================================================================


def estimate_condition_number(xdata: np.ndarray) -> float:
    """
    Estimate the condition number of the data matrix.

    This checks if the independent variable data is well-conditioned for
    least squares fitting. High condition numbers indicate numerical instability.

    Parameters
    ----------
    xdata : array_like
        Independent variable data (can be 1D or 2D)

    Returns
    -------
    condition_number : float
        Estimated condition number. Values > 1e10 indicate potential problems.

    Notes
    -----
    For 1D data, constructs a Vandermonde-like matrix with [1, x, x^2].
    For multidimensional data, computes the condition number directly.
    """
    xdata = np.asarray(xdata)

    # Handle 1D data
    if xdata.ndim == 1:
        # Create a simple design matrix [1, x, x^2]
        X = np.column_stack([np.ones_like(xdata), xdata, xdata**2])
    else:
        # Use data directly for multidimensional case
        X = xdata

    # Compute condition number
    try:
        cond = np.linalg.cond(X)
        return float(cond)
    except (np.linalg.LinAlgError, ValueError):
        # If computation fails, return infinity
        return np.inf


def detect_parameter_scale_mismatch(
    p0: np.ndarray, threshold: float = 1e6
) -> tuple[bool, float]:
    """
    Detect if parameter scales differ by too many orders of magnitude.

    Large scale differences can cause numerical issues and slow convergence.

    Parameters
    ----------
    p0 : array_like
        Initial parameter guess
    threshold : float, optional
        Ratio threshold for detecting mismatch. Default: 1e6

    Returns
    -------
    has_mismatch : bool
        True if parameter scales differ by more than threshold
    scale_ratio : float
        Ratio of largest to smallest parameter magnitude

    Examples
    --------
    >>> p0 = np.array([1e-3, 1e3, 1.0])
    >>> has_mismatch, ratio = detect_parameter_scale_mismatch(p0)
    >>> print(f"Mismatch: {has_mismatch}, Ratio: {ratio:.2e}")
    Mismatch: True, Ratio: 1.00e+06
    """
    p0 = np.asarray(p0)

    # Get absolute values (ignore sign)
    abs_p0 = np.abs(p0)

    # Handle zero parameters
    nonzero_p0 = abs_p0[abs_p0 > 0]
    if len(nonzero_p0) == 0:
        return False, 1.0

    max_val = np.max(nonzero_p0)
    min_val = np.min(nonzero_p0)

    scale_ratio = max_val / min_val
    has_mismatch = scale_ratio > threshold

    return bool(has_mismatch), float(scale_ratio)


def detect_collinearity(
    xdata: np.ndarray, threshold: float = 0.95
) -> tuple[bool, list]:
    """
    Detect collinearity in multidimensional input data.

    Collinearity occurs when predictors are highly correlated, leading to
    unstable parameter estimates.

    Parameters
    ----------
    xdata : array_like
        Independent variable data (multidimensional)
    threshold : float, optional
        Correlation threshold for detecting collinearity. Default: 0.95

    Returns
    -------
    has_collinearity : bool
        True if any pair of variables is highly correlated
    collinear_pairs : list of tuple
        List of (i, j, correlation) for collinear variable pairs

    Examples
    --------
    >>> x1 = np.linspace(0, 10, 100)
    >>> x2 = 2 * x1 + 0.01 * np.random.randn(100)  # Nearly collinear
    >>> xdata = np.column_stack([x1, x2])
    >>> has_coll, pairs = detect_collinearity(xdata)
    >>> print(f"Collinear: {has_coll}")
    Collinear: True
    """
    xdata = np.asarray(xdata)

    # Only makes sense for multidimensional data
    if xdata.ndim != 2 or xdata.shape[1] < 2:
        return False, []

    # Compute correlation matrix
    try:
        corr_matrix = np.corrcoef(xdata, rowvar=False)
    except (ValueError, np.linalg.LinAlgError):
        return False, []

    # Find highly correlated pairs - vectorized (10-50x faster than nested loop)
    n_vars = corr_matrix.shape[0]

    # Get upper triangular indices (excluding diagonal)
    i_idx, j_idx = np.triu_indices(n_vars, k=1)

    # Extract upper triangular correlations (vectorized)
    upper_corr = np.abs(corr_matrix[i_idx, j_idx])

    # Find indices where correlation exceeds threshold
    mask = upper_corr > threshold
    high_corr_i = i_idx[mask]
    high_corr_j = j_idx[mask]
    high_corr_vals = upper_corr[mask]

    # Build result list (only for pairs exceeding threshold)
    collinear_pairs = [
        (int(i), int(j), float(c))
        for i, j, c in zip(high_corr_i, high_corr_j, high_corr_vals, strict=False)
    ]

    has_collinearity = len(collinear_pairs) > 0
    return bool(has_collinearity), collinear_pairs


def check_problem_stability(
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray | None = None,
    f: Callable | None = None,
) -> dict:
    """
    Comprehensive pre-flight stability check for optimization problem.

    Identifies potential numerical issues before optimization begins,
    providing warnings and recommendations for fixes.

    Parameters
    ----------
    xdata : array_like
        Independent variable data
    ydata : array_like
        Dependent variable data
    p0 : array_like, optional
        Initial parameter guess
    f : callable, optional
        Model function (currently unused, reserved for future checks)

    Returns
    -------
    report : dict
        Stability report with keys:
        - 'issues': list of (issue_type, message, severity) tuples
        - 'condition_number': float
        - 'parameter_scale_ratio': float or None
        - 'has_collinearity': bool
        - 'recommendations': list of str
        - 'severity': str ('ok', 'warning', 'critical')

    Examples
    --------
    >>> x = np.linspace(0, 1e6, 100)
    >>> y = 2.0 * x + 1.0
    >>> p0 = [2.0, 1.0]
    >>> report = check_problem_stability(x, y, p0)
    >>> print(f"Severity: {report['severity']}")
    >>> for issue_type, message, severity in report['issues']:
    ...     print(f"{severity}: {message}")
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    if p0 is not None:
        p0 = np.asarray(p0)

    issues = []
    recommendations = []

    # Check 1: Data validity
    if np.any(~np.isfinite(xdata)):
        issues.append(("invalid_xdata", "xdata contains NaN or Inf values", "critical"))
        recommendations.append("Remove or interpolate NaN/Inf values in xdata")

    if np.any(~np.isfinite(ydata)):
        issues.append(("invalid_ydata", "ydata contains NaN or Inf values", "critical"))
        recommendations.append("Remove or interpolate NaN/Inf values in ydata")

    # Check 2: Condition number
    cond = estimate_condition_number(xdata)
    if cond > 1e12:
        issues.append(
            (
                "ill_conditioned_data",
                f"xdata is ill-conditioned (cond={cond:.2e})",
                "critical",
            )
        )
        recommendations.append(
            "Rescale xdata to a smaller range (e.g., [0, 1] or [-1, 1])"
        )
    elif cond > 1e10:
        issues.append(
            (
                "poor_conditioning",
                f"xdata has poor conditioning (cond={cond:.2e})",
                "warning",
            )
        )
        recommendations.append("Consider rescaling xdata")

    # Check 3: Data range issues
    x_range = np.ptp(xdata)
    y_range = np.ptp(ydata)

    if x_range == 0:
        issues.append(("constant_xdata", "xdata has zero range", "critical"))
        recommendations.append("xdata must vary to fit a model")

    if y_range == 0:
        issues.append(("constant_ydata", "ydata has zero range", "warning"))
        recommendations.append("ydata is constant - model fit may be trivial")

    # Extreme ranges
    if x_range > 1e6:
        issues.append(
            ("large_x_range", f"xdata spans large range ({x_range:.2e})", "warning")
        )
        recommendations.append("Consider normalizing xdata")

    if y_range > 1e6:
        issues.append(
            ("large_y_range", f"ydata spans large range ({y_range:.2e})", "warning")
        )
        recommendations.append("Consider normalizing ydata")

    # Check 4: Parameter scale mismatch
    param_scale_ratio = None
    if p0 is not None and len(p0) > 1:
        has_mismatch, param_scale_ratio = detect_parameter_scale_mismatch(p0)
        if has_mismatch:
            issues.append(
                (
                    "parameter_scale_mismatch",
                    f"Parameter scales differ by {param_scale_ratio:.2e}",
                    "warning",
                )
            )
            recommendations.append(
                "Use x_scale parameter or rescale p0 to similar magnitudes"
            )

    # Check 5: Collinearity (for multidimensional data)
    has_collinearity = False
    collinear_pairs: list[tuple[int, int, float]] = []
    if xdata.ndim == 2 and xdata.shape[1] > 1:
        has_collinearity, collinear_pairs = detect_collinearity(xdata)
        if has_collinearity:
            pair_info = ", ".join(
                [f"({i},{j}): {corr:.3f}" for i, j, corr in collinear_pairs[:3]]
            )
            issues.append(
                (
                    "collinear_data",
                    f"Collinear predictors detected: {pair_info}",
                    "warning",
                )
            )
            recommendations.append(
                "Remove or combine highly correlated predictors, or use regularization"
            )

    # Determine overall severity
    if any(sev == "critical" for _, _, sev in issues):
        severity = "critical"
    elif any(sev == "warning" for _, _, sev in issues):
        severity = "warning"
    else:
        severity = "ok"

    return {
        "issues": issues,
        "condition_number": cond,
        "parameter_scale_ratio": param_scale_ratio,
        "has_collinearity": has_collinearity,
        "collinear_pairs": collinear_pairs,
        "recommendations": recommendations,
        "severity": severity,
    }


def apply_automatic_fixes(
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray | None = None,
    stability_report: dict | None = None,
    rescale_data: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """
    Automatically apply fixes for detected stability issues.

    This function applies common fixes such as rescaling data and parameters
    based on the stability report.

    Parameters
    ----------
    xdata : array_like
        Independent variable data
    ydata : array_like
        Dependent variable data
    p0 : array_like, optional
        Initial parameter guess
    stability_report : dict, optional
        Report from check_problem_stability(). If None, will be computed.
    rescale_data : bool, optional
        If True (default), rescale xdata/ydata to [0, 1] when ill-conditioned
        or large range is detected. Set to False for physics applications where
        data must maintain physical units (e.g., time delays in seconds,
        scattering vectors in nm^-1). Default: True.

    Returns
    -------
    xdata_fixed : ndarray
        Fixed xdata
    ydata_fixed : ndarray
        Fixed ydata
    p0_fixed : ndarray or None
        Fixed p0
    fix_info : dict
        Information about applied fixes with keys:
        - 'applied_fixes': list of str
        - 'x_scale': float
        - 'y_scale': float
        - 'x_offset': float
        - 'y_offset': float

    Examples
    --------
    >>> x = np.linspace(0, 1e6, 100)
    >>> y = 2.0 * x + 1.0
    >>> x_fixed, y_fixed, p0_fixed, info = apply_automatic_fixes(x, y, [2.0, 1.0])
    >>> print(f"Applied fixes: {info['applied_fixes']}")

    >>> # For physics data, disable rescaling to preserve physical units
    >>> x_fixed, y_fixed, p0_fixed, info = apply_automatic_fixes(
    ...     x, y, [2.0, 1.0], rescale_data=False
    ... )
    """
    xdata = np.asarray(xdata, dtype=np.float64)
    ydata = np.asarray(ydata, dtype=np.float64)
    if p0 is not None:
        p0 = np.asarray(p0, dtype=np.float64)

    applied_fixes: list[str] = []
    fix_info: dict[str, float | list[str]] = {
        "x_scale": 1.0,
        "y_scale": 1.0,
        "x_offset": 0.0,
        "y_offset": 0.0,
    }

    # Get stability report if not provided
    if stability_report is None:
        stability_report = check_problem_stability(xdata, ydata, p0)

    # Fix 1: Rescale xdata if ill-conditioned or large range (only if rescale_data=True)
    if rescale_data:
        cond = stability_report["condition_number"]
        x_range = np.ptp(xdata)

        if cond > 1e10 or x_range > 1e4:
            # Normalize to [0, 1]
            x_min = np.min(xdata)
            x_max = np.max(xdata)
            # Check for finite range to avoid division warnings with inf data
            if x_range > 0 and np.isfinite(x_range):
                xdata = (xdata - x_min) / x_range
                fix_info["x_scale"] = x_range
                fix_info["x_offset"] = x_min
                applied_fixes.append(
                    f"Rescaled xdata from [{x_min:.2e}, {x_max:.2e}] to [0, 1]"
                )

        # Fix 2: Rescale ydata if large range
        y_range = np.ptp(ydata)
        if y_range > 1e4 and np.isfinite(y_range):
            # Normalize to similar scale as x
            y_min = np.min(ydata)
            y_max = np.max(ydata)
            ydata = (ydata - y_min) / y_range
            fix_info["y_scale"] = y_range
            fix_info["y_offset"] = y_min
            applied_fixes.append(
                f"Rescaled ydata from [{y_min:.2e}, {y_max:.2e}] to [0, 1]"
            )

    # Fix 3: Replace NaN/Inf in data
    if np.any(~np.isfinite(xdata)):
        # Use mean of finite values only (nanmean doesn't ignore inf)
        finite_mean = (
            np.mean(xdata[np.isfinite(xdata)]) if np.any(np.isfinite(xdata)) else 0.0
        )
        xdata = np.where(np.isfinite(xdata), xdata, finite_mean)
        applied_fixes.append("Replaced NaN/Inf in xdata with mean")

    if np.any(~np.isfinite(ydata)):
        # Use mean of finite values only (nanmean doesn't ignore inf)
        finite_mean = (
            np.mean(ydata[np.isfinite(ydata)]) if np.any(np.isfinite(ydata)) else 0.0
        )
        ydata = np.where(np.isfinite(ydata), ydata, finite_mean)
        applied_fixes.append("Replaced NaN/Inf in ydata with mean")

    # Fix 4: Adjust p0 scales if needed
    p0_fixed = p0
    if (
        p0 is not None
        and stability_report["parameter_scale_ratio"]
        and stability_report["parameter_scale_ratio"] > 1e6
    ):
        # Normalize each parameter independently to order of magnitude 1
        p0_fixed = np.copy(p0)
        for i in range(len(p0_fixed)):
            if abs(p0_fixed[i]) > 0:
                # Get order of magnitude
                order = 10 ** np.floor(np.log10(abs(p0_fixed[i])))
                p0_fixed[i] = p0_fixed[i] / order
        applied_fixes.append("Normalized p0 to unit order of magnitude")

    fix_info["applied_fixes"] = applied_fixes

    return xdata, ydata, p0_fixed, fix_info
