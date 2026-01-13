"""CovarianceComputer component for CurveFit decomposition.

Handles covariance matrix computation via SVD, sigma transformation,
and condition number estimation.

Reference: specs/017-curve-fit-decomposition/spec.md FR-003
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.numpy.linalg import cholesky as jax_cholesky
from jax.numpy.linalg import svd as jax_svd

from nlsq.interfaces.orchestration_protocol import CovarianceResult

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from nlsq.result.optimize_result import OptimizeResult


class CovarianceComputer:
    """Computer for parameter covariance from optimization results.

    Handles:
    1. Jacobian-based covariance via SVD
    2. Sigma transformation (1D and 2D)
    3. Absolute vs relative sigma handling
    4. Singularity detection and handling

    Example:
        >>> computer = CovarianceComputer()
        >>> result = computer.compute(
        ...     result=optimize_result,
        ...     n_data=100,
        ...     sigma=uncertainties,
        ...     absolute_sigma=True,
        ... )
        >>> print(f"Parameter errors: {result.perr}")
    """

    def __init__(self) -> None:
        """Initialize CovarianceComputer with JIT-compiled functions."""
        self._sigma_transform1d = self._create_sigma_transform1d()
        self._sigma_transform2d = self._create_sigma_transform2d()
        self._covariance_svd = self._create_covariance_svd()

    def _create_sigma_transform1d(self) -> Callable:
        """Create JIT-compiled 1D sigma transform."""

        @jit
        def sigma_transform1d(
            sigma: jnp.ndarray, data_mask: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute the sigma transform for 1D data.

            Args:
                sigma: The standard deviation of the data.
                data_mask: A binary mask indicating which data points to use.

            Returns:
                The sigma transform (1/sigma).
            """
            return 1.0 / sigma

        return sigma_transform1d

    def _create_sigma_transform2d(self) -> Callable:
        """Create JIT-compiled 2D sigma transform."""

        @jit
        def sigma_transform2d(
            sigma: jnp.ndarray, data_mask: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute the sigma transform for 2D covariance matrix.

            Args:
                sigma: The covariance matrix.
                data_mask: A binary mask indicating which data points to use.

            Returns:
                The Cholesky decomposition (lower triangular).
            """
            sigma = jnp.asarray(sigma)
            return jax_cholesky(sigma, lower=True)

        return sigma_transform2d

    def _create_covariance_svd(self) -> Callable:
        """Create JIT-compiled SVD function for covariance."""

        @jit
        def covariance_svd(jac: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Compute SVD of Jacobian for covariance estimation.

            Args:
                jac: Jacobian matrix at solution.

            Returns:
                Tuple of (singular_values, V_transpose).
            """
            _, s, VT = jax_svd(jac, full_matrices=False)
            return s, VT

        return covariance_svd

    def compute(
        self,
        result: OptimizeResult,
        n_data: int,
        *,
        sigma: jax.Array | None = None,
        absolute_sigma: bool = False,
        full_output: bool = False,
    ) -> CovarianceResult:
        """Compute parameter covariance from optimization result.

        Uses the Jacobian at the solution to compute covariance via:
        pcov = (J^T @ J)^(-1) * s_sq

        where s_sq is the residual variance.

        Args:
            result: OptimizeResult from LeastSquares
            n_data: Number of data points
            sigma: Observation uncertainties/weights
            absolute_sigma: If True, sigma is absolute uncertainty
            full_output: If True, include additional diagnostics

        Returns:
            CovarianceResult with covariance matrix and metadata

        Raises:
            ValueError: If Jacobian is unavailable or invalid
        """
        jac = result.jac
        n_params = len(result.x)
        cost = 2 * result.cost  # result.cost is half sum of squares

        # Compute SVD for Moore-Penrose pseudo-inverse
        s, VT = self._covariance_svd(jac)
        s_np = np.asarray(s)
        VT_np = np.asarray(VT)

        # Determine threshold for singular values
        threshold = np.finfo(float).eps * max(jac.shape) * s_np[0]

        # Filter out near-zero singular values
        valid_mask = s_np > threshold
        s_valid = s_np[valid_mask]
        VT_valid = VT_np[: len(s_valid)]

        # Check for singularity
        is_singular = len(s_valid) < n_params
        warn_cov = is_singular

        # Compute condition number
        if len(s_valid) > 0:
            condition_number = float(s_valid[0] / s_valid[-1])
        else:
            condition_number = float("inf")

        # Compute covariance matrix
        if is_singular:
            pcov = np.full((n_params, n_params), np.inf)
        else:
            pcov = np.dot(VT_valid.T / s_valid**2, VT_valid)

        # Apply scaling for relative sigma
        if not absolute_sigma:
            if n_data > n_params and not is_singular:
                s_sq = cost / (n_data - n_params)
                pcov = pcov * s_sq
            else:
                pcov = np.full((n_params, n_params), np.inf)
                warn_cov = True

        # Compute parameter errors
        perr = np.sqrt(np.diag(pcov))

        # Convert to JAX arrays
        jnp_pcov = jnp.asarray(pcov)
        jnp_perr = jnp.asarray(perr)

        return CovarianceResult(
            pcov=jnp_pcov,
            perr=jnp_perr,
            method="svd",
            condition_number=condition_number,
            is_singular=warn_cov,
            sigma_used=sigma is not None,
            absolute_sigma=absolute_sigma,
        )

    def create_sigma_transform(
        self,
        sigma: jax.Array,
        n_data: int,
    ) -> tuple[Callable, bool]:
        """Create sigma transformation function.

        Handles both 1D (diagonal) and 2D (full covariance) sigma.

        Args:
            sigma: Sigma array, shape (n,) or (n, n)
            n_data: Number of data points

        Returns:
            Tuple of (transform_func, is_2d)
            - transform_func: Function to apply sigma weighting
            - is_2d: True if sigma is full covariance matrix
        """
        sigma_np = np.asarray(sigma)

        if sigma_np.ndim == 1:
            # 1D sigma: errors, transform is 1/sigma
            return self._sigma_transform1d, False
        elif sigma_np.ndim == 2:
            # 2D sigma: covariance matrix, transform is Cholesky
            return self._sigma_transform2d, True
        else:
            msg = f"Sigma must be 1D or 2D, got {sigma_np.ndim}D"
            raise ValueError(msg)

    def compute_condition_number(
        self,
        jacobian: jax.Array,
    ) -> float:
        """Compute condition number of Jacobian.

        Uses singular values: cond = max(s) / min(s)

        Args:
            jacobian: Jacobian matrix at solution

        Returns:
            Condition number (inf if singular)
        """
        s, _ = self._covariance_svd(jacobian)
        s_np = np.asarray(s)

        # Filter near-zero singular values
        threshold = np.finfo(float).eps * max(jacobian.shape) * s_np[0]
        s_valid = s_np[s_np > threshold]

        if len(s_valid) == 0:
            return float("inf")

        return float(s_valid[0] / s_valid[-1])

    def setup_sigma_transform(
        self,
        sigma: np.ndarray | None,
        ydata: np.ndarray,
        data_mask: np.ndarray,
        len_diff: int,
        m: int,
    ) -> jax.Array | None:
        """Setup sigma transformation for weighted least squares.

        This is the legacy interface matching CurveFit._setup_sigma_transform.

        Args:
            sigma: Uncertainty in ydata (1-D errors or 2-D covariance matrix)
            ydata: Dependent data array
            data_mask: Boolean mask for valid data points
            len_diff: Difference in length for padding
            m: Original number of data points

        Returns:
            Transformation array for sigma or None

        Raises:
            ValueError: If sigma has incorrect shape or is not positive definite
        """
        if sigma is None:
            return None

        if not isinstance(sigma, np.ndarray):
            msg = "Sigma must be numpy array."
            raise ValueError(msg)

        ysize = ydata.size - len_diff

        # 1-D sigma: errors, define transform = 1/sigma
        if sigma.shape == (ysize,):
            if len_diff > 0:
                sigma = np.concatenate([sigma, np.ones([len_diff])])
            return self._sigma_transform1d(jnp.asarray(sigma), jnp.asarray(data_mask))

        # 2-D sigma: covariance matrix, define transform = L such that L L^T = C
        elif sigma.shape == (ysize, ysize):
            try:
                if len_diff >= 0:
                    sigma_padded = np.identity(m + len_diff)
                    sigma_padded[:m, :m] = sigma
                    sigma = sigma_padded
                return self._sigma_transform2d(
                    jnp.asarray(sigma), jnp.asarray(data_mask)
                )
            except Exception as e:
                # Check eigenvalues for better error message
                try:
                    eigenvalues = np.linalg.eigvalsh(sigma[:ysize, :ysize])
                    min_eig = np.min(eigenvalues)
                    if min_eig <= 0:
                        msg = (
                            f"Covariance matrix `sigma` is not positive definite. "
                            f"Minimum eigenvalue: {min_eig:.6e}. "
                            "All eigenvalues must be positive."
                        )
                        raise ValueError(msg) from e
                except Exception:
                    pass
                msg = (
                    "Failed to compute Cholesky decomposition of `sigma`. "
                    "The covariance matrix must be symmetric and positive definite."
                )
                raise ValueError(msg) from e
        else:
            msg = "`sigma` has incorrect shape."
            raise ValueError(msg)
