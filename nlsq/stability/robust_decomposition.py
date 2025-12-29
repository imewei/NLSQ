"""Robust matrix decomposition with multi-level fallback strategies.

This module extends the SVD fallback to provide comprehensive fallback
strategies for all matrix decompositions used in optimization.
"""

from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cholesky as jax_cholesky
from jax.scipy.linalg import qr as jax_qr
from jax.scipy.linalg import svd as jax_svd

# Use NLSQ logging system
from nlsq.utils.logging import get_logger

# Import the existing SVD fallback utilities


class RobustDecomposition:
    """Multi-level fallback for matrix decompositions.

    This class provides robust matrix decomposition methods that
    automatically fall back through multiple strategies if the
    primary method fails. The fallback chain goes from:
    1. JAX on GPU (if available)
    2. JAX on CPU
    3. SciPy (if available)
    4. NumPy
    5. Safe mode with regularization

    Attributes
    ----------
    fallback_chain : list
        Ordered list of (name, method) tuples for fallback strategies
    logger : logging.Logger
        Logger for debugging decomposition issues
    """

    def __init__(self, enable_logging: bool = False):
        """Initialize robust decomposition handler.

        Parameters
        ----------
        enable_logging : bool
            Whether to enable detailed logging of fallback attempts
        """
        self.logger = get_logger("robust_decomposition")
        if enable_logging:
            from nlsq.utils.logging import LogLevel

            self.logger.logger.setLevel(LogLevel.DEBUG)

        # Build fallback chain
        self.fallback_chain = [
            ("jax_gpu", self._jax_gpu_decomp),
            ("jax_cpu", self._jax_cpu_decomp),
            ("scipy", self._scipy_decomp),
            ("numpy", self._numpy_decomp),
            ("safe_mode", self._safe_mode_decomp),
        ]

        # Regularization parameters
        self.eps = np.finfo(np.float64).eps
        self.regularization_factor = 1e-10

    def svd(
        self, matrix: jnp.ndarray, full_matrices: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute SVD with automatic fallback.

        Parameters
        ----------
        matrix : jnp.ndarray
            Matrix to decompose
        full_matrices : bool
            Whether to compute full matrices

        Returns
        -------
        U : jnp.ndarray
            Left singular vectors
        s : jnp.ndarray
            Singular values
        Vt : jnp.ndarray
            Right singular vectors (transposed)

        Raises
        ------
        RuntimeError
            If all decomposition methods fail
        """
        for name, method in self.fallback_chain:
            try:
                result = method(matrix, "svd", full_matrices)
                if result is not None and self._validate_svd(result):
                    self.logger.debug(f"SVD succeeded with {name}")
                    return result
            except Exception as e:
                self.logger.debug(f"{name} SVD failed: {e}")
                continue

        raise RuntimeError("All SVD methods failed")

    def qr(
        self, matrix: jnp.ndarray, mode: str = "reduced"
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute QR decomposition with fallback.

        Parameters
        ----------
        matrix : jnp.ndarray
            Matrix to decompose
        mode : str
            QR mode ('reduced' or 'complete')

        Returns
        -------
        Q : jnp.ndarray
            Orthogonal matrix
        R : jnp.ndarray
            Upper triangular matrix

        Raises
        ------
        RuntimeError
            If all decomposition methods fail
        """
        for name, method in self.fallback_chain:
            try:
                result = method(matrix, "qr", mode)
                if result is not None and self._validate_qr(result):
                    self.logger.debug(f"QR succeeded with {name}")
                    return result
            except Exception as e:
                self.logger.debug(f"{name} QR failed: {e}")
                continue

        raise RuntimeError("All QR methods failed")

    def cholesky(self, matrix: jnp.ndarray, lower: bool = True) -> jnp.ndarray:
        """Compute Cholesky decomposition with fallback and regularization.

        Parameters
        ----------
        matrix : jnp.ndarray
            Positive definite matrix to decompose
        lower : bool
            Whether to return lower triangular matrix

        Returns
        -------
        L : jnp.ndarray
            Cholesky factor (lower or upper triangular)

        Raises
        ------
        RuntimeError
            If all decomposition methods fail
        """
        # First ensure matrix is positive definite
        matrix = self._ensure_positive_definite(matrix)

        for name, method in self.fallback_chain:
            try:
                result = method(matrix, "cholesky", lower)
                if result is not None:
                    self.logger.debug(f"Cholesky succeeded with {name}")
                    return result
            except Exception as e:
                self.logger.debug(f"{name} Cholesky failed: {e}")
                continue

        # Last resort: eigendecomposition
        return self._cholesky_via_eigen(matrix, lower)

    def _jax_gpu_decomp(self, matrix: jnp.ndarray, decomp_type: str, *args):
        """Try decomposition on GPU using JAX."""
        if not jax.devices("gpu"):
            return None

        gpu_device = jax.devices("gpu")[0]
        with jax.default_device(gpu_device):
            if decomp_type == "svd":
                full_matrices = args[0] if args else False
                U, s, Vt = jax_svd(matrix, full_matrices=full_matrices)
                return U, s, Vt
            elif decomp_type == "qr":
                mode = args[0] if args else "reduced"
                qr_result = cast(
                    tuple[jnp.ndarray, jnp.ndarray], jax_qr(matrix, mode=mode)
                )
                Q, R = qr_result
                return Q, R
            elif decomp_type == "cholesky":
                lower = args[0] if args else True
                L = jax_cholesky(matrix, lower=lower)
                return L

    def _jax_cpu_decomp(self, matrix: jnp.ndarray, decomp_type: str, *args):
        """Try decomposition on CPU using JAX."""
        try:
            cpu_device = jax.devices("cpu")[0]
        except IndexError:
            return None

        with jax.default_device(cpu_device):
            # Move data to CPU
            matrix_cpu = jax.device_put(matrix, cpu_device)

            if decomp_type == "svd":
                full_matrices = args[0] if args else False
                U, s, Vt = jax_svd(matrix_cpu, full_matrices=full_matrices)
                return U, s, Vt
            elif decomp_type == "qr":
                mode = args[0] if args else "reduced"
                qr_result = cast(
                    tuple[jnp.ndarray, jnp.ndarray], jax_qr(matrix_cpu, mode=mode)
                )
                Q, R = qr_result
                return Q, R
            elif decomp_type == "cholesky":
                lower = args[0] if args else True
                L = jax_cholesky(matrix_cpu, lower=lower)
                return L

    def _scipy_decomp(self, matrix: jnp.ndarray, decomp_type: str, *args):
        """Try decomposition using SciPy."""
        try:
            import scipy.linalg
        except ImportError:
            return None

        # Convert to numpy
        matrix_np = np.array(matrix)

        if decomp_type == "svd":
            full_matrices = args[0] if args else False
            U, s, Vt = scipy.linalg.svd(matrix_np, full_matrices=full_matrices)
            return jnp.array(U), jnp.array(s), jnp.array(Vt)
        elif decomp_type == "qr":
            mode = args[0] if args else "reduced"
            Q, R = scipy.linalg.qr(matrix_np, mode=mode)
            return jnp.array(Q), jnp.array(R)
        elif decomp_type == "cholesky":
            lower = args[0] if args else True
            L = scipy.linalg.cholesky(matrix_np, lower=lower)
            return jnp.array(L)

    def _numpy_decomp(self, matrix: jnp.ndarray, decomp_type: str, *args):
        """Try decomposition using NumPy."""
        # Convert to numpy
        matrix_np = np.array(matrix)

        if decomp_type == "svd":
            full_matrices = args[0] if args else False
            U, s, Vt = np.linalg.svd(matrix_np, full_matrices=full_matrices)
            return jnp.array(U), jnp.array(s), jnp.array(Vt)
        elif decomp_type == "qr":
            mode = args[0] if args else "reduced"
            mode = cast(Literal["reduced", "complete", "r", "raw"], mode)
            Q, R = np.linalg.qr(matrix_np, mode=mode)
            return jnp.array(Q), jnp.array(R)
        elif decomp_type == "cholesky":
            L = np.linalg.cholesky(matrix_np)
            return jnp.array(L)

    def _safe_mode_decomp(self, matrix: jnp.ndarray, decomp_type: str, *args):
        """Safe mode decomposition with regularization."""
        m, n = matrix.shape

        if decomp_type == "svd":
            # Add regularization to improve conditioning
            reg_matrix = matrix + self.regularization_factor * jnp.eye(m, n)
            return self._numpy_decomp(reg_matrix, "svd", *args)

        elif decomp_type == "qr":
            # QR with column pivoting for stability
            matrix_np = np.array(matrix)
            try:
                import scipy.linalg

                Q, R, P = scipy.linalg.qr(matrix_np, mode="economic", pivoting=True)
                # Reorder to undo pivoting
                R_reordered = R[:, np.argsort(P)]
                return jnp.array(Q), jnp.array(R_reordered)
            except (np.linalg.LinAlgError, ValueError, ImportError):
                # Fall back to basic QR with regularization
                reg_matrix = matrix + self.eps * jnp.eye(m, n)
                return self._numpy_decomp(reg_matrix, "qr", *args)

        elif decomp_type == "cholesky":
            # Ensure positive definite with stronger regularization
            matrix = self._ensure_positive_definite(
                matrix, factor=self.regularization_factor * 100
            )
            return self._numpy_decomp(matrix, "cholesky", *args)

    def _ensure_positive_definite(
        self, matrix: jnp.ndarray, factor: float | None = None
    ) -> jnp.ndarray:
        """Make matrix positive definite.

        Parameters
        ----------
        matrix : jnp.ndarray
            Matrix to make positive definite
        factor : float, optional
            Regularization factor (uses default if None)

        Returns
        -------
        matrix_pd : jnp.ndarray
            Positive definite matrix
        """
        if factor is None:
            factor = self.regularization_factor

        n = matrix.shape[0]

        # Ensure symmetry
        matrix = 0.5 * (matrix + matrix.T)

        try:
            # Check minimum eigenvalue
            eigenvalues = jnp.linalg.eigvalsh(matrix)
            min_eig = jnp.min(eigenvalues)

            if min_eig < factor:
                # Add diagonal to ensure positive definiteness
                shift = factor - min_eig + self.eps
                matrix = matrix + shift * jnp.eye(n)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback: add diagonal regularization
            matrix = matrix + factor * jnp.eye(n)

        return matrix

    def _cholesky_via_eigen(
        self, matrix: jnp.ndarray, lower: bool = True
    ) -> jnp.ndarray:
        """Compute Cholesky via eigendecomposition.

        Parameters
        ----------
        matrix : jnp.ndarray
            Positive definite matrix
        lower : bool
            Whether to return lower triangular

        Returns
        -------
        L : jnp.ndarray
            Cholesky factor
        """
        try:
            # Eigendecomposition
            eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)

            # Ensure all eigenvalues are positive
            eigenvalues = jnp.maximum(eigenvalues, self.eps)

            # Reconstruct: A = V * diag(lambda) * V^T
            # So L = V * sqrt(diag(lambda))
            L = eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues))

            if lower:
                return L
            else:
                return L.T
        except Exception as e:
            raise RuntimeError(f"Cholesky via eigendecomposition failed: {e}") from e

    def _validate_svd(self, result: tuple) -> bool:
        """Validate SVD result.

        Parameters
        ----------
        result : tuple
            (U, s, Vt) from SVD

        Returns
        -------
        valid : bool
            Whether the result is valid
        """
        try:
            U, s, Vt = result
            return (
                bool(jnp.all(jnp.isfinite(U)))
                and bool(jnp.all(jnp.isfinite(s)))
                and bool(jnp.all(jnp.isfinite(Vt)))
                and bool(jnp.all(s >= 0))  # Singular values must be non-negative
            )
        except (ValueError, TypeError, AttributeError):
            return False

    def _validate_qr(self, result: tuple) -> bool:
        """Validate QR result.

        Parameters
        ----------
        result : tuple
            (Q, R) from QR decomposition

        Returns
        -------
        valid : bool
            Whether the result is valid
        """
        try:
            Q, R = result
            return bool(jnp.all(jnp.isfinite(Q))) and bool(jnp.all(jnp.isfinite(R)))
        except (ValueError, TypeError, AttributeError):
            return False

    def solve_least_squares(self, A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Solve least squares problem with robust decomposition.

        Parameters
        ----------
        A : jnp.ndarray
            Coefficient matrix
        b : jnp.ndarray
            Right-hand side

        Returns
        -------
        x : jnp.ndarray
            Solution vector
        """
        try:
            # Try SVD first (most stable)
            U, s, Vt = self.svd(A, full_matrices=False)

            # Compute pseudoinverse solution
            # x = V @ (S^+ @ (U^T @ b))
            s_inv = jnp.where(s > self.eps * jnp.max(s), 1.0 / s, 0.0)
            x = Vt.T @ (s_inv * (U.T @ b))
            return x

        except Exception as e:
            self.logger.warning(f"SVD failed for least squares, trying QR: {e}")

            try:
                # Fall back to QR
                Q, R = self.qr(A)
                # Solve R @ x = Q^T @ b
                y = Q.T @ b
                x = jnp.linalg.solve(R, y)
                return x

            except Exception as e2:
                self.logger.warning(
                    f"QR failed for least squares, using normal equations: {e2}"
                )

                # Last resort: normal equations (less stable)
                AtA = A.T @ A
                Atb = A.T @ b

                # Regularize if needed
                AtA = self._ensure_positive_definite(AtA)

                try:
                    L = self.cholesky(AtA)
                    # Solve L @ L^T @ x = A^T @ b
                    y = jnp.linalg.solve(L, Atb)
                    x = jnp.linalg.solve(L.T, y)
                    return x
                except (np.linalg.LinAlgError, ValueError):
                    # Ultimate fallback: direct solve with regularization
                    x = jnp.linalg.solve(AtA, Atb)
                    return x


# Create global instance
robust_decomp = RobustDecomposition()
