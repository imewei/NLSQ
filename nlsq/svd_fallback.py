"""SVD computation with GPU/CPU fallback and randomized SVD for robustness.

This module provides:
- compute_svd_with_fallback: Standard SVD with GPU/CPU fallback
- randomized_svd: Fast approximate SVD for large matrices (O(mn*k) vs O(mn*min(m,n)))
- compute_svd_adaptive: Automatically chooses between full and randomized SVD
"""

import warnings
from functools import wraps

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import svd as jax_svd

# Threshold for switching to randomized SVD (m*n > threshold)
RANDOMIZED_SVD_THRESHOLD = 500_000  # ~700x700 matrix


def randomized_svd(
    A: jnp.ndarray,
    n_components: int | None = None,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute truncated randomized SVD (3-10x faster for large matrices).

    Uses the algorithm from Halko, Martinsson, and Tropp (2011):
    "Finding Structure with Randomness: Probabilistic Algorithms for
    Constructing Approximate Matrix Decompositions"

    Complexity: O(m*n*k) vs O(m*n*min(m,n)) for full SVD

    Parameters
    ----------
    A : jnp.ndarray
        Input matrix of shape (m, n)
    n_components : int, optional
        Number of singular values/vectors to compute.
        Default: min(m, n, 100)
    n_oversamples : int, default=10
        Additional samples for accuracy (trade-off with speed)
    n_iter : int, default=2
        Number of power iterations for accuracy
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    U : jnp.ndarray
        Left singular vectors, shape (m, n_components)
    s : jnp.ndarray
        Singular values, shape (n_components,)
    V : jnp.ndarray
        Right singular vectors (NOT transposed), shape (n, n_components)
    """
    m, n = A.shape

    # Default to capturing most information
    if n_components is None:
        n_components = min(m, n, 100)

    # Ensure we don't request more components than possible
    n_components = min(n_components, m, n)
    k = n_components + n_oversamples

    # Generate random projection matrix
    key = random.PRNGKey(random_state)
    Omega = random.normal(key, (n, k))

    # Range finder with power iterations for accuracy
    Y = A @ Omega
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # QR factorization to get orthonormal basis
    Q, _ = jnp.linalg.qr(Y)

    # Project A to smaller subspace
    B = Q.T @ A

    # SVD of the small matrix
    U_hat, s, Vt = jax_svd(B, full_matrices=False)

    # Recover left singular vectors
    U = Q @ U_hat

    # Truncate to requested components
    U = U[:, :n_components]
    s = s[:n_components]
    V = Vt[:n_components, :].T

    return U, s, V


def compute_svd_adaptive(
    J_h: jnp.ndarray,
    full_matrices: bool = False,
    use_randomized: bool | None = None,
    n_components: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute SVD with automatic algorithm selection.

    For large matrices (m*n > threshold), uses randomized SVD for 3-10x speedup.
    Falls back to full SVD for small matrices or when high precision is needed.

    Parameters
    ----------
    J_h : jnp.ndarray
        Jacobian matrix in hat space
    full_matrices : bool, default=False
        Whether to compute full matrices (only for full SVD)
    use_randomized : bool, optional
        Force randomized (True) or full (False) SVD.
        If None, automatically decide based on matrix size.
    n_components : int, optional
        Number of components for randomized SVD.
        Default: min(m, n, n_params) where n_params is the smaller dimension

    Returns
    -------
    U, s, V : jnp.ndarray
        SVD decomposition (V is transposed back for consistency)
    """
    m, n = J_h.shape
    matrix_size = m * n

    # Decide whether to use randomized SVD
    if use_randomized is None:
        use_randomized = matrix_size > RANDOMIZED_SVD_THRESHOLD

    if use_randomized:
        # For least squares, we typically need all n components
        if n_components is None:
            n_components = min(m, n)
        return randomized_svd(J_h, n_components=n_components)
    else:
        return compute_svd_with_fallback(J_h, full_matrices=full_matrices)


def with_cpu_fallback(func):
    """Decorator to add CPU fallback for GPU operations that might fail."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Try GPU first
            return func(*args, **kwargs)
        except Exception as e:
            if "cuSolver" in str(e) or "INTERNAL" in str(e):
                warnings.warn(
                    f"GPU operation failed ({e}), falling back to CPU",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # Force CPU execution
                with jax.default_device(jax.devices("cpu")[0]):
                    return func(*args, **kwargs)
            else:
                # Re-raise if not a GPU-specific error
                raise

    return wrapper


@with_cpu_fallback
def safe_svd(matrix, full_matrices=False):
    """Compute SVD with automatic CPU fallback if GPU fails.

    Parameters
    ----------
    matrix : jnp.ndarray
        Matrix to decompose
    full_matrices : bool
        Whether to compute full matrices

    Returns
    -------
    U, s, Vt : jnp.ndarray
        SVD decomposition components
    """
    return jax_svd(matrix, full_matrices=full_matrices)


def compute_svd_with_fallback(J_h, full_matrices=False):
    """Compute SVD with multiple fallback strategies.

    Parameters
    ----------
    J_h : jnp.ndarray
        Jacobian matrix in hat space
    full_matrices : bool
        Whether to compute full matrices

    Returns
    -------
    U, s, V : jnp.ndarray
        SVD decomposition (note: V is transposed back)
    """
    try:
        # First attempt: Direct GPU computation
        U, s, Vt = jax_svd(J_h, full_matrices=full_matrices)
        return U, s, Vt.T
    except Exception as gpu_error:
        # Check if it's a cuSolver error
        error_msg = str(gpu_error)
        if "cuSolver" in error_msg or "INTERNAL" in error_msg:
            warnings.warn(
                "GPU SVD failed with cuSolver error, attempting CPU fallback",
                RuntimeWarning,
            )

            try:
                # Second attempt: CPU computation
                cpu_device = jax.devices("cpu")[0]
                with jax.default_device(cpu_device):
                    # Move data to CPU
                    J_h_cpu = jax.device_put(J_h, cpu_device)
                    U, s, Vt = jax_svd(J_h_cpu, full_matrices=full_matrices)
                    return U, s, Vt.T
            except Exception:
                # Third attempt: Use numpy as last resort
                warnings.warn(
                    "CPU JAX SVD also failed, using NumPy SVD", RuntimeWarning
                )
                import numpy as np

                # Convert to numpy, compute, convert back
                J_h_np = np.array(J_h)
                U_np, s_np, Vt_np = np.linalg.svd(J_h_np, full_matrices=full_matrices)

                # Convert back to JAX arrays
                U = jnp.array(U_np)
                s = jnp.array(s_np)
                V = jnp.array(Vt_np.T)

                return U, s, V
        else:
            # Not a GPU-specific error, re-raise
            raise


def initialize_gpu_safely():
    """Initialize GPU with proper memory settings to avoid cuSolver issues."""
    try:
        # Set memory preallocation to avoid fragmentation
        import os

        if "JAX_PREALLOCATE_GPU_MEMORY" not in os.environ:
            os.environ["JAX_PREALLOCATE_GPU_MEMORY"] = "false"

        # Try to configure XLA to be more conservative with memory
        if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        # Set memory fraction if not already set
        if "JAX_GPU_MEMORY_FRACTION" not in os.environ:
            os.environ["JAX_GPU_MEMORY_FRACTION"] = "0.8"

    except Exception as e:
        warnings.warn(f"Could not configure GPU memory settings: {e}")
