"""SVD computation with GPU/CPU fallback for robustness.

This module provides:
- compute_svd_with_fallback: Deterministic full SVD with GPU/CPU/NumPy fallback chain
- compute_svd_adaptive: Deprecated alias for compute_svd_with_fallback (full SVD only)

IMPORTANT: As of v0.3.5, this module uses ONLY full deterministic SVD.
Randomized/approximate SVD has been completely removed because it causes
optimization divergence in iterative least-squares solvers.

Historical note (v0.3.1-v0.3.4):
  Randomized SVD was available but caused 3-25x worse fitting errors in
  XPCS applications due to accumulated approximation error across trust-region
  iterations. See tests/test_svd_regression.py for evidence.
"""

import warnings
from functools import wraps

import jax
import jax.numpy as jnp
from jax.scipy.linalg import svd as jax_svd


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
    """Compute full deterministic SVD with multiple fallback strategies.

    This is the primary SVD function for NLSQ. It uses full (exact) SVD
    to ensure numerical precision and reproducibility in optimization.

    Fallback chain:
    1. JAX GPU SVD (if GPU available)
    2. JAX CPU SVD (if GPU fails with cuSolver error)
    3. NumPy SVD (last resort)

    Parameters
    ----------
    J_h : jnp.ndarray
        Jacobian matrix in hat space
    full_matrices : bool
        Whether to compute full matrices (default: False for efficiency)

    Returns
    -------
    U : jnp.ndarray
        Left singular vectors
    s : jnp.ndarray
        Singular values (sorted in descending order)
    V : jnp.ndarray
        Right singular vectors (note: V is transposed back, NOT Vt)
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


def compute_svd_adaptive(
    J_h: jnp.ndarray,
    full_matrices: bool = False,
    use_randomized: bool | None = None,
    n_components: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute SVD - ALWAYS uses full deterministic SVD.

    .. deprecated:: 0.3.5
        The `use_randomized` and `n_components` parameters are ignored.
        This function now always uses full deterministic SVD via
        `compute_svd_with_fallback`. Use that function directly instead.

    Randomized SVD was removed in v0.3.5 because it caused optimization
    divergence in iterative least-squares solvers.

    Parameters
    ----------
    J_h : jnp.ndarray
        Jacobian matrix in hat space
    full_matrices : bool, default=False
        Whether to compute full matrices
    use_randomized : bool, optional
        IGNORED. Kept for API compatibility only.
        If True, a deprecation warning is issued.
    n_components : int, optional
        IGNORED. Kept for API compatibility only.

    Returns
    -------
    U, s, V : jnp.ndarray
        Full deterministic SVD decomposition (V is transposed back)
    """
    # Warn if caller explicitly requested randomized SVD
    if use_randomized is True:
        warnings.warn(
            "Randomized SVD was removed in v0.3.5 due to optimization divergence. "
            "Using full deterministic SVD instead. See tests/test_svd_regression.py "
            "for evidence of the issue.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Always use full deterministic SVD
    return compute_svd_with_fallback(J_h, full_matrices=full_matrices)


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
