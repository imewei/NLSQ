"""SVD computation with GPU/CPU fallback for robustness."""

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
