"""SVD computation with GPU/CPU fallback for robustness.

This module provides:
- compute_svd_with_fallback: Deterministic full SVD with GPU/CPU/NumPy fallback chain

IMPORTANT: As of v0.3.5, this module uses ONLY full deterministic SVD.
Randomized/approximate SVD has been completely removed because it causes
optimization divergence in iterative least-squares solvers.

Historical note (v0.3.1-v0.3.4):
  Randomized SVD was available but caused 3-25x worse fitting errors in
  iterative least-squares applications due to accumulated approximation error
  across trust-region iterations. See tests/test_svd_regression.py for evidence.
"""

import warnings

import jax
import jax.numpy as jnp
from jax.scipy.linalg import svd as jax_svd


def is_gpu_error(error: Exception | str) -> bool:
    """Check if an exception indicates a GPU/CUDA-specific failure.

    Prefers type-based matching against jaxlib.xla_extension.XlaRuntimeError,
    which is stable across JAX versions. Falls back to string heuristics for
    older JAX (<0.8) or error types that don't inherit from XlaRuntimeError:
    - Legacy: "cuSolver internal error" (JAX <0.8)
    - FFI: "No FFI handler registered for cusolver_gesvdj_ffi" (JAX >=0.8)
    - XLA status: "INTERNAL: ..." (gRPC/XLA status code prefix, case-sensitive)
    """
    # Type-based check: robust against JAX error message format changes
    if not isinstance(error, str):
        try:
            from jaxlib.xla_extension import XlaRuntimeError

            if isinstance(error, XlaRuntimeError):
                return True
        except (ImportError, AttributeError):
            pass

    # String fallback: covers older JAX and non-XlaRuntimeError GPU failures
    msg = str(error)
    msg_lower = msg.lower()
    return (
        "cusolver" in msg_lower
        or "cublas" in msg_lower
        or ("ffi" in msg_lower and "cuda" in msg_lower)
        or msg.startswith("INTERNAL:")
    )


# Backwards-compatible private alias used by tests
_is_gpu_error = is_gpu_error


def compute_svd_with_fallback(
    J_h: jnp.ndarray, full_matrices: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        # Check if it's a GPU-specific error (cuSolver or CUDA FFI)
        if is_gpu_error(gpu_error):
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
            except Exception as cpu_error:
                # Third attempt: Use numpy as last resort
                warnings.warn(
                    f"CPU JAX SVD also failed ({cpu_error}), using NumPy SVD",
                    RuntimeWarning,
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
