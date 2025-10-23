"""GPU detection and warning utilities for NLSQ.

This module provides runtime GPU availability checks to help users
realize when GPU acceleration is available but not being used.
"""

import os
import subprocess


def check_gpu_availability() -> None:
    """Check if GPU is available but not being used by JAX.

    Prints a helpful warning if:
    - NVIDIA GPU hardware is detected (nvidia-smi works)
    - But JAX is running in CPU-only mode

    This helps users realize they can enable GPU acceleration for
    150-270x speedup on large datasets (1M+ points).

    The check is silent on errors to avoid disrupting workflow when:
    - GPU hardware is not present
    - nvidia-smi is not installed
    - JAX is not installed yet
    - Other unexpected errors occur

    Environment Variables
    ---------------------
    NLSQ_SKIP_GPU_CHECK : str, optional
        Set to "1", "true", or "yes" (case-insensitive) to suppress GPU warnings.
        Useful for CI/CD pipelines or users who intentionally use CPU-only JAX.

        Examples:
            export NLSQ_SKIP_GPU_CHECK=1
            NLSQ_SKIP_GPU_CHECK=true python script.py

    Example
    -------
    This function is automatically called when importing nlsq:

    >>> import nlsq  # Automatically checks GPU availability
    ⚠️  GPU ACCELERATION AVAILABLE
    ═══════════════════════════════
    NVIDIA GPU detected: Tesla V100-SXM2-16GB
    JAX is currently using: CPU-only

    Enable 150-270x speedup with GPU acceleration:
      make install-jax-gpu

    Or manually:
      pip uninstall -y jax jaxlib
      pip install "jax[cuda12-local]>=0.6.0"

    See README.md GPU Installation section for details.

    To suppress this warning:
      export NLSQ_SKIP_GPU_CHECK=1

    Notes
    -----
    This check runs automatically on import but has minimal overhead:
    - Subprocess call to nvidia-smi (~5ms)
    - JAX device query (~1ms)
    - Only prints warning when mismatch detected
    - Silent failures prevent disruption
    - Can be suppressed with NLSQ_SKIP_GPU_CHECK environment variable
    """
    # Early exit if user wants to skip GPU check
    if os.environ.get("NLSQ_SKIP_GPU_CHECK", "").lower() in ("1", "true", "yes"):
        return

    try:
        # Check if nvidia-smi detects GPU hardware
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()

            # Check if JAX is using GPU
            import jax

            devices = jax.devices()
            using_gpu = any(
                "cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices
            )

            if not using_gpu:
                # Sanitize GPU name to prevent display issues
                # Limit to 100 chars and convert to ASCII
                gpu_name_safe = (
                    gpu_name[:100].encode("ascii", "replace").decode("ascii")
                )

                print("\n⚠️  GPU ACCELERATION AVAILABLE")
                print("═══════════════════════════════")
                print(f"NVIDIA GPU detected: {gpu_name_safe}")
                print("JAX is currently using: CPU-only")
                print("\nEnable 150-270x speedup with GPU acceleration:")
                print("  make install-jax-gpu")
                print("\nOr manually:")
                print("  pip uninstall -y jax jaxlib")
                print('  pip install "jax[cuda12-local]>=0.6.0"')
                print("\nSee README.md GPU Installation section for details.\n")

    except (subprocess.TimeoutExpired, FileNotFoundError, ImportError):
        # nvidia-smi not found or JAX not installed - silently skip
        pass
    except RuntimeError:
        # Unexpected runtime error - silently skip to avoid disrupting workflow
        pass
