"""GPU detection and warning utilities (System CUDA version).

This module provides runtime GPU availability checks and system CUDA detection
to help users realize when GPU acceleration is available but not being used.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import warnings

from nlsq.config import _env_flag

_logger = logging.getLogger(__name__)


def _device_platform(d: object) -> str:
    """Lower-case platform of a JAX device ("gpu", "tpu", "cpu", ...).

    Uses ``.platform`` when present, else falls back to substring matching on
    ``str(d)`` for mocks/older JAX ("cuda"/"rocm"/"gpu" -> "gpu", "tpu" -> "tpu").
    """
    platform = getattr(d, "platform", None)
    if isinstance(platform, str) and platform:
        platform = platform.lower()
        return "gpu" if platform in ("cuda", "rocm") else platform
    s = str(d).lower()
    if "tpu" in s:
        return "tpu"
    if "cuda" in s or "gpu" in s or "rocm" in s:
        return "gpu"
    return "cpu"


def _is_gpu_device(d: object) -> bool:
    """True if a JAX device is a GPU (TPUs are not GPUs)."""
    return _device_platform(d) == "gpu"


def _is_accelerator_device(d: object) -> bool:
    """True if a JAX device is a GPU or TPU (any vendor/backend).

    Prefers the real ``.platform`` attribute ("gpu"/"tpu"/"cpu") that JAX
    devices expose; falls back to substring matching on ``str(d)`` for
    mocks/older JAX that lack it. The old code only matched "cuda", which
    silently reported TPU and ROCm devices as CPU-only.
    """
    return _device_platform(d) in ("gpu", "tpu")


def get_system_cuda_version() -> tuple[str | None, int | None]:
    """Detect system CUDA version from nvcc.

    Returns
    -------
    tuple[str | None, int | None]
        Tuple of (full_version, major_version) or (None, None) if not found.
        Example: ("12.6", 12) or ("13.0", 13)
    """
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0:
            # Parse "release X.Y" from output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    # Extract version like "12.6" from "release 12.6, V12.6.77"
                    parts = line.split("release")[-1].strip()
                    version = parts.split(",")[0].strip()
                    major = int(version.split(".")[0])
                    return version, major

    except subprocess.TimeoutExpired:
        _logger.debug("nvcc timed out")
    except FileNotFoundError:
        _logger.debug("nvcc not found")
    except (ValueError, IndexError) as e:
        _logger.debug(f"Failed to parse CUDA version: {e}")
    except Exception as e:
        _logger.debug(f"CUDA detection failed: {e}")

    return None, None


def get_gpu_info() -> tuple[str | None, float | None]:
    """Detect GPU name and SM version.

    Returns
    -------
    tuple[str | None, float | None]
        Tuple of (gpu_name, sm_version) or (None, None) if not found.
        Example: ("NVIDIA GeForce RTX 4090", 8.9)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            if len(parts) >= 2:
                gpu_name = parts[0]
                sm_version = float(parts[1])
                return gpu_name, sm_version

    except subprocess.TimeoutExpired:
        _logger.debug("nvidia-smi timed out")
    except FileNotFoundError:
        _logger.debug("nvidia-smi not found")
    except (ValueError, IndexError) as e:
        _logger.debug(f"Failed to parse GPU info: {e}")
    except Exception as e:
        _logger.debug(f"GPU detection failed: {e}")

    return None, None


def check_plugin_conflicts() -> list[str]:
    """Check for known JAX CUDA plugin conflicts.

    Returns
    -------
    list[str]
        List of issue descriptions (empty means no issues detected).
    """
    issues: list[str] = []
    try:
        import importlib.metadata as md

        jaxlib_v = md.version("jaxlib")

        cuda12 = cuda13 = None
        with contextlib.suppress(md.PackageNotFoundError):
            cuda12 = md.version("jax-cuda12-plugin")
        with contextlib.suppress(md.PackageNotFoundError):
            cuda13 = md.version("jax-cuda13-plugin")

        # Check for dual plugin conflict
        if cuda12 and cuda13:
            issues.append(
                f"Both cuda12 ({cuda12}) and cuda13 ({cuda13}) plugins installed. "
                "Only ONE can be active — this causes PJRT registration conflicts.",
            )

        # Check for version mismatch
        for name, version in [("cuda12", cuda12), ("cuda13", cuda13)]:
            if version and version != jaxlib_v:
                issues.append(
                    f"jax-{name}-plugin {version} != jaxlib {jaxlib_v}. "
                    "Plugin version must exactly match jaxlib.",
                )

    except Exception as e:
        _logger.debug(f"Plugin conflict check failed: {e}")

    return issues


def get_recommended_package() -> str | None:
    """Get recommended JAX package based on system CUDA.

    Returns
    -------
    str | None
        Package name like "jax[cuda12-local]" or "jax[cuda13-local]",
        or None if no compatible setup found.
    """
    # GPU acceleration is Linux-only
    if sys.platform != "linux":
        return None

    _cuda_version, cuda_major = get_system_cuda_version()
    _gpu_name, sm_version = get_gpu_info()

    if cuda_major is None:
        _logger.debug("No system CUDA detected")
        return None

    if sm_version is None:
        _logger.debug("No GPU detected")
        return None

    # Check compatibility
    if cuda_major == 13:
        if sm_version >= 7.5:
            return "jax[cuda13-local]"
        _logger.debug(f"GPU SM {sm_version} doesn't support CUDA 13")
        return None
    if cuda_major == 12:
        if sm_version >= 5.2:
            return "jax[cuda12-local]"
        _logger.debug(f"GPU SM {sm_version} too old for CUDA 12")
        return None
    _logger.debug(f"CUDA {cuda_major} not supported")
    return None


def check_gpu_availability(warn: bool = True, *, warn_on_error: bool = True) -> bool:
    """Check if GPU is available but not being used by JAX.

    Prints a helpful warning if GPU hardware and system CUDA are detected
    but JAX is running in CPU-only mode.

    Parameters
    ----------
    warn : bool, default=True
        If True, print warning when GPU available but not used.
    warn_on_error : bool, default=True
        If True (and ``warn`` is True), emit a ``UserWarning`` when the check itself
        fails (e.g. the JAX backend cannot initialise). ``import nlsq`` passes False
        so that importing never warns about a broken backend.

    Returns
    -------
    bool
        True if GPU is being used by JAX, False otherwise.

    Environment Variables
    ---------------------
    NLSQ_SKIP_GPU_CHECK : str, optional
        Set to "1", "true", "yes", or "on" (case-insensitive, surrounding whitespace
        ignored) to suppress GPU warnings.
        Useful for CI/CD pipelines or users who intentionally use CPU-only JAX.

    Notes
    -----
    This check runs automatically on import but has minimal overhead:
    - Subprocess call to nvidia-smi (~5ms)
    - JAX device query (~1ms)
    - Only prints warning when mismatch detected
    - Silent failures prevent disruption
    """
    # Early exit if user wants to skip GPU check
    if _env_flag("NLSQ_SKIP_GPU_CHECK"):
        return False

    # GPU acceleration is Linux-only; skip detection on other platforms
    if sys.platform != "linux":
        return False

    try:
        gpu_name, sm_version = get_gpu_info()
        cuda_version, cuda_major = get_system_cuda_version()

        # No NVIDIA hardware: return before touching jax.devices(). This runs on
        # every `import nlsq`; querying devices would initialize the JAX backend
        # at import time and lock `jax_platforms` before the user can set it.
        if gpu_name is None:
            _logger.debug("No GPU hardware detected")
            return False

        # Check if JAX is using GPU
        import jax

        devices = jax.devices()
        using_gpu = any(_is_gpu_device(d) for d in devices)

        if using_gpu:
            _logger.debug(f"JAX is using GPU: {devices}")
            # Check for plugin issues even when GPU works
            for issue in check_plugin_conflicts():
                _logger.warning(f"Plugin issue: {issue}")
            return True

        # GPU available but not being used
        if warn:
            plugin_issues = check_plugin_conflicts()
            _print_gpu_warning(
                gpu_name,
                sm_version,
                cuda_version,
                cuda_major,
                plugin_issues,
            )

        return False

    except ImportError:
        _logger.debug("JAX not installed")
        return False
    except Exception as e:
        _logger.debug(f"GPU check failed: {e}")
        if warn and warn_on_error:  # a GPU is present (or JAX is broken)
            warnings.warn(f"GPU check failed: {e}", UserWarning, stacklevel=2)
        return False


def _print_gpu_warning(
    gpu_name: str,
    sm_version: float | None,
    cuda_version: str | None,
    cuda_major: int | None,
    plugin_issues: list[str] | None = None,
) -> None:
    """Print the "GPU available but not used" advisory to stderr.

    stderr, not stdout: this runs on ``import nlsq`` and must not precede machine-readable
    stdout output (``nlsq fit ... --stdout`` prints JSON).
    """

    def out(text: str = "") -> None:
        print(text, file=sys.stderr)

    out("\nGPU AVAILABLE BUT NOT USED")
    out("===========================")
    out(f"  GPU: {gpu_name} (SM {sm_version})")
    out(f"  System CUDA: {cuda_version or 'Not found'}")
    out("  JAX backend: CPU-only")

    if plugin_issues:
        out("\n  Issues detected:")
        for issue in plugin_issues:
            out(f"    - {issue}")

    out()

    # Same compatibility rules as get_recommended_package(): only these CUDA/SM
    # pairs have a JAX wheel, so never suggest one that check would reject.
    supported = (cuda_major == 13 and (sm_version or 0) >= 7.5) or (
        cuda_major == 12 and (sm_version or 0) >= 5.2
    )
    if not supported:
        out("To enable GPU acceleration:")
        out("  1. Install CUDA toolkit (12.x or 13.x) supporting this GPU")
        out("  2. Ensure nvcc is in PATH")
        out("  3. Run: make install-jax-gpu")
    else:
        pkg = f"jax[cuda{cuda_major}-local]"
        out("Enable GPU acceleration:")
        out("  make install-jax-gpu")
        out()
        out("Or manually:")
        out(
            "  pip uninstall -y jax jaxlib "
            "jax-cuda13-plugin jax-cuda13-pjrt "
            "jax-cuda12-plugin jax-cuda12-pjrt",
        )
        out(f'  pip install "{pkg}"')

    out("\nNote: GPU acceleration is supported on Linux only.")
    out("See README.md for details.\n")


def get_device_info() -> dict:
    """Get comprehensive device information.

    Returns
    -------
    dict
        Dictionary with:
        - jax_version: JAX version string
        - jax_backend: Current backend (cpu, gpu)
        - devices: List of device strings
        - gpu_count: Number of GPU devices (TPUs are not counted)
        - tpu_count: Number of TPU devices
        - using_gpu: Boolean
        - gpu_hardware: GPU name
        - gpu_sm_version: SM version (float)
        - system_cuda_version: System CUDA version string
        - system_cuda_major: System CUDA major version (int)
        - recommended_package: Recommended JAX package
        - plugin_issues: List of detected plugin conflict descriptions
        - jax_error: Error text when JAX is importable but its backend cannot
          initialise (e.g. ``JAX_PLATFORMS=bogus``), else None
    """
    info: dict = {
        "jax_version": None,
        "jax_backend": None,
        "devices": [],
        "gpu_count": 0,
        "tpu_count": 0,
        "using_gpu": False,
        "gpu_hardware": None,
        "gpu_sm_version": None,
        "system_cuda_version": None,
        "system_cuda_major": None,
        "recommended_package": None,
        "plugin_issues": [],
        "jax_error": None,
    }

    # JAX info
    try:
        import jax

        info["jax_version"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
        devices = jax.devices()
        info["devices"] = [str(d) for d in devices]
        info["gpu_count"] = sum(1 for d in devices if _is_gpu_device(d))
        info["tpu_count"] = sum(1 for d in devices if _device_platform(d) == "tpu")
        info["using_gpu"] = info["gpu_count"] > 0
    except ImportError:
        pass
    except (
        Exception
    ) as exc:  # backend init failure: report it, don't crash a diagnostic
        info["jax_error"] = f"{type(exc).__name__}: {exc}"

    # GPU hardware info
    gpu_name, sm_version = get_gpu_info()
    info["gpu_hardware"] = gpu_name
    info["gpu_sm_version"] = sm_version

    # System CUDA info
    cuda_version, cuda_major = get_system_cuda_version()
    info["system_cuda_version"] = cuda_version
    info["system_cuda_major"] = cuda_major

    # Recommended package and plugin diagnostics
    info["recommended_package"] = get_recommended_package()
    info["plugin_issues"] = check_plugin_conflicts()

    return info
