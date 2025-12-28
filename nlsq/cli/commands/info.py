"""Info command handler for NLSQ CLI.

This module provides the 'nlsq info' command for displaying system and
environment information.

Example Usage
-------------
From command line:
    $ nlsq info
    $ nlsq info --verbose

From Python:
    >>> from nlsq.cli.commands.info import run_info
    >>> run_info()
"""

import inspect
import sys
from typing import Any

import nlsq
import nlsq.core.functions


def run_info(verbose: bool = False) -> None:
    """Display system and environment information.

    Shows:
    - NLSQ version
    - Python version
    - JAX backend (CPU/GPU)
    - Available GPU devices with device IDs
    - System memory info
    - List of builtin models with parameter signatures

    Parameters
    ----------
    verbose : bool
        If True, show additional details.
    """
    print("=" * 60)
    print("NLSQ System Information")
    print("=" * 60)
    print()

    # NLSQ Version
    print(f"NLSQ Version: {nlsq.__version__}")
    print()

    # Python Version
    print(f"Python Version: {sys.version}")
    print()

    # JAX Backend and Devices
    _print_jax_info(verbose)

    # Memory Info
    _print_memory_info()

    # Builtin Models
    _print_builtin_models(verbose)

    print("=" * 60)


def _print_jax_info(verbose: bool) -> None:
    """Print JAX backend and device information.

    Parameters
    ----------
    verbose : bool
        Show additional JAX details.
    """
    print("JAX Configuration:")
    print("-" * 40)

    try:
        import jax

        # JAX version
        print(f"  JAX Version: {jax.__version__}")

        # Get devices
        devices = jax.devices()

        # Determine backend
        backends = set()
        for device in devices:
            platform = getattr(device, "platform", "unknown")
            backends.add(platform)

        # Print backend summary
        if "gpu" in backends or "cuda" in backends:
            print("  Backend: GPU (CUDA)")
        elif "tpu" in backends:
            print("  Backend: TPU")
        else:
            print("  Backend: CPU")

        # Print device details
        print(f"  Devices: {len(devices)}")

        for i, device in enumerate(devices):
            platform = getattr(device, "platform", "unknown")
            device_kind = getattr(device, "device_kind", "")

            if platform in ("gpu", "cuda"):
                # GPU device
                print(f"    [{i}] GPU: {device_kind}")
            elif platform == "tpu":
                print(f"    [{i}] TPU: {device_kind}")
            else:
                print(f"    [{i}] CPU")

        print()

    except ImportError:
        print("  JAX: Not installed")
        print()
    except Exception as e:
        print(f"  JAX: Error detecting - {e}")
        print()


def _print_memory_info() -> None:
    """Print system memory information."""
    print("System Memory:")
    print("-" * 40)

    try:
        import psutil

        mem = psutil.virtual_memory()

        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        percent = mem.percent

        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Available: {available_gb:.1f} GB")
        print(f"  Used: {used_gb:.1f} GB ({percent:.1f}%)")

        print()

    except ImportError:
        print("  psutil: Not installed (memory info unavailable)")
        print()
    except Exception as e:
        print(f"  Error detecting memory: {e}")
        print()


def _print_builtin_models(verbose: bool) -> None:
    """Print list of builtin models with signatures.

    Parameters
    ----------
    verbose : bool
        Show detailed parameter information.
    """
    print("Builtin Models:")
    print("-" * 40)

    # Get all models from nlsq.core.functions.__all__
    models = []
    for name in nlsq.core.functions.__all__:
        func = getattr(nlsq.core.functions, name, None)
        if func is not None and callable(func):
            models.append((name, func))

    for name, func in sorted(models):
        # Get function signature
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # First parameter is x (independent variable)
            if params and params[0] in ("x", "xdata"):
                fit_params = params[1:]
            else:
                fit_params = params

            # Format signature
            sig_str = ", ".join(fit_params)
            print(f"  {name}({sig_str})")

            # Get docstring summary if verbose
            if verbose and func.__doc__:
                doc_lines = func.__doc__.strip().split("\n")
                if doc_lines:
                    first_line = doc_lines[0].strip()
                    if first_line:
                        print(f"      {first_line}")

        except (ValueError, TypeError):
            # Function has no inspectable signature (e.g., polynomial factory)
            print(f"  {name}(...)")

    print()


def get_info_dict() -> dict[str, Any]:
    """Get system information as a dictionary.

    Useful for programmatic access to system info.

    Returns
    -------
    dict
        Dictionary containing system information.
    """
    info: dict[str, Any] = {
        "nlsq_version": nlsq.__version__,
        "python_version": sys.version,
    }

    # JAX info
    try:
        import jax

        info["jax_version"] = jax.__version__

        devices = jax.devices()
        info["jax_devices"] = [
            {
                "id": i,
                "platform": getattr(d, "platform", "unknown"),
                "device_kind": getattr(d, "device_kind", ""),
            }
            for i, d in enumerate(devices)
        ]

        # Determine backend
        backends = {d.get("platform") for d in info["jax_devices"]}
        if "gpu" in backends or "cuda" in backends:
            info["jax_backend"] = "gpu"
        elif "tpu" in backends:
            info["jax_backend"] = "tpu"
        else:
            info["jax_backend"] = "cpu"

    except ImportError:
        info["jax_version"] = None
        info["jax_backend"] = None
        info["jax_devices"] = []

    # Memory info
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }
    except ImportError:
        info["memory"] = None

    # Builtin models
    info["builtin_models"] = list(nlsq.functions.__all__)

    return info
