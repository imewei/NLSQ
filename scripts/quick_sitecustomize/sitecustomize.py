"""
Lightweight runtime patches for running example scripts in quick mode.

Activated when NLSQ_EXAMPLES_QUICK=1. Patches reduce dataset sizes, enforce
deterministic RNG, and force non-interactive plotting to keep scripts fast and
repeatable during automated runs. Only used by the example-script test harness.
"""

from __future__ import annotations

import numbers
import os
import random
import sys
from collections.abc import Iterable


def _get_max_samples() -> int:
    try:
        return int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "5000"))
    except ValueError:
        return 5000


def _cap_size(
    size: int | Iterable[int] | None, max_samples: int
) -> int | tuple[int, ...] | None:
    if size is None:
        return None
    if isinstance(size, numbers.Integral):
        return min(int(size), max_samples)
    if isinstance(size, tuple):
        return tuple(min(int(s), max_samples) for s in size)
    # Fall back to original size for unexpected types
    return size


def _patch_numpy(max_samples: int) -> None:
    import numpy as np

    np.random.seed(0)
    random.seed(0)

    def _wrap_rng(fn):
        def wrapper(*args, **kwargs):
            args = list(args)
            if "size" in kwargs:
                kwargs["size"] = _cap_size(kwargs["size"], max_samples)
            elif args:
                if len(args) >= 3:
                    args[2] = _cap_size(args[2], max_samples)
                else:
                    args[0] = _cap_size(args[0], max_samples)
            return fn(*args, **kwargs)

        return wrapper

    rng = np.random
    rng.normal = _wrap_rng(rng.normal)
    rng.random = _wrap_rng(rng.random)
    rng.rand = _wrap_rng(rng.rand)
    rng.randn = _wrap_rng(rng.randn)
    rng.random_sample = _wrap_rng(rng.random_sample)
    rng.uniform = _wrap_rng(rng.uniform)

    linspace_orig = np.linspace

    def linspace_patched(start, stop, num=50, *args, **kwargs):
        caller_mod = sys._getframe(1).f_globals.get("__name__", "")
        if caller_mod.startswith(("matplotlib", "scipy")):
            capped_num = int(num)
        else:
            capped_num = min(int(num), max_samples)
        return linspace_orig(start, stop, capped_num, *args, **kwargs)

    np.linspace = linspace_patched

    logspace_orig = np.logspace

    def logspace_patched(start, stop, num=50, *args, **kwargs):
        caller_mod = sys._getframe(1).f_globals.get("__name__", "")
        if caller_mod.startswith(("matplotlib", "scipy")):
            capped_num = int(num)
        else:
            capped_num = min(int(num), max_samples)
        return logspace_orig(start, stop, capped_num, *args, **kwargs)

    np.logspace = logspace_patched

    geomspace_orig = np.geomspace

    def geomspace_patched(start, stop, num=50, *args, **kwargs):
        caller_mod = sys._getframe(1).f_globals.get("__name__", "")
        if caller_mod.startswith(("matplotlib", "scipy")):
            capped_num = int(num)
        else:
            capped_num = min(int(num), max_samples)
        return geomspace_orig(start, stop, capped_num, *args, **kwargs)

    np.geomspace = geomspace_patched


def _patch_jax(max_samples: int) -> None:
    try:
        import jax.numpy as jnp
    except Exception:
        return

    linspace_orig = jnp.linspace

    def linspace_patched(start, stop, num=50, *args, **kwargs):
        capped_num = min(int(num), max_samples)
        return linspace_orig(start, stop, capped_num, *args, **kwargs)

    jnp.linspace = linspace_patched

    logspace_orig = jnp.logspace

    def logspace_patched(start, stop, num=50, *args, **kwargs):
        capped_num = min(int(num), max_samples)
        return logspace_orig(start, stop, capped_num, *args, **kwargs)

    jnp.logspace = logspace_patched

    geomspace_orig = jnp.geomspace

    def geomspace_patched(start, stop, num=50, *args, **kwargs):
        capped_num = min(int(num), max_samples)
        return geomspace_orig(start, stop, capped_num, *args, **kwargs)

    jnp.geomspace = geomspace_patched


def _configure_matplotlib(tmp_dir: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    if tmp_dir:
        os.environ.setdefault("MPLCONFIGDIR", tmp_dir)


if os.environ.get("NLSQ_EXAMPLES_QUICK"):
    max_samples = _get_max_samples()
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_DISABLE_JIT", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    _patch_numpy(max_samples)
    _patch_jax(max_samples)
    _configure_matplotlib(os.environ.get("NLSQ_EXAMPLES_TMPDIR"))
