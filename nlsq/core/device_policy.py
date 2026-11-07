"""Device placement policy for small exact SVDs under an accelerator backend.

A GPU SVD of a tall-skinny Jacobian has ~1 ms of launch/solver overhead regardless
of size; below ``HOST_SVD_MAX_ELEMENTS`` the same SVD on the CPU device, including
the device-to-host copy of ``J`` and the copy of the results back, is faster
(benchmarks/baselines/2026-11-06-perf-phase0.md item (c)). The host path costs one
temporary n x p host copy of the Jacobian.
"""

from __future__ import annotations

import functools

import jax

# Largest n x p at which host-incl.-transfer beat the GPU in every sweep (p = 3 and 6).
HOST_SVD_MAX_ELEMENTS: int = 30_000

_GPU_BACKENDS = frozenset({"gpu", "cuda", "rocm"})


@functools.cache
def _cpu_platform_available() -> bool:
    """False under e.g. ``JAX_PLATFORMS=cuda``, where there is no CPU device to use."""
    try:
        return bool(jax.devices("cpu"))
    except RuntimeError:
        return False


def use_host_for_small_svd(shape: tuple[int, ...], backend: str) -> bool:
    """True when a ``shape`` SVD should run on the CPU device under ``backend``."""
    return (
        backend in _GPU_BACKENDS
        and shape[0] * shape[1] <= HOST_SVD_MAX_ELEMENTS
        and _cpu_platform_available()
    )


def host_device() -> jax.Device:
    """The CPU device used for host-side SVDs."""
    return jax.devices("cpu")[0]
