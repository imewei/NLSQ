"""Device placement policy for small exact SVDs under an accelerator backend.

A GPU SVD of a tall-skinny Jacobian has ~1 ms of launch/solver overhead regardless
of size; below ``HOST_SVD_MAX_ELEMENTS`` the same SVD on the CPU device, including
the device-to-host copy of ``J`` and the copy of the results back, is faster
(benchmarks/baselines/2026-11-06-perf-phase0.md item (c)). The host path costs one
temporary n x p host copy of the Jacobian.

Two limits: ``HOST_SVD_MAX_ELEMENTS`` (30_000) for the per-iteration TRF full SVD, and
``HOST_COV_SVD_MAX_ELEMENTS`` (150_000) for the covariance SVD only
(``minpack._covariance_svd``, benchmarks/baselines/2026-11-10-qr-covariance-ab.md).
"""

from __future__ import annotations

import functools

import jax

# Largest n x p at which host-incl.-transfer beat the GPU in every sweep (p = 3 and 6).
HOST_SVD_MAX_ELEMENTS: int = 30_000

# Covariance-only limit (the QR-then-SVD kernel in ``minpack._covariance_svd``): host won at every
# size up to 150,000 elements (benchmarks/baselines/2026-11-10-qr-covariance-ab.md). Kept separate
# because the TRF per-iteration SVD shares ``HOST_SVD_MAX_ELEMENTS`` and was not re-measured.
HOST_COV_SVD_MAX_ELEMENTS: int = 150_000

_GPU_BACKENDS = frozenset({"gpu", "cuda", "rocm"})


@functools.cache
def _cpu_platform_available() -> bool:
    """False under e.g. ``JAX_PLATFORMS=cuda``, where there is no CPU device to use."""
    try:
        return bool(jax.devices("cpu"))
    except RuntimeError:
        return False


def use_host_for_small_svd(
    shape: tuple[int, ...],
    backend: str,
    max_elements: int | None = None,
) -> bool:
    """True when a ``shape`` SVD should run on the CPU device under ``backend``.

    ``max_elements`` defaults to ``HOST_SVD_MAX_ELEMENTS`` (read at call time).
    """
    limit = HOST_SVD_MAX_ELEMENTS if max_elements is None else max_elements
    return (
        backend in _GPU_BACKENDS
        and shape[0] * shape[1] <= limit
        and _cpu_platform_available()
    )


def host_device() -> jax.Device:
    """The CPU device used for host-side SVDs."""
    return jax.devices("cpu")[0]


def use_host_svd_for(J: object, max_elements: int | None = None) -> bool:
    """Policy for a concrete array ``J``: small, one accelerator device, never a tracer.

    Tracers (``J`` inside ``jax.jit``) and non-JAX arrays keep the plain jitted path.
    """
    return (
        isinstance(J, jax.Array)
        and not isinstance(J, jax.core.Tracer)
        and len(J.devices()) == 1
        and use_host_for_small_svd(
            J.shape, jax.default_backend(), max_elements=max_elements
        )
    )
