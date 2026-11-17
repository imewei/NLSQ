"""Map unit-cube samples into a (possibly half-infinite) bounds box."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sample_in_bounds(
    u: ArrayLike,
    lb: ArrayLike,
    ub: ArrayLike,
    center: ArrayLike,
    halfwidth: ArrayLike,
) -> NDArray[np.float64]:
    """Map unit samples ``u`` (shape ``(n, d)``) into ``[lb, ub]``.

    Dimensions with two finite endpoints use ``lb * (1 - u) + ub * u`` (safe from
    overflow of ``ub - lb``). Dimensions where either endpoint is infinite or
    NaN (one-sided or unbounded) sample
    ``center + (u - 0.5) * 2 * halfwidth`` instead and are clipped into
    ``[lb, ub]``. A result that would overflow float64 saturates at
    ``+-finfo(float64).max``, so the output is finite unless an input is NaN.
    """
    u, lb, ub, center, halfwidth = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (u, lb, ub, center, halfwidth))
    )
    # Finiteness is a property of each endpoint: ub - lb overflows for a wide
    # finite box, and lb*(1-u) + ub*u cannot.
    boxed = np.isfinite(lb) & np.isfinite(ub)
    out = np.empty(u.shape, dtype=np.float64)
    # Each branch is evaluated only where it applies, so a huge center/halfwidth
    # of an unused open dimension cannot raise (or poison) a boxed one.
    out[boxed] = lb[boxed] * (1.0 - u[boxed]) + ub[boxed] * u[boxed]
    op = ~boxed
    with np.errstate(over="ignore", invalid="ignore"):
        # (2u - 1) is in [-1, 1], so the product cannot overflow; only the sum can
        out[op] = center[op] + (2.0 * u[op] - 1.0) * halfwidth[op]
    # Unrepresentable open intervals saturate at the largest finite float
    fmax = np.finfo(np.float64).max
    out = np.clip(out, -fmax, fmax)
    with np.errstate(invalid="ignore"):
        return np.clip(out, lb, ub)
