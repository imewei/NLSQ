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

    Dimensions with a finite range use ``lb + u * (ub - lb)``. Dimensions where
    ``ub - lb`` is infinite or NaN (one-sided or unbounded) sample
    ``center + (u - 0.5) * 2 * halfwidth`` instead and are clipped into
    ``[lb, ub]``, so the result is always finite when ``center`` and
    ``halfwidth`` are.
    """
    u = np.asarray(u, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    halfwidth = np.asarray(halfwidth, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        span = ub - lb
    finite = np.isfinite(span)
    boxed = np.where(finite, lb, 0.0) + u * np.where(finite, span, 0.0)
    open_ = center + (u - 0.5) * 2.0 * halfwidth
    return np.clip(np.where(finite, boxed, open_), lb, ub)
