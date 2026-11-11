"""Parameter covariance from a Jacobian (SVD) or from the normal matrix (eigh).

Two bodies, moved unchanged from ``CurveFit._compute_covariance`` and
``exact_lm.compute_covariance``. Their rank cutoffs deliberately differ and are NOT
unified: ``eps * max(n_eff, p) * s0`` on the Jacobian's singular values versus
``p * eps * lambda_max`` on the eigenvalues of ``H = J^T J``.

Both take ``cost = 0.5 * sum(r^2)`` (``OptimizeResult.cost``) and apply the factor 2
once, internally: ``pcov = pinv * 2 * cost / (n_eff - p)`` unless ``absolute_sigma``.
Both return ``(pcov, fallback_fired)``; ``fallback_fired`` is True when the all-``inf``
fallback was used (and an ``OptimizeWarning`` emitted).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from nlsq.result import OptimizeWarning
from nlsq.utils.logging import get_logger

_logger = get_logger("curve_fit")

COND_WARN = 1e4  # warn when sqrt(cond(H)), the cond(J) estimate, exceeds this
# warnings point at the user's call site, whether exact_lm_fit or curve_fit_large was called
_NLSQ_DIR = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep,)


def pcov_from_jacobian(
    J: Any,
    cost: float,
    n_eff: int,
    p: int,
    absolute_sigma: bool,
    svd: Callable[[Any], tuple[Any, Any]],
    logger: Any = None,
) -> tuple[np.ndarray, bool]:
    """Covariance from the SVD of ``J`` (Moore-Penrose, as ``scipy.optimize.curve_fit``).

    ``svd(J) -> (s, VT)`` is injected so ``CurveFit.covariance_svd`` stays the hook.
    ``n_eff`` is the true data count (``J`` may carry padded rows); it sets the
    singular-value cutoff ``eps * max(n_eff, J.shape[1]) * s[0]`` and the degrees of
    freedom ``n_eff - p``. Fallback messages go to ``logger`` (default: the
    ``curve_fit`` module logger; ``CurveFit`` passes ``self.logger``). The warning is
    attributed to the first frame outside ``nlsq/``.
    """
    cost = 2 * cost  # cost is half the sum of squares!

    # Do Moore-Penrose inverse discarding zero singular values
    outputs = svd(J)
    # Convert JAX arrays to NumPy more efficiently using np.asarray
    s, VT = (np.asarray(output) for output in outputs)

    n_params = J.shape[1]
    # NaN/Inf in J (e.g. a numerically unstable model at the solution)
    # propagates into s via SVD; without this check, the threshold below is
    # NaN and `s > threshold` is all-False, silently producing a deceptive
    # all-ZERO pcov (reads as "perfectly determined") instead of the correct
    # all-inf fallback.
    singular = len(s) == 0 or not np.all(np.isfinite(s))
    if not singular:
        # Use n_eff (true data count) not J.shape[0] (may be padded) for
        # the threshold, so padding doesn't cause over-aggressive SV truncation.
        threshold = np.finfo(float).eps * max(n_eff, J.shape[1]) * s[0]
        valid_mask = s > threshold
        s = s[valid_mask]
        VT = VT[valid_mask]
        singular = len(s) == 0

    rank_deficient = not singular and len(s) < n_params
    warn_cov = False
    if singular:
        pcov = np.full((n_params, n_params), np.inf)
        warn_cov = True
        reason = "singular_jacobian"
    else:
        pcov = np.dot(VT.T / s**2, VT)
        reason = None
        if not absolute_sigma:
            if n_eff > p:
                s_sq = cost / (n_eff - p)
                pcov = pcov * s_sq
            else:
                pcov.fill(np.inf)
                warn_cov = True
                reason = "insufficient_data"

    if rank_deficient:
        # Pseudo-inverse (SciPy-trf compatible) gives zero variance along the
        # null space, so unidentifiable parameters would read as known exactly.
        rank_msg = (
            f"Jacobian is rank-deficient (rank {len(s)} < {n_params} parameters); "
            "some parameters are not identifiable and their reported "
            "uncertainty is not meaningful"
        )
        (_logger if logger is None else logger).warning(
            rank_msg, reason="rank_deficient"
        )
        warnings.warn(rank_msg, OptimizeWarning, skip_file_prefixes=_NLSQ_DIR)

    if warn_cov:
        (_logger if logger is None else logger).warning(
            "Covariance could not be estimated", reason=reason
        )
        warnings.warn(
            "Covariance of the parameters could not be estimated",
            OptimizeWarning,
            skip_file_prefixes=_NLSQ_DIR,
        )

    return pcov, warn_cov


def warn_not_estimated() -> None:
    """Emit the all-``inf`` fallback ``OptimizeWarning``, attributed to the user's call site."""
    warnings.warn(
        "Covariance of the parameters could not be estimated",
        OptimizeWarning,
        skip_file_prefixes=_NLSQ_DIR,
    )


def _power_of_two_scale(H: np.ndarray) -> float:
    """A power of two ``s`` with ``max|H/s| < 2``: scaling is exact, and ``H + H.T`` of
    ``H/s`` cannot overflow (it would for ``|H| > 1e308 / 2``)."""
    top = np.max(np.abs(H))
    return float(np.ldexp(1.0, int(np.frexp(top)[1]) - 1)) if top > 0 else 1.0


def normal_condition(H: np.ndarray) -> float:
    """``sqrt(lambda_max / lambda_min)`` of symmetric ``H = J^T J``, the ``cond(J)`` estimate.

    Overflow-safe (eigenvalues of the power-of-two scaled ``H``). ``inf`` when ``H`` is
    non-finite or ``lambda_min <= 0``.
    """
    H = np.asarray(H, dtype=np.float64)
    if not np.isfinite(H).all():
        return np.inf
    Hs = H / _power_of_two_scale(H)
    w = np.linalg.eigvalsh((Hs + Hs.T) / 2)  # ascending
    if w[0] <= 0:
        return np.inf
    with np.errstate(over="ignore"):  # the ratio may overflow to inf
        return float(np.sqrt(w[-1] / w[0]))


def pcov_from_normal(
    H: np.ndarray,
    cost: float,
    n_eff: int,
    p: int,
    absolute_sigma: bool,
    cond_warn: bool = True,
) -> tuple[np.ndarray, bool]:
    """Covariance from ``H = J^T J`` with the ``nlsq.curve_fit`` contract.

    Pseudo-inverse over the eigen-directions with ``lambda_i > p*eps*lambda_max`` (dropped
    directions give zero rows and columns), scaled by ``2*cost/(n_eff-p)`` unless
    ``absolute_sigma``. All-``inf`` with an :class:`OptimizeWarning` if no direction
    survives, any value is non-finite, or ``n_eff <= p`` without ``absolute_sigma``.
    A ``UserWarning`` is emitted when ``sqrt(cond(H)) > COND_WARN`` unless
    ``cond_warn`` is False (the hybrid optimizer, whose callers never got that warning).
    """
    n = n_eff
    H = np.asarray(H, dtype=np.float64)
    pcov = np.full((p, p), np.inf)
    if np.isfinite(H).all():
        s = _power_of_two_scale(H)
        Hs = H / s
        w, V = np.linalg.eigh((Hs + Hs.T) / 2)  # ascending
        keep = w > p * np.finfo(np.float64).eps * w[-1]
        if keep.any():
            if cond_warn and normal_condition(H) > COND_WARN:
                warnings.warn(
                    f"normal-equation Hessian is ill-conditioned (sqrt(cond(H)) > "
                    f"{COND_WARN:.0e}); the covariance may be inaccurate and the "
                    "parameters may stop short of the optimum: check the result",
                    UserWarning,
                    skip_file_prefixes=_NLSQ_DIR,
                )
            Vk = V[:, keep]
            # overflow is caught below as all-inf + OptimizeWarning; silence numpy's own
            with np.errstate(over="ignore", invalid="ignore"):
                cov = ((Vk / w[keep]) @ Vk.T) / s  # pinv(H) = pinv(H/s) / s
                if absolute_sigma:
                    pcov = cov
                elif n > p:
                    pcov = cov * (2.0 * cost / (n - p))
    if not np.isfinite(pcov).all():
        pcov = np.full((p, p), np.inf)
    fired = bool(np.isinf(pcov).any())
    if fired:
        warn_not_estimated()
    return pcov, fired
