"""Parameter covariance from a Jacobian (SVD) or from the normal matrix (eigh).

Two bodies, moved unchanged from ``CurveFit._compute_covariance`` and
``exact_lm.compute_covariance``. Their rank cutoffs deliberately differ and are NOT
unified: ``eps * max(n_eff, p) * s0`` on the singular values of the column-equilibrated
Jacobian versus
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

import jax
import jax.numpy as jnp
import numpy as np

from nlsq.result import OptimizeWarning
from nlsq.utils.logging import get_logger

_logger = get_logger("curve_fit")


@jax.jit
def _column_norms(J: Any) -> Any:
    """Column 2-norms of ``J``, on ``J``'s own device (max-scaled: no over/underflow).

    Only ``p`` scalars leave the device. Reducing a GPU ``J`` on the host copies all of
    it: measured 45.8 ms on the host against 0.9 ms here for a 1M x 3 float64 ``J``.
    """
    col_max = jnp.max(jnp.abs(J), axis=0, initial=0.0)
    return col_max * jnp.linalg.norm(J / jnp.where(col_max > 0, col_max, 1.0), axis=0)


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

    ``svd(J_eq) -> (s, VT)`` is injected so ``CurveFit.covariance_svd`` stays the hook;
    it is called on ``J_eq = J / column_norms`` (columns equilibrated to unit norm) and
    ``pcov`` is scaled back afterwards. ``n_eff`` is the true data count (``J`` may carry
    padded rows); it sets the singular-value cutoff
    ``eps * max(n_eff, J.shape[1]) * s[0]`` on the equilibrated ``s`` and the degrees of
    freedom ``n_eff - p``. Fallback messages go to ``logger`` (default: the
    ``curve_fit`` module logger; ``CurveFit`` passes ``self.logger``). The warning is
    attributed to the first frame outside ``nlsq/``.
    """
    # Equilibrate columns to unit norm before the SVD so the rank cutoff sees
    # identifiability, not parameter units (x in 1e8 units made one direction look
    # "dropped" and its variance collapse); pcov is scaled back below. Departs from
    # SciPy's unscaled cutoff on purpose. Zero columns are left unscaled; if any
    # column norm is non-finite, no column is scaled.
    col_norm = np.asarray(_column_norms(J))
    if not np.all(np.isfinite(col_norm)):
        col_norm = np.ones_like(col_norm)
    col_norm = np.where(col_norm > 0, col_norm, 1.0)
    # Do Moore-Penrose inverse discarding zero singular values
    outputs = svd(J / col_norm)
    # Convert JAX arrays to NumPy more efficiently using np.asarray
    s, VT = (np.asarray(output) for output in outputs)
    s = s.astype(np.result_type(s.dtype, np.float64), copy=False)

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
        scaled_cost = (
            not absolute_sigma and n_eff > p and np.isfinite(cost) and cost >= 0
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if scaled_cost:
                scale = np.sqrt(cost) * np.sqrt(2.0 / (n_eff - p))
                scaled = (VT * scale) / s[:, None]
            else:
                scaled = VT / s[:, None]
            scaled = scaled / col_norm  # undo the equilibration before squaring
            pcov = scaled.T @ scaled
        reason = None
        if not absolute_sigma:
            if n_eff > p:
                if not scaled_cost:
                    with np.errstate(over="ignore", invalid="ignore"):
                        pcov = pcov * (2 * cost / (n_eff - p))
            else:
                pcov.fill(np.inf)
                warn_cov = True
                reason = "insufficient_data"
        # A non-finite cost with relative sigma keeps NaN/inf pcov, no warning: pinned
        # by tests/core/test_covariance_characterization.py (deliberate site difference).
        if (
            not warn_cov
            and (absolute_sigma or np.isfinite(cost))
            and not np.isfinite(pcov).all()
        ):
            pcov.fill(np.inf)
            warn_cov = True
            reason = "nonfinite_covariance"

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

    Inverse over the eigen-directions of the Jacobi-equilibrated ``H`` (unit diagonal,
    so independent of parameter units) when that matrix is full rank; otherwise the
    pseudo-inverse of ``H`` itself over ``lambda_i > p*eps*lambda_max`` (dropped
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
        # Jacobi-equilibrate so the rank cutoff sees identifiability, not parameter units
        # (columns of J in 1e9 units made well-determined directions look "dropped").
        # Divide by d twice (not by outer(d, d): denormal-safe); skip when the diagonal
        # is already within 1e8 (d within 1e4), keeping the cutoff exact for mild scales.
        dg = np.diag(Hs)
        d = np.sqrt(np.where(dg > 0, dg, 1.0))
        if d.max() <= 1e4 * d.min():
            d = np.ones_like(d)

        def _eig(dvec):
            Hn = (Hs / dvec[:, None]) / dvec[None, :]
            w, V = np.linalg.eigh((Hn + Hn.T) / 2)  # ascending
            return w, V, w > p * np.finfo(np.float64).eps * w[-1]

        w, V, keep = _eig(d)
        if not keep.all() and np.any(d != 1.0):
            # Rank-deficient even after equilibration: a pseudo-inverse is not
            # congruence-invariant (scaling changes its null space), so use the
            # unscaled Moore-Penrose semantics.
            d = np.ones_like(d)
            w, V, keep = _eig(d)
        if keep.any():
            if cond_warn and normal_condition(H) > COND_WARN:
                warnings.warn(
                    f"normal-equation Hessian is ill-conditioned (sqrt(cond(H)) > "
                    f"{COND_WARN:.0e}); the covariance may be inaccurate and the "
                    "parameters may stop short of the optimum: check the result",
                    UserWarning,
                    skip_file_prefixes=_NLSQ_DIR,
                )
            # pinv of H = D^-1 (pinv of Hn) D^-1
            Vk = V[:, keep] / d[:, None]
            # overflow is caught below as all-inf + OptimizeWarning; silence numpy's own
            with np.errstate(over="ignore", invalid="ignore"):
                if absolute_sigma:
                    pcov = ((Vk / w[keep]) @ Vk.T) / s
                elif n > p:
                    if np.isfinite(cost) and cost >= 0:
                        scaled = Vk * (np.sqrt(cost) * np.sqrt(2.0 / (n - p)))
                        scaled = scaled / np.sqrt(w[keep]) / np.sqrt(s)
                        pcov = scaled @ scaled.T
                    else:
                        cov = ((Vk / w[keep]) @ Vk.T) / s
                        pcov = cov * (2.0 * cost / (n - p))
    if not np.isfinite(pcov).all():
        pcov = np.full((p, p), np.inf)
    fired = bool(np.isinf(pcov).any())
    if fired:
        warn_not_estimated()
    return pcov, fired
