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


def _column_norms_host(J: np.ndarray) -> np.ndarray:
    """``_column_norms`` in NumPy float64 (a jitted call would narrow to float32)."""
    with np.errstate(invalid="ignore"):
        col_max = np.max(np.abs(J), axis=0, initial=0.0)
        return col_max * np.linalg.norm(J / np.where(col_max > 0, col_max, 1.0), axis=0)


COND_WARN = 1e4  # warn when sqrt(cond(H)), the cond(J) estimate, exceeds this
# warnings point at the user's call site, whether exact_lm_fit or curve_fit_large was called
_NLSQ_DIR = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep,)


# The rank-deficient pcov is returned only if all three checks pass (otherwise all-inf
# plus an OptimizeWarning). kappa = max D / min D over the nonzero columns; the equilibrated
# V_r carries ~eps errors that the column norms D rescale, so the original-coordinate
# pseudo-inverse is off by ~eps * kappa in its dependent block and by ~eps * kappa^2 in
# structurally zero entries and in X @ n (measured against an mpmath reference).
# (0) eps * kappa and eps * s_1 / s_r (the retained equilibrated spectrum) must be
#     <= _KAPPA_TOL.
# (1) Probe: a rounding-level perturbation of V_r may move an entry by at most
#     _UNRESOLVED_TOL * |X_ij| + floor * sqrt(X_ii X_jj), floor = min(_UNRESOLVED_TOL,
#     max(_CROSS_FLOOR * _UNRESOLVED_TOL, 10 * eps * kappa^2)). The per-entry term polices
#     small cross-covariances (a plain sqrt(X_ii X_jj) scale hid them); the floor keeps
#     structurally zero cross terms from being judged against D's roundoff.
# (2) Null space: X @ n must vanish for every null vector n of J (original coordinates,
#     D^-1 v) to min(_NULL_CAP, max(_NULL_TOL, 10 * eps * kappa^2)) of the magnitude
#     sum_j |X_ij||n_j|, so cancellation of large terms is detected.
_UNRESOLVED_TOL = 1e-3
_CROSS_FLOOR = 1e-8  # relative correlation floor before the kappa^2 noise term
_NULL_TOL = 1e-6
_PROBE_ULPS = 4.0
_PROBE_DRAWS = 4
_KAPPA_TOL = 1e-5
_NULL_CAP = 1e-4


def _reduced_inverse(s: np.ndarray, V: np.ndarray, D: np.ndarray):
    """``(s_M, VT_M)`` of ``M = diag(s) V D`` (so ``pinv(J^T J) = VT_M^T diag(1/s_M^2) VT_M``)."""
    _, sm, VTm = np.linalg.svd((s[:, None] * V) * D, full_matrices=False)
    return sm, VTm


def _rank_deficient_factors(
    s: np.ndarray,
    VT: np.ndarray,
    VT_null: np.ndarray,
    col_norm: np.ndarray,
    zero_cols: np.ndarray,
    n_eff: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Factors of the original-coordinate pseudo-inverse, and whether it is unresolvable.

    Penrose conditions in equilibrated coordinates do not catch the failure (they hold
    to ~1e-14 for a wrong matrix), because ``D A_s D`` amplifies the rounding in ``A_s``.
    Two checks replace them (see ``_UNRESOLVED_TOL``): a backward-stability probe, which
    perturbs ``V_r`` by ``max(n_eff, p) * eps`` (the scale of the rank cutoff, four
    deterministic sign patterns), and the original-coordinate null-space test. Exactly
    zero Jacobian columns are structural: their row and column of ``pcov`` are exactly 0,
    so they are zeroed in ``V_r`` and never perturbed (the probe is then unit-free, as the
    perturbation is scaled by the true column norms ``D``).
    """
    VT = np.where(zero_cols, 0.0, VT)
    sm, VTm = _reduced_inverse(s, VT, col_norm)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sc = VTm / sm[:, None]
        X0 = sc.T @ sc
        d = np.sqrt(np.abs(np.diag(X0)))
        # Parameters with exactly zero variance are exactly unidentifiable: no scale to
        # compare against, so leave them out.
        used = d > 0
        ok = bool(np.isfinite(X0).all() and used.any())
        if ok:
            absX = np.abs(X0)
            nz = col_norm[~zero_cols]
            kappa = (
                nz.max() / nz.min() if nz.size else np.float64(1.0)
            )  # numpy: may overflow to inf
            # The equilibrated V_r carries ~eps errors that D rescales: the dependent
            # block of X is off by ~eps * kappa (measured, 5e-3 at kappa = 3e13), and
            # structurally zero entries / X @ n carry ~eps * kappa^2 noise.
            noise = eps * kappa**2
            null_tol = min(_NULL_CAP, max(_NULL_TOL, 10.0 * noise))
            cross_floor = min(
                _UNRESOLVED_TOL, max(_CROSS_FLOOR * _UNRESOLVED_TOL, 10.0 * noise)
            )
            # the retained equilibrated spectrum: the smallest retained value s_r of
            # ~eps * s_1 absolute error leaves the pseudo-inverse good to ~eps * s_1 / s_r
            if eps * kappa > _KAPPA_TOL or eps * s[0] / s[-1] > _KAPPA_TOL:
                ok = False
        if ok:
            for v_null in VT_null:
                v = np.where(zero_cols, 0.0, v_null)
                if np.linalg.norm(v) < np.sqrt(eps):  # a pure zero-column direction
                    continue
                n = v / col_norm
                n = n / np.max(np.abs(n))
                resid = np.abs(X0 @ n)
                weight = absX @ np.abs(n)
                if not np.all(resid <= null_tol * weight):  # also catches NaN
                    ok = False
                    break
        if ok:
            rng = np.random.default_rng(0)
            amp = _PROBE_ULPS * eps
            allowed = _UNRESOLVED_TOL * absX[np.ix_(used, used)]
            allowed = allowed + cross_floor * np.outer(d[used], d[used])
            for _ in range(_PROBE_DRAWS):
                signs = np.where(zero_cols, 0.0, rng.choice([-1.0, 1.0], size=VT.shape))
                s1, V1 = _reduced_inverse(s, VT + amp * signs, col_norm)
                sc1 = V1 / s1[:, None]
                dev = np.abs(X0 - sc1.T @ sc1)[np.ix_(used, used)]
                if not np.all(dev <= allowed):  # also catches NaN
                    ok = False
                    break
    return sm, VTm, not ok


def _numpy_svd(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Host float64 ``(s, VT)`` of ``J``, via QR as in ``CurveFit.covariance_svd``."""
    if not np.all(np.isfinite(J)):  # LAPACK raises; the JAX path yields NaN s
        k = min(J.shape)
        return np.full(k, np.nan), np.full((k, J.shape[1]), np.nan)
    _, s, VT = np.linalg.svd(np.linalg.qr(J, mode="r"), full_matrices=False)
    return s, VT


def _working_precision(J: Any, svd: Callable) -> tuple[Any, Callable, float]:
    """``(J, svd, eps)`` at float64 precision, or the widest JAX allows.

    A narrower J (float32) is upcast, its entries taken as exact: a float32 SVD has
    ~1e-7 roundoff that the float64 rank cutoff and acceptance checks would read as
    identifiable directions. With JAX x64 off a float64 J would be narrowed by the
    injected (JAX) SVD, so it is decomposed on the host in NumPy float64 instead.
    float64 inputs with x64 on pass through untouched.
    """
    wide = np.promote_types(J.dtype, np.float64)
    work = jax.dtypes.canonicalize_dtype(wide)
    if work != wide:
        return np.asarray(J, dtype=wide), _numpy_svd, float(np.finfo(wide).eps)
    if J.dtype != work:
        J = np.asarray(J, dtype=work)  # host widening: a device cast flushes subnormals
    return J, svd, float(np.finfo(work).eps)


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
    freedom ``n_eff - p``. When that cutoff finds ``J`` rank-deficient, ``pcov`` is the
    Moore-Penrose inverse in the original coordinates (``pinv(J.T @ J)``), taken from
    the SVD of ``M = S_r V_r^T D`` (the equilibrated factors cut to the detected rank,
    rescaled by the column norms ``D``). If that matrix fails the probe or null-space
    checks of ``_rank_deficient_factors`` it is not trusted: all-``inf`` with an
    ``OptimizeWarning`` instead. Fallback messages go to ``logger`` (default: the
    ``curve_fit`` module logger; ``CurveFit`` passes ``self.logger``). The warning is
    attributed to the first frame outside ``nlsq/``.
    """
    # Equilibrate columns to unit norm before the SVD so the rank cutoff sees
    # identifiability, not parameter units (x in 1e8 units made one direction look
    # "dropped" and its variance collapse); pcov is scaled back below. Departs from
    # SciPy's unscaled cutoff on purpose. Zero columns are left unscaled; if any
    # column norm is non-finite, no column is scaled.
    J, svd, eps = _working_precision(J, svd)
    col_norm = np.asarray(
        _column_norms_host(J) if svd is _numpy_svd else _column_norms(J)
    )
    if not np.all(np.isfinite(col_norm)):
        col_norm = np.ones_like(col_norm)
    zero_cols = col_norm == 0
    col_norm = np.where(zero_cols, 1.0, col_norm)
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
        threshold = eps * max(n_eff, J.shape[1]) * s[0]
        valid_mask = s > threshold
        s = s[valid_mask]
        VT_null = VT[~valid_mask]
        VT = VT[valid_mask]
        singular = len(s) == 0

    rank_deficient = not singular and len(s) < n_params
    unresolved = False
    if rank_deficient:
        # A pseudo-inverse is not invariant under the column equilibration, so the
        # equilibrated SVD (J = J_s D, J_s = U S V^T) only DETECTED the rank. J^T J =
        # M^T M for the small r x p matrix M = S_r V_r^T D, so pinv(J^T J) comes from
        # the SVD of M, built from the well-resolved equilibrated factors (an SVD of the
        # raw J loses a small identifiable column to roundoff in large ones).
        s, VT, unresolved = _rank_deficient_factors(
            s, VT, VT_null, col_norm, zero_cols, n_eff, eps
        )
        col_norm = np.ones_like(col_norm)
        singular = unresolved
    warn_cov = False
    if singular:
        pcov = np.full((n_params, n_params), np.inf)
        warn_cov = True
        reason = "unresolved_rank_deficiency" if unresolved else "singular_jacobian"
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
                if cost < 0:  # negative variances are meaningless
                    pcov.fill(np.inf)
                    warn_cov = True
                    reason = "negative_cost"
                elif not scaled_cost:
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
        rank_msg = _rank_deficient_message(len(s), n_params)
        (_logger if logger is None else logger).warning(
            rank_msg, reason="rank_deficient"
        )
        warnings.warn(rank_msg, OptimizeWarning, skip_file_prefixes=_NLSQ_DIR)

    if unresolved:
        unresolved_msg = (
            "The rank-deficient covariance cannot be resolved in the original parameter "
            "coordinates at this column scaling (rounding-level changes to J move it "
            "by more than "
            f"{_UNRESOLVED_TOL:.0e}); rescale the parameters (or fit an equilibrated "
            "parameterization) so the columns of J have comparable norms"
        )
        (_logger if logger is None else logger).warning(
            unresolved_msg, reason="unresolved_rank_deficiency"
        )
        warnings.warn(unresolved_msg, OptimizeWarning, skip_file_prefixes=_NLSQ_DIR)

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


def _rank_deficient_message(rank: int, n_params: int) -> str:
    return (
        f"Jacobian is rank-deficient (rank {rank} < {n_params} parameters); "
        "some parameters are not identifiable and their reported "
        "uncertainty is not meaningful"
    )


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
            if not keep.all():
                # same warning as pcov_from_jacobian: the pseudo-inverse gives zero
                # variance along dropped directions, which would read as "known exactly"
                warnings.warn(
                    _rank_deficient_message(int(keep.sum()), p),
                    OptimizeWarning,
                    skip_file_prefixes=_NLSQ_DIR,
                )
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
                    # else: a negative or non-finite cost stays all-inf (fired below)
    if not np.isfinite(pcov).all():
        pcov = np.full((p, p), np.inf)
    fired = bool(np.isinf(pcov).any())
    if fired:
        warn_not_estimated()
    return pcov, fired
