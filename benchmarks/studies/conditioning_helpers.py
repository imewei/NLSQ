"""Helpers for the Stage 3 conditioning study (normal equations versus TSQR).

See the "Conditioning study (pre-registered)" section of
``docs/superpowers/specs/2026-10-31-exact-streaming-solver-design.md``. ``mpmath`` is not a
project dependency: only :func:`mp_reference` imports it, lazily.

Row schema consumed by :func:`verdict` (produced by the study script)::

    {"kappa": float,                       # kappa(J) at the reference optimum
     "normal_equations": error_norms(...), # candidate errors versus the mpmath reference
     "tsqr": error_norms(...), "scipy": error_norms(...)}  # recorded, not used by the rule
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

# import-time side effect (process-wide): float64 Jacobians, as every nlsq module does
jax.config.update("jax_enable_x64", True)

_EPS = float(np.finfo(np.float64).eps)
KAPPA_MAX = 1e4  # the rule covers accuracy-set rows with kappa(J) <= 1e4
PARAM_TOL = 1e-6
COV_TOL = 1e-3


def mp_reference(
    model_mp: Callable[..., Any],
    x: np.ndarray,
    y: np.ndarray,
    p0: np.ndarray,
    digits: int = 50,
    jac_mp: Callable[..., Sequence[Any]] | None = None,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Extended-precision Gauss-Newton reference ``(theta, cost, cov, rank)``.

    ``model_mp(x_i, *params)`` is written with mpmath functions and evaluated pointwise
    at ``digits`` decimal digits; the Jacobian is ``mpmath.diff`` per partial, or the
    optional analytic ``jac_mp(x_i, *params) -> p partials`` (much faster). Steps are
    least-squares solves (``mpmath.qr_solve``) with step halving while the cost rises,
    iterated until the step is below ``10**-(digits-20)`` relative to ``max(|theta|, 1)``
    (far below float64 eps); ``RuntimeError`` if that does not happen in 200 iterations
    or a step cannot decrease the cost. Undamped, so ``p0`` should be near the optimum
    (e.g. the float64 SciPy solution); a poor start on a near-degenerate model can fail.
    ``r = y - f``, ``cost = 0.5*sum(r**2)``. ``cov`` is the pseudo-inverse of ``J^T J``
    over the eigen-directions with ``lambda > p*eps*lambda_max`` (``eps`` the float64
    machine epsilon, the module's cutoff, evaluated in extended precision), scaled by
    ``2*cost/(n-p)`` as the module's default (``absolute_sigma=False``); ``rank`` is the
    number of retained directions. Results are rounded to float64.
    """
    import mpmath as mp  # type: ignore[import-untyped]  # not a project dependency

    n, p = len(x), len(p0)
    if n <= p:
        raise ValueError(f"need n > p for the 2*cost/(n-p) scale, got n={n}, p={p}")
    with mp.workdps(digits):
        xs = [mp.mpf(float(v)) for v in np.asarray(x, dtype=np.float64)]
        ys = [mp.mpf(float(v)) for v in np.asarray(y, dtype=np.float64)]
        th = [mp.mpf(float(v)) for v in np.asarray(p0, dtype=np.float64)]

        def resid(t: list[Any]) -> Any:
            return mp.matrix(
                [yi - model_mp(xi, *t) for xi, yi in zip(xs, ys, strict=True)]
            )

        def jac(t: list[Any]) -> Any:
            J = mp.matrix(n, p)
            for i, xi in enumerate(xs):
                if jac_mp is not None:
                    for j, v in enumerate(jac_mp(xi, *t)):
                        J[i, j] = v
                    continue
                for j in range(p):
                    J[i, j] = mp.diff(
                        lambda s, xi=xi, j=j: model_mp(xi, *t[:j], s, *t[j + 1 :]), t[j]
                    )
            return J

        def half_ss(r: Any) -> Any:
            return sum(v * v for v in r) / 2

        tol = mp.mpf(10) ** (-(digits - 20))
        slack = 1 + mp.mpf(10) ** (-(digits - 10))
        r = resid(th)
        cost = half_ss(r)
        for _ in range(200):
            step = mp.qr_solve(jac(th), r)[0]
            delta = [step[j] for j in range(p)]
            if mp.norm(step) <= tol * max(mp.norm(mp.matrix(th)), 1):
                th = [a + b for a, b in zip(th, delta, strict=True)]
                break
            for _ in range(60):
                trial = [a + b for a, b in zip(th, delta, strict=True)]
                r_t = resid(trial)
                if half_ss(r_t) <= cost * slack:  # rounding-level rises accepted
                    break
                delta = [b / 2 for b in delta]
            else:
                raise RuntimeError(
                    "mp_reference stalled: no decrease after 60 halvings"
                )
            th, r, cost = trial, r_t, half_ss(r_t)
        else:
            raise RuntimeError("mp_reference did not converge in 200 iterations")
        r = resid(th)
        cost = half_ss(r)
        J = jac(th)
        E, Q = mp.eigsy(J.T * J)
        lam_max = max(E[i] for i in range(p))
        keep = [i for i in range(p) if E[i] > p * _EPS * lam_max]
        scale = 2 * cost / (n - p)
        cov = np.array(
            [
                [
                    float(sum(Q[a, i] * Q[b, i] / E[i] for i in keep) * scale)
                    for b in range(p)
                ]
                for a in range(p)
            ]
        )
        return np.array([float(t) for t in th]), float(cost), cov, len(keep)


def tsqr_factor(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]],
    theta: np.ndarray,
    model: Callable[..., Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(R, c, rho)`` of the TSQR of the augmented ``[J | r]`` at ``theta``.

    Each ``(x, y)`` chunk gives ``J`` (``jax.jacfwd`` of the pointwise jnp ``model``) and
    ``r = y - model(x, *theta)``; the running ``(p+1)x(p+1)`` triangle ``[[R, c], [0, rho]]``
    is the R factor of the stacked previous triangle and the new block, so ``R^T R = J^T J``,
    ``R^T c = J^T r`` and ``|c|^2 + rho^2 = |r|^2``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    p = theta.size
    th = jnp.asarray(theta, dtype=jnp.float64)
    T = np.zeros((0, p + 1))
    for xc, yc in chunks:
        xb = jnp.asarray(np.asarray(xc, dtype=np.float64))

        def f(q: jax.Array, xb: jax.Array = xb) -> jax.Array:
            return model(xb, *q)

        J = np.asarray(jax.jacfwd(f)(th))
        r = np.asarray(yc, dtype=np.float64) - np.asarray(f(th))
        T = np.linalg.qr(np.vstack([T, np.column_stack([J, r])]), mode="r")
    full = np.zeros((p + 1, p + 1))
    full[: T.shape[0]] = T  # fewer than p+1 rows in total
    return full[:p, :p], full[:p, p], float(full[p, p])


def tsqr_step(R: np.ndarray, c: np.ndarray, lam: float, d: np.ndarray) -> np.ndarray:
    """Solve ``min |[R; sqrt(lam*d)] s - [c; 0]|`` (``(J^T J + lam*diag(d)) s = J^T r``).

    SVD-truncated at ``s_i > sqrt(p*eps)*s_max``, the covariance cutoff on ``J^T J``.
    """
    p = len(c)
    A = np.vstack([R, np.diag(np.sqrt(lam * np.asarray(d, dtype=np.float64)))])
    b = np.concatenate([c, np.zeros(p)])
    return np.linalg.lstsq(A, b, rcond=np.sqrt(p * _EPS))[0]


def tsqr_solve(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]],
    theta: np.ndarray,
    model: Callable[..., Any],
    lam: float,
    d: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Damped Gauss-Newton step and cost ``F = 0.5*(|c|^2 + rho^2)`` by TSQR.

    :func:`tsqr_factor` then :func:`tsqr_step`; the damping is the module's
    ``H + lam*diag(D)``, with ``d`` the caller's running-maximum vector used as given.
    """
    R, c, rho = tsqr_factor(chunks, theta, model)
    return tsqr_step(R, c, lam, d), float(0.5 * (c @ c + rho * rho))


def svd_covariance(A: np.ndarray, cost: float, n: int) -> tuple[np.ndarray, int]:
    """Covariance and rank from a factor ``A`` with ``A^T A = J^T J`` (``J`` or ``R``).

    ``V diag(1/s^2) V^T`` over ``s^2 > p*eps*s_max^2`` (the module's eigenvalue cutoff),
    scaled by ``2*cost/(n-p)``.
    """
    p = A.shape[1]
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s**2 > p * _EPS * s[0] ** 2 if s.size and s[0] > 0 else np.zeros(0, bool)
    Vk = Vt[keep].T
    return (Vk / s[keep] ** 2) @ Vk.T * (2.0 * cost / (n - p)), int(keep.sum())


def eigh_rank(H: np.ndarray) -> int:
    """Retained-direction count of ``nlsq.streaming.exact_lm.compute_covariance`` for ``H``.

    Same power-of-two scaling and ``w > p*eps*w_max`` cutoff; the module does not return it.
    """
    top = np.max(np.abs(H))
    s = np.ldexp(1.0, int(np.frexp(top)[1]) - 1) if top > 0 else 1.0
    Hs = H / s
    w = np.linalg.eigvalsh((Hs + Hs.T) / 2)
    return int(np.sum(w > len(w) * _EPS * w[-1]))


def error_norms(
    theta: np.ndarray,
    theta_ref: np.ndarray,
    cov: np.ndarray,
    cov_ref: np.ndarray,
    rank: int,
    rank_ref: int,
) -> dict[str, Any]:
    """``param_err``, ``cov_err`` and ``rank_equal`` of a candidate versus the reference.

    ``param_err = max_i |theta_i - ref_i| / max(|ref_i|, 1e-12)``; ``cov_err =
    |cov - cov_ref|_F / max(|cov_ref|_F, 1e-300)``, the absolute ``|cov|_F`` when
    ``cov_ref`` is exactly zero; ``rank_equal`` compares the supplied ranks.
    """
    t, t_ref = np.asarray(theta, np.float64), np.asarray(theta_ref, np.float64)
    param_err = np.max(np.abs(t - t_ref) / np.maximum(np.abs(t_ref), 1e-12))
    ref = np.linalg.norm(cov_ref)
    diff = np.linalg.norm(np.asarray(cov, np.float64) - np.asarray(cov_ref, np.float64))
    cov_err = diff if ref == 0 else diff / max(ref, 1e-300)
    return {
        "param_err": float(param_err),
        "cov_err": float(cov_err),
        "rank_equal": bool(rank == rank_ref),
    }


def verdict(
    rows: Sequence[Mapping[str, Any]],
) -> Literal["normal_equations_stay", "tsqr_default"]:
    """Pre-registered decision rule; ``rows`` must be accuracy-set rows only.

    Normal equations pass if, on every row with ``kappa <= 1e4``, their errors versus the
    mpmath reference satisfy ``param_err <= 1e-6``, ``cov_err <= 1e-3`` and
    ``rank_equal``; otherwise TSQR becomes the default. The TSQR and SciPy entries are
    not part of the rule. A NaN or inf error fails. ``ValueError`` if any ``kappa`` is
    NaN or if no row is in range.
    """
    if any(np.isnan(row["kappa"]) for row in rows):
        raise ValueError(
            "a row has kappa = NaN; it cannot be placed inside or outside 1e4"
        )
    covered = [row for row in rows if row["kappa"] <= KAPPA_MAX]
    if not covered:
        raise ValueError(
            "no accuracy-set row with kappa(J) <= 1e4; the rule cannot apply"
        )
    for row in covered:
        ne = row["normal_equations"]
        ok = (
            ne["param_err"] <= PARAM_TOL
            and ne["cov_err"] <= COV_TOL
            and ne["rank_equal"]
        )
        if not ok:
            return "tsqr_default"
    return "normal_equations_stay"
