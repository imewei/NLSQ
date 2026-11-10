"""Exact chunk-accumulated Levenberg-Marquardt for datasets of any length.

One pass over the data per trial point accumulates ``F = 0.5*sum r^2``, ``g = J^T r`` and
``H = J^T J`` in float64, so the solver state is O(p^2) and the result equals a full-batch
fit (up to rounding and the stopping tolerances) regardless of how the data is chunked. The accepted trial's statistics are reused as the
next iterate's, so no pass is spent re-evaluating an accepted point.

Ported from ``benchmarks/prototypes/exact_accumulated_lm.py`` (Stage 0, kept frozen as the
independent baseline). Stage 2a adds 1-D ``sigma`` (weights ``1/sigma`` on residuals and
Jacobian rows); Stage 2b adds box ``bounds`` by an active set (not Coleman-Li TRF): the
result is a KKT point of the box-constrained problem to within the stopping tolerances,
reached by a different path than TRF.

shortcut: covariance: TSQR when kappa > 1e3; step solve stays on normal equations
(conditioning study, benchmarks/baselines/2026-10-31-exact-streaming-conditioning.md).
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

from nlsq.config import JAXConfig
from nlsq.core._option_checks import (
    coerce_absolute_sigma,
    is_integral_count,
    require_usable_tolerances,
    validate_max_nfev,
    validate_tolerance,
)
from nlsq.core.covariance import (
    _NLSQ_DIR,
    COND_WARN,
    normal_condition,
    pcov_from_jacobian,
    pcov_from_normal,
)
from nlsq.result import OptimizeResult

# Moved to normal_equations; re-exported here (tests and benchmarks import and patch them).
from nlsq.streaming.normal_equations import (
    FactorySource,
    PassStats,
    _block,
    _source_len,
    is_chunk_factory,
    iter_blocks,
    make_chunk_jacobian,
    make_chunk_stats,
    run_pass,
)
from nlsq.streaming.validators import validate_positive
from nlsq.utils.logging import get_logger

_jax_config = JAXConfig()  # enables jax_enable_x64, as every nlsq module does
logger = get_logger(__name__)

CHUNK = 1 << 22  # default chunk: 4M points
CHUNK_FLOOR = 1024  # soft floor when a memory budget shrinks the chunk
WORKSPACE_FACTOR = 4  # chunk workspace estimate: chunk * p * 8 B * WORKSPACE_FACTOR
DEVICE_FRACTION = (
    0.5  # place data on the accelerator if it fits in this share of free memory
)
LAMBDA_MAX = 1e14  # damping ceiling
ATTEMPT_CAP = 20  # solve attempts per iteration
DAMPING_FLOOR = 1e-10  # D_ii >= DAMPING_FLOOR * max_j D_jj
LAM0 = 1e-3  # initial damping; read at call time (tests move it near a limit)
TSQR_KAPPA = 1e3  # covariance by TSQR above this sqrt(cond(H)); set by Gate G-cov

Termination = Literal["gtol", "ftol", "xtol", "max_nfev", "stalled"]
_SUCCESS: frozenset[Termination] = frozenset({"gtol", "ftol", "xtol"})

_MESSAGES: dict[Termination, str] = {
    "gtol": "gradient below gtol (or exact fit)",
    "ftol": "cost decrease below ftol",
    "xtol": "step below xtol",
    "max_nfev": "maximum number of data passes reached",
    "stalled": (
        f"no further progress: trials rejected until damping reached {LAMBDA_MAX:g}"
        f" or the {ATTEMPT_CAP}-attempt cap"
    ),
}


class ExactSolverError(RuntimeError):
    """Raised by ``curve_fit_large(method="exact_streaming")`` on a failed fit.

    The full :class:`OptimizeResult` (termination ``"stalled"`` or ``"max_nfev"``) is
    available as ``.result``.
    """

    def __init__(self, message: str, result: OptimizeResult) -> None:
        super().__init__(message, result)  # args = both, so pickle can rebuild it
        self.result = result

    def __str__(self) -> str:
        return str(self.args[0])


def derive_chunk_size(
    n: int, p: int, memory_limit_gb: float | None, chunk_size: int | None
) -> int:
    """Chunk length for ``n`` points and ``p`` parameters.

    An explicit positive ``chunk_size`` wins, clamped to ``n``. Otherwise ``1 << 22``,
    reduced so ``chunk * p * 8 B * 4`` fits ``memory_limit_gb`` (GiB) with a soft floor of
    1024 points (``n`` if ``n < 1024``). Raises ``ValueError`` if even the floor chunk does
    not fit the budget. The budget goes to chunk workspace first; :func:`_place` puts
    in-memory data on a device under a limit only if data plus workspace also fits.
    """
    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        return min(int(chunk_size), n)
    floor = min(CHUNK_FLOOR, n)
    chunk = CHUNK
    if memory_limit_gb is not None:
        per_point = p * 8 * WORKSPACE_FACTOR
        budget = memory_limit_gb * 1024**3
        if floor * per_point > budget:
            raise ValueError(
                f"memory_limit_gb={memory_limit_gb} is too small for a {floor}-point chunk "
                f"with {p} parameters (needs {floor * per_point / 1024**3:.3g} GiB)"
            )
        chunk = min(chunk, int(budget // per_point))  # >= floor, checked above
    return min(chunk, n)


def step_termination(
    dF: float,
    F: float,
    rho: float,
    s_norm: float,
    x_norm: float,
    ftol: float,
    xtol: float,
) -> Termination | None:
    """Step-based criteria after an accepted step; ``F`` is the cost before the step."""
    if dF < ftol * F and rho > 0.25:
        return "ftol"
    if s_norm < xtol * (xtol + x_norm):
        return "xtol"
    return None


def _scaled_eig(
    g: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Jacobi-scaled ``(g_hat, eigvals, eigvecs, truncated)`` over columns with H_ii > 0."""
    d = np.diag(H)
    keep = d > 0
    s = 1.0 / np.sqrt(d[keep])
    Hs = H[np.ix_(keep, keep)] * s[:, None] * s[None, :]
    gs = g[keep] * s
    if gs.size == 0:
        return gs, gs, np.zeros((0, 0)), False
    w, U = np.linalg.eigh(Hs)
    truncated = not w[0] > len(g) * np.finfo(float).eps * w[-1]
    return gs, w, U, truncated


def newton_certificate(
    g: np.ndarray, H: np.ndarray, F: float, ftol: float, bounds_active: bool = False
) -> bool:
    """Damping-independent stationarity: the undamped GN step cannot gain ``ftol*F``."""
    if F == 0:
        return True
    if bounds_active or not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
        return False
    gs, w, U, truncated = _scaled_eig(g, H)
    if truncated:
        return False
    delta_n = 0.5 * float(np.sum((U.T @ gs) ** 2 / w)) if gs.size else 0.0
    return delta_n <= ftol * F


def box_certificate(
    g: np.ndarray,
    H: np.ndarray,
    F: float,
    ftol: float,
    theta: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> bool:
    """Constrained stationarity: no feasible step can gain ``ftol*F`` on the GN model.

    The bounded counterpart of :func:`newton_certificate` (spec, Termination (iii)): the
    maximum of ``m(s) = s^T g - s^T H s / 2`` over ``lo <= theta + s <= hi``, in the same
    Jacobi-scaled coordinates (``H^ = R^T R`` from its eigendecomposition; the maximiser
    is the bounded least-squares solution of ``R u = R^-T g^``; no data pass), with the
    same truncation guard. Zero columns have ``g_i = 0`` and do not move.

    The solve runs on ``g^`` divided by a power of two near ``max|g^|`` (exact), so the
    solver tolerance is relative and the decision does not depend on the scale of ``g``
    and ``F``. The solver's answer ``u`` is not trusted: with ``r = g^ - H^ u`` and ``r~``
    equal to ``r`` except where ``u`` sits on a bound with ``r`` pointing out of the box
    (set to 0, as such a component cannot gain), concavity gives for every feasible ``v``
    ``m(v) - m(u) <= r~.(v - u) - (v - u)^T H^ (v - u) / 2 <= r~^T H^-1 r~ / 2``, and also
    ``<= sum_i max over the box of r~_i (v_i - u_i)``. Certified only if ``m(u)`` plus the
    smaller bound is at most ``ftol*F``; solver failure gives no certificate.
    """
    if F == 0:
        return True
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
        return False
    gs, w, U, truncated = _scaled_eig(g, H)
    if truncated:
        return False
    if gs.size == 0:
        return True
    top = np.max(np.abs(gs))
    if top == 0:
        return True
    scale = np.ldexp(1.0, int(np.frexp(top)[1]))  # power of two: gn = gs/scale is exact
    gn = gs / scale
    keep = np.diag(H) > 0
    root = np.sqrt(np.diag(H)[keep])
    R = np.sqrt(w)[:, None] * U.T
    c = (U.T @ gn) / np.sqrt(w)
    a, b = (lo - theta)[keep] * root / scale, (hi - theta)[keep] * root / scale
    try:
        res = lsq_linear(R, c, bounds=(a, b), method="bvls")
    except Exception as exc:  # any solver failure: no certificate, iterate on
        logger.debug("exact_lm: box certificate solve failed", error=str(exc))
        return False
    # status <= 0 (iteration cap): no certificate even before the dual bound
    if res.status <= 0 or not np.all(np.isfinite(res.x)):
        return False
    u = np.clip(res.x, a, b)
    Ru = R @ u
    gain = float(u @ gn - 0.5 * (Ru @ Ru))  # direct: no O(F) cancellation
    r = gn - R.T @ Ru
    r[((u <= a) & (r <= 0)) | ((u >= b) & (r >= 0))] = 0.0  # KKT-satisfied: no gain
    quad = 0.5 * float(np.sum((U.T @ r) ** 2 / w))
    with np.errstate(invalid="ignore"):  # 0 * inf for components that cannot move
        far = np.where(r > 0, r * (b - u), np.where(r < 0, r * (a - u), 0.0))
    lin = float(np.sum(far))
    bound = min(quad, lin)
    if not np.isfinite(bound):
        return False
    return bool(gain + bound <= (ftol * F / scale) / scale)


def free_set(
    theta: np.ndarray, g: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Variables the next step may move: a variable at a bound is held when ``g`` points out.

    ``g = J^T r`` is the descent direction of ``F``, so ``g <= 0`` pushes below ``lo`` and
    ``g >= 0`` above ``hi``.
    """
    return ~(((theta <= lo) & (g <= 0)) | ((theta >= hi) & (g >= 0)))


def _stationary(
    cost: float,
    g: np.ndarray,
    H: np.ndarray,
    free: np.ndarray,
    theta: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    gtol: float,
    ftol: float,
) -> Termination | None:
    if cost == 0:  # exact fit: decided before any division
        return "gtol"
    if not free.any() or np.max(np.abs(g[free])) < gtol:
        return "gtol"
    # spec Termination (iii): the unconstrained decrement certifies only when every
    # variable is free; it is an upper bound on any feasible gain, so it stays valid
    # when a free variable sits on a bound with an inward gradient
    if newton_certificate(g, H, cost, ftol, bounds_active=not free.all()):
        return "ftol"
    # any variable on a bound (held or free): the box can cut the Newton step, so the
    # exact constrained decrease decides; never reached on the unbounded path
    if np.any((theta <= lo) | (theta >= hi)) and box_certificate(
        g, H, cost, ftol, theta, lo, hi
    ):
        return "ftol"
    return None


def _damping(D_prev: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Moré running maximum of diag(H), floored; identity if every diagonal is zero."""
    D = np.maximum(D_prev, np.diag(H))
    top = D.max()
    return np.maximum(D, DAMPING_FLOOR * top) if top > 0 else np.ones_like(D)


def _finite(st: PassStats) -> bool:
    return bool(
        np.isfinite(st.cost) and np.isfinite(st.g).all() and np.isfinite(st.H).all()
    )


def _solve(
    Hf: np.ndarray, lam: float, Df: np.ndarray, gf: np.ndarray
) -> np.ndarray | None:
    """Damped step, or None if the solve fails or is non-finite."""
    try:
        d = np.linalg.solve(Hf + lam * np.diag(Df), gf)
    except np.linalg.LinAlgError:
        return None
    return d if np.isfinite(d).all() else None


def _free_device_bytes() -> int | None:
    """Free accelerator memory in bytes; None on the CPU backend or without memory stats."""
    if jax.default_backend() == "cpu":
        return None
    try:
        mem = jax.devices()[0].memory_stats()
    except Exception as exc:  # backend without memory stats: stream chunks instead
        logger.debug(
            "exact_lm: no device memory stats; streaming chunks", error=str(exc)
        )
        return None
    if not mem:
        return None
    return int(mem.get("bytes_limit", 0) - mem.get("bytes_in_use", 0))


def _place(
    x: Any,
    y: Any,
    sigma: Any,
    workspace_bytes: int = 0,
    memory_limit_gb: float | None = None,
) -> tuple[Any, Any, Any]:
    """Move in-memory x, y (and sigma) to the accelerator once if they fit.

    They fit when the float64 data takes at most DEVICE_FRACTION of free device memory
    and, with ``memory_limit_gb``, data plus ``workspace_bytes`` (the chunk workspace)
    stays within that budget. Only a plain ``np.ndarray`` or a ``jax.Array`` is placed: a
    lazy source (memmap, HDF5, Zarr, any other array-like) is always streamed chunk by
    chunk from the source. On the CPU backend nothing is copied: chunks are host views.
    """
    arrays = (x, y) if sigma is None else (x, y, sigma)
    if not all(type(a) is np.ndarray or isinstance(a, jax.Array) for a in arrays):
        return x, y, sigma
    free = _free_device_bytes()
    if free is None:
        return x, y, sigma
    # float64; a scalar sigma is broadcast per chunk and costs nothing per point
    need = 8 * sum(1 for a in arrays if len(a.shape) != 0) * _source_len(x)
    if need > DEVICE_FRACTION * free:
        return x, y, sigma
    if (
        memory_limit_gb is not None
        and need + workspace_bytes > memory_limit_gb * 1024**3
    ):
        return x, y, sigma
    placed = [jnp.asarray(a, dtype=jnp.float64) for a in arrays]
    return placed[0], placed[1], (None if sigma is None else placed[2])


def check_model_contract(
    model: Callable[..., Any], theta: Sequence[float] | np.ndarray, x_chunk: Any
) -> None:
    """Guard (not a proof) that ``model`` is pointwise: output ``i`` depends only on ``x[i]``.

    Evaluates the model on ``x_chunk``, its first half, its second half and its reversed
    copy, and requires ``f(x[:k]) == f(x)[:k]``, ``f(x[k:]) == f(x)[k:]`` and
    ``f(x[::-1]) == f(x)[::-1]`` to relative 1e-12 (the second half catches statistics
    anchored at the batch min or max on sorted ``x``, which the first half keeps). Values
    within 1e-12 of the largest ``|f|`` also pass, so a zero crossing cannot trigger a
    false alarm. A model that behaves differently only on later chunks is not detected.
    """
    m = len(x_chunk)
    # the real fit passes JAX float64 arrays/tracers: probe with the same types
    th = [jnp.asarray(t, dtype=jnp.float64) for t in theta]
    x_chunk = jnp.asarray(x_chunk, dtype=jnp.float64)

    # compared on the model's own device: a host round trip of the 4M-point chunk and
    # three np.allclose passes cost ~0.2 s per fit on GPU
    def f(xs: Any) -> jax.Array:
        return jnp.asarray(model(xs, *th), dtype=jnp.float64)

    full = f(x_chunk)
    bad = ValueError(
        "model is not pointwise: f(x, *theta)[i] must depend only on x[i] and theta, but "
        "evaluating on a sub-batch or a reversed batch changed the output (batch "
        "statistics, cumulative sums and randomness break exact_streaming's exactness)"
    )
    if full.shape != (m,):
        raise bad
    if m < 2:
        return
    scale = jnp.max(jnp.where(jnp.isfinite(full), jnp.abs(full), 0.0))
    same = jnp.asarray(True)
    for got, want in (
        (f(x_chunk[: m // 2]), full[: m // 2]),
        (f(x_chunk[m // 2 :]), full[m // 2 :]),
        (f(x_chunk[::-1]), full[::-1]),
    ):
        if got.shape != want.shape:
            raise bad
        same = same & jnp.allclose(
            got, want, rtol=1e-12, atol=1e-12 * scale, equal_nan=True
        )
    if not bool(jax.device_get(same)):  # explicit: allowed by a device-to-host guard
        raise bad


def check_bounds(bounds: tuple[Any, Any], p: int) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` as float64 arrays of shape ``(p,)``; each bound is a scalar or ``(p,)``.

    Raises ``ValueError`` on any other shape and unless ``lo < hi`` elementwise, so
    ``lo == hi``, inverted infinite bounds such as ``(inf, -inf)`` and NaN are rejected.
    """
    lo, hi = (np.asarray(b, dtype=np.float64) for b in bounds)
    for name, b in (("lower", lo), ("upper", hi)):
        if b.ndim != 0 and b.shape != (p,):
            raise ValueError(
                f"{name} bound has shape {b.shape}, expected a scalar or ({p},)"
            )
    lo, hi = np.broadcast_to(lo, (p,)), np.broadcast_to(hi, (p,))
    if not np.all(lo < hi):
        raise ValueError(
            "each lower bound must be strictly less than its upper bound (lb < ub "
            "elementwise; equal, inverted or NaN bounds are rejected)"
        )
    return lo, hi


def _check_inputs(
    x: Any,
    y: Any,
    p0: Sequence[float] | np.ndarray,
    sigma: Any,
    bounds: tuple[Any, Any],
    n_points: int | None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Validate shapes (from ``shape`` only, no data read) and scope; return ``(n, p, lo, hi)``."""
    if hasattr(x, "shape") and len(x.shape) > 1:
        raise ValueError(
            f"multi-feature (2-D) xdata is not supported by exact_streaming: x must be "
            f"1-D, got shape {tuple(x.shape)}"
        )
    if not hasattr(x, "shape") or len(x.shape) != 1 or tuple(y.shape) != tuple(x.shape):
        raise ValueError("x and y must be 1-D arrays of equal length")
    n, p = _source_len(x), len(p0)
    if sigma is not None and len(sigma.shape) != 0:  # a scalar is broadcast per chunk
        if len(sigma.shape) != 1:
            raise ValueError(
                f"sigma must be 1-D (or a scalar), got shape {tuple(sigma.shape)} "
                "(correlated 2-D sigma is not supported by exact_streaming)"
            )
        if sigma.shape[0] != n:
            raise ValueError(f"sigma has length {sigma.shape[0]}, expected {n}")
    if n < 2:
        raise ValueError(f"Need at least 2 data points for fitting, got {n}.")
    if n_points is not None and n_points != n:
        raise ValueError(f"n_points={n_points} does not match len(x)={n}")
    lo, hi = check_bounds(bounds, p)
    return n, p, lo, hi


def compute_covariance(
    H: np.ndarray, cost: float, n: int, p: int, absolute_sigma: bool
) -> np.ndarray:
    """Parameter covariance from ``H = J^T J`` with the ``nlsq.curve_fit`` contract.

    Thin wrapper over :func:`nlsq.core.covariance.pcov_from_normal` returning only
    ``pcov`` (``cost = 0.5*sum(r^2)``).
    """
    return pcov_from_normal(H, cost, n, p, absolute_sigma)[0]


def _open_source(
    x: Any,
    y: Any,
    sigma: Any,
    chunk: int,
    p: int,
    memory_limit_gb: float | None,
    n_points: int | None,
) -> tuple[tuple[Any, Any, Any] | FactorySource, bool]:
    """``(source, weighted)`` for :func:`run_pass`: a :class:`FactorySource` that opens
    pass 1 (``n_points`` the validated declared count, or None) or the placed arrays."""
    if is_chunk_factory(x):
        # opens pass 1 and reads its leading blocks
        source = FactorySource(x, chunk, n_points)
        return source, source.weighted
    return _place(
        x, y, sigma, chunk * p * 8 * WORKSPACE_FACTOR, memory_limit_gb
    ), sigma is not None


# one slot: (model, weighted, kernel) of the last TSQR pass; read and replaced whole
_jac_kernel: list[tuple[Any, bool, Callable[..., jax.Array]] | None] = [None]


def _chunk_jacobian(
    model: Callable[..., Any], weighted: bool
) -> Callable[..., jax.Array]:
    """:func:`make_chunk_jacobian` kernel, cached for the last ``(model, weighted)``.

    Rebuilt only when ``model`` is a different object or ``weighted`` changes, so repeated
    fits of one model do not re-trace (the jitted kernel bakes ``model`` in). Thread-safe:
    the slot is read once and the kernel returned is always the local one, so another
    thread replacing the slot cannot hand this call a different model's kernel.
    """
    entry = _jac_kernel[0]
    if entry is not None and entry[0] is model and entry[1] == weighted:
        return entry[2]
    kernel = make_chunk_jacobian(model, weighted=weighted)
    _jac_kernel[0] = (model, weighted, kernel)
    return kernel


@jax.jit
def _qr_append(R: jax.Array, J: jax.Array) -> jax.Array:
    return jnp.linalg.qr(jnp.vstack([R, J]), mode="r")


def tsqr_r_factor(
    model: Callable[..., Any],
    theta: Any,
    source: tuple[Any, Any, Any] | FactorySource,
    chunk: int,
    *,
    weighted: bool,
) -> np.ndarray:
    """The ``min(n, p) x p`` R factor of the full weighted Jacobian at ``theta``, one pass.

    TSQR: ``R <- qr(vstack(R, J_chunk), mode="r")`` per chunk, in float64 on the chunk's
    device, so the state is ``O(p^2)`` and the workspace ``O(chunk * p)``. ``J^T J = R^T R``
    without forming ``J^T J``, so ``R`` carries ``cond(J)``, not its square. Chunks and the
    padded-tail mask are those of :func:`run_pass`; a factory source verifies its count.
    """
    th = jnp.asarray(theta, dtype=jnp.float64)
    jac = _chunk_jacobian(model, weighted)
    R = jnp.zeros((0, th.shape[0]), dtype=jnp.float64)
    blocks = (
        source.blocks()
        if isinstance(source, FactorySource)
        else iter_blocks(*source, chunk)
    )
    for xc, _, sc, mask in blocks:  # exhausted: a factory checks its count at the end
        Jc = jac(th, xc, mask) if sc is None else jac(th, xc, sc, mask)
        R = _qr_append(R, Jc)
    return np.asarray(R)


def _warn_tsqr_route(H: np.ndarray, kappa: float) -> None:
    """The TSQR-route ``UserWarning`` above ``COND_WARN`` (non-finite ``H`` named as such)."""
    if not np.isfinite(H).all():
        state = "is non-finite"
    elif kappa > COND_WARN:
        state = f"is ill-conditioned (sqrt(cond(H)) > {COND_WARN:.0e})"
    else:
        return
    warnings.warn(
        f"normal-equation Hessian {state}; covariance computed by QR; the parameters "
        "may stop short of the optimum: check the result",
        UserWarning,
        skip_file_prefixes=_NLSQ_DIR,
    )


def _check_factory_inputs(
    y: Any,
    p0: Sequence[float] | np.ndarray,
    sigma: Any,
    bounds: tuple[Any, Any],
    n_points: int | None,
) -> tuple[int | None, int, np.ndarray, np.ndarray]:
    """Argument checks for a chunk factory source; return ``(n_points, p, lo, hi)``."""
    if y is not None:
        raise ValueError(
            "ydata must be None when xdata is a chunk factory (callable): the factory "
            "yields (x, y) or (x, y, sigma) blocks"
        )
    if sigma is not None:
        raise ValueError(
            "sigma must be None with a chunk factory: yield (x, y, sigma) blocks instead"
        )
    if n_points is not None and not (
        isinstance(n_points, int | np.integer) and is_integral_count(n_points, 1)
    ):
        raise ValueError(f"n_points must be a positive integer, got {n_points!r}")
    p = len(p0)
    lo, hi = check_bounds(bounds, p)
    return (None if n_points is None else int(n_points)), p, lo, hi


def _check_tolerances(
    ftol: float, xtol: float, gtol: float
) -> tuple[float, float, float]:
    """Reject NaN/inf/negative tolerances and all-below-eps, before any data pass.

    Returns the validated values (``None`` becomes 0) for the caller to use.
    """
    ftol, xtol, gtol = (
        validate_tolerance(name, tol, strict=True)
        for name, tol in (("ftol", ftol), ("xtol", xtol), ("gtol", gtol))
    )
    require_usable_tolerances(ftol, xtol, gtol)
    return ftol, xtol, gtol


def exact_lm_fit(  # noqa: C901 - ported LM loop kept intact from the prototype
    model: Callable[..., Any],
    x: Any,
    y: Any,
    p0: Sequence[float] | np.ndarray,
    *,
    sigma: Any = None,
    bounds: tuple[Any, Any] = (-np.inf, np.inf),
    chunk_size: int | None = None,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int | None = None,
    check_finite: bool = True,
    absolute_sigma: bool = False,
    memory_limit_gb: float | None = None,
    show_progress: bool = False,
    n_points: int | None = None,
    compute_covariance: bool = True,
) -> OptimizeResult:
    """Exact full-data Levenberg-Marquardt fit of ``y ~ model(x, *p)``.

    ``max_nfev`` counts full-data passes including the initial one (default ``100*p``).
    Returns an :class:`OptimizeResult` with ``x``, ``cost``, ``pcov``, ``nfev`` (passes),
    ``success``, ``message``, ``termination`` (``"gtol" | "ftol" | "xtol" | "max_nfev" |
    "stalled"``; success only for the first three), ``n_points`` and
    ``equivalent_to_full_batch=True``. A failed fit is returned, not raised.

    ``pcov`` comes from ``H`` at the final ``x`` when ``sqrt(cond(H)) <= TSQR_KAPPA``
    (``covariance_method="normal"``); above it, from the SVD of the TSQR ``R`` factor of
    the full Jacobian, one extra data pass not counted in ``nfev`` (``"tsqr"``);
    ``"skipped"`` with ``compute_covariance=False``.

    ``sigma`` (optional, a scalar or 1-D of length ``n``) gives per-point standard
    deviations: residuals and Jacobian rows are divided by ``|sigma|`` (the objective
    depends on ``sigma^2`` only, so a negative value acts as its magnitude), so
    ``cost = 0.5*sum(((y - f)/sigma)^2)`` and ``pcov`` is scaled by ``2*cost/(n - p)``
    unless ``absolute_sigma``, as in ``nlsq.curve_fit``. A scalar is broadcast per chunk,
    never to an ``n``-length array. Its length and dimension are checked from its shape;
    finiteness and nonzero-ness are reduced on device during the first pass (always,
    regardless of ``check_finite``).

    ``bounds = (lb, ub)`` (scalars or length-``p`` arrays, ``lb < ub`` elementwise) are
    handled by an active set: ``p0`` is clipped into the box; each iteration holds the
    variables of :func:`free_set` (on a bound, gradient pointing out), solves the damped
    system on the rest, projects the trial onto the box and predicts the decrease from
    the projected step. ``gtol`` applies to the free gradient; while a bound is active
    the free-subspace Newton certificate is not used and :func:`box_certificate` (the
    exact constrained model decrease) takes its place. ``pcov`` uses all ``p``
    parameters for the degrees of freedom, as the legacy path.

    ``x``, ``y`` and ``sigma`` may be NumPy or JAX arrays or out-of-memory array-likes
    (``np.memmap``, ``h5py.Dataset``, ``zarr.Array``, any object with ``shape``, ``len``
    and slicing). Array-likes are read by slice, one chunk at a time, in order: each pass
    reads every point once; the first-chunk model probe reads one extra chunk of ``x``.
    They are never converted or placed on a device whole; ``memory_limit_gb`` caps the
    resident data plus chunk workspace when an in-memory array is placed on a device.
    The chunk is sized from that budget first, so under a limit whole-array placement
    happens only when data plus workspace fit (in practice, small inputs). A
    ``jax.Array`` already on a device stays there regardless of the limit.

    ``x`` may instead be a chunk factory (``y`` and ``sigma`` None), see
    :class:`FactorySource`: ``x(chunk) -> Iterator[(x, y) | (x, y, sigma)]``, replayable
    (called once per full pass, same data in the same order), at most ``chunk`` points per
    block. Replay must yield identical data on every pass; only the point count is
    verified. ``chunk`` is :func:`derive_chunk_size` of ``n_points``, or, with ``n_points``
    None, of an unbounded length (``chunk_size`` as given, else ``1 << 22`` reduced to the
    budget, never clamped to the unknown length). ``n_points`` is then optional: the count
    of pass 1 is used when it is None, and every pass verifies the count (``ValueError``
    naming both counts on a mismatch). With an array source ``n_points`` must match its
    length. Without ``n_points`` and ``chunk_size`` every block is padded to the derived
    chunk (default 4,194,304 rows), so pass one of them for small factories. The model
    contract probe runs on the leading blocks of pass 1, joined until they hold at least 2
    points; a factory yielding fewer than 2 points in total raises ``ValueError``.

    Raises ``ValueError`` for non-finite ``x``/``y`` (when ``check_finite``), a ``sigma``
    that is not a scalar or 1-D, of the wrong length, non-finite or zero, a non-finite cost,
    gradient or Hessian at ``p0``, and for ``bounds`` of the wrong shape or with
    ``lb >= ub`` anywhere (equal, inverted or NaN bounds).
    """
    if not 0 < LAM0 <= LAMBDA_MAX:
        raise ValueError(f"LAM0 must satisfy 0 < LAM0 <= {LAMBDA_MAX:g}")
    ftol, xtol, gtol = _check_tolerances(ftol, xtol, gtol)
    absolute_sigma = coerce_absolute_sigma(absolute_sigma)
    if memory_limit_gb is not None:
        validate_positive(memory_limit_gb, "memory_limit_gb")
    factory = is_chunk_factory(x)
    source: tuple[Any, Any, Any] | FactorySource
    if factory:
        n, p, lo, hi = _check_factory_inputs(y, p0, sigma, bounds, n_points)
        # n unknown: no clamp to n (the default or budget chunk, or chunk_size as given)
        chunk = derive_chunk_size(
            sys.maxsize if n is None else n, p, memory_limit_gb, chunk_size
        )
    else:
        # lists and tuples have no .shape; array-likes (ndarray, JAX, memmap, ...) are never copied
        x = x if hasattr(x, "shape") else np.asarray(x)
        y = y if hasattr(y, "shape") else np.asarray(y)
        if sigma is not None and not hasattr(sigma, "shape"):
            sigma = np.asarray(sigma)
        n, p, lo, hi = _check_inputs(x, y, p0, sigma, bounds, n_points)
        chunk = derive_chunk_size(n, p, memory_limit_gb, chunk_size)
    max_nfev = validate_max_nfev(max_nfev)
    if max_nfev is None:
        max_nfev = 100 * p

    def evaluate(theta: np.ndarray, nfev: int, first: bool = False) -> PassStats:
        st = run_pass(
            stats,
            theta,
            source,
            chunk,
            check_finite=first and check_finite,
            check_sigma=first,
        )
        if (
            not factory and st.n_seen != n
        ):  # factory sources verify their count per pass
            raise ValueError(
                f"pass {nfev} saw {st.n_seen} points, expected {n}: the source changed "
                "length between passes"
            )
        if show_progress:
            logger.info("exact_lm pass", nfev=nfev, cost=st.cost)
        return st

    theta = np.clip(np.asarray(p0, np.float64), lo, hi)
    source, weighted = _open_source(x, y, sigma, chunk, p, memory_limit_gb, n)
    if isinstance(source, FactorySource):
        check_model_contract(model, theta, source.first_x)
    else:
        # at least 2 points even for chunk_size=1: a 1-point probe proves nothing
        check_model_contract(model, theta, _block(x, 0, min(n, max(chunk, 2))))
    stats = make_chunk_stats(model, weighted=weighted)
    st = evaluate(theta, 1, first=True)
    if isinstance(source, FactorySource):
        n = source.n  # counted (or verified against n_points) during pass 1
        if n < 2:
            raise ValueError(f"Need at least 2 data points for fitting, got {n}.")
    if check_finite and not st.data_finite:
        raise ValueError("x and y must be finite (found NaN or inf)")
    if not st.sigma_finite:
        raise ValueError("sigma must be finite (found NaN or inf)")
    if not st.sigma_nonzero:
        raise ValueError("sigma must be nonzero (found a zero value)")
    if not _finite(st):
        raise ValueError("model cost, gradient or Hessian is not finite at p0")
    cost, g, H = st.cost, st.g, st.H
    nfev, lam, nu = 1, LAM0, 2.0
    D = np.zeros(p)  # Moré running maximum of diag(H)
    while True:
        free = free_set(theta, g, lo, hi)
        termination = _stationary(cost, g, H, free, theta, lo, hi, gtol, ftol)
        if termination:
            break
        if nfev >= max_nfev:
            termination = "max_nfev"
            break
        Hf, gf = H[np.ix_(free, free)], g[free]
        D = _damping(D, H)
        accepted = out_of_budget = False
        attempts = 0
        while attempts < ATTEMPT_CAP and lam <= LAMBDA_MAX:
            if nfev >= max_nfev:
                out_of_budget = True
                break
            attempts += 1
            d = _solve(Hf, lam, D[free], gf)
            if d is None:  # failed solve: no data pass, still an attempt
                lam, nu = lam * nu, nu * 2
                continue
            new = theta.copy()
            new[free] += d
            new = np.clip(new, lo, hi)
            s = new - theta
            pred = s @ g - 0.5 * s @ H @ s  # quadratic-model decrease in 0.5*sum r^2
            trial = evaluate(new, nfev + 1)
            nfev += 1
            actual = cost - trial.cost
            if _finite(trial) and actual > 0 and pred > 0:
                rho = actual / pred
                # cap: an acceptance must not leave lam above the ceiling (next iteration would skip its trials)
                lam = min(
                    max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-14), LAMBDA_MAX
                )
                nu = 2.0
                accepted = True
                break
            lam, nu = lam * nu, nu * 2
        if not accepted:
            termination = "max_nfev" if out_of_budget else "stalled"
            break
        # MINPACK-style scaled norms (||D^1/2 s||, ||D^1/2 x||): the relative
        # step test must not depend on the units of the parameters.
        root_D = np.sqrt(np.where(D > 0, D, 1.0))
        step = step_termination(
            actual,
            cost,
            rho,
            np.linalg.norm(root_D * s),
            np.linalg.norm(root_D * theta),
            ftol,
            xtol,
        )
        theta, cost, g, H = new, trial.cost, trial.g, trial.H
        if step:
            termination = step
            break
    message = _MESSAGES[termination]
    if termination == "stalled" and _scaled_eig(g, H)[3]:
        message += "; rank cannot be certified from normal equations"
    # The parameter shadows the module function compute_covariance
    if not compute_covariance:
        pcov, covariance_method = None, "skipped"
        logger.debug("covariance skipped", reason="compute_covariance=False")
    elif (kappa := normal_condition(H)) > TSQR_KAPPA:
        # normal equations lose cond(J)^2: one more pass for R at theta (not in nfev)
        from nlsq.core.minpack import _covariance_svd  # lazy: minpack is heavy

        R = tsqr_r_factor(model, theta, source, chunk, weighted=weighted)
        pcov = pcov_from_jacobian(R, cost, n, p, absolute_sigma, _covariance_svd)[0]
        covariance_method = "tsqr"
        _warn_tsqr_route(H, kappa)
    else:
        pcov = pcov_from_normal(H, cost, n, p, absolute_sigma)[0]
        covariance_method = "normal"
    return OptimizeResult(
        x=theta,
        cost=cost,
        pcov=pcov,
        nfev=nfev,
        success=termination in _SUCCESS,
        message=message,
        termination=termination,
        n_points=n,
        equivalent_to_full_batch=True,
        covariance_method=covariance_method,
    )
