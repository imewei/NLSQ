"""Exact chunk-accumulated Levenberg-Marquardt for datasets of any length.

One pass over the data per trial point accumulates ``F = 0.5*sum r^2``, ``g = J^T r`` and
``H = J^T J`` in float64, so the solver state is O(p^2) and the result equals a full-batch
fit regardless of how the data is chunked. The accepted trial's statistics are reused as the
next iterate's, so no pass is spent re-evaluating an accepted point.

Ported from ``benchmarks/prototypes/exact_accumulated_lm.py`` (Stage 0, kept frozen as the
independent baseline). Stage 1 scope: unweighted and unbounded; ``sigma`` and finite
``bounds`` raise ``ValueError`` until Stage 2.

shortcut: normal equations square cond(J); the Stage 3 TSQR study decides whether to upgrade.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nlsq.config import JAXConfig
from nlsq.result import OptimizeResult
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

_STAGE2 = "is not supported yet by exact_streaming (added in Stage 2)"

_MESSAGES = {
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
        super().__init__(message)
        self.result = result


class PassStats(NamedTuple):
    """Summed statistics of one full data pass at one parameter vector."""

    cost: float  # 0.5 * sum r^2
    g: np.ndarray  # J^T r
    H: np.ndarray  # J^T J
    n_seen: int  # real (unpadded) points that contributed
    data_finite: bool  # all x, y finite (True when the check is off)


def derive_chunk_size(
    n: int, p: int, memory_limit_gb: float | None, chunk_size: int | None
) -> int:
    """Chunk length for ``n`` points and ``p`` parameters.

    An explicit positive ``chunk_size`` wins, clamped to ``n``. Otherwise ``1 << 22``,
    reduced so ``chunk * p * 8 B * 4`` fits ``memory_limit_gb`` (GiB) with a soft floor of
    1024 points (``n`` if ``n < 1024``). Raises ``ValueError`` if even the floor chunk does
    not fit the budget.
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


def make_chunk_stats(
    model: Callable[..., Any],
) -> Callable[[Any, Any, Any, Any], tuple[jax.Array, jax.Array, jax.Array]]:
    """Jitted ``(theta, x, y, mask) -> (F, g, H)`` for one chunk; ``r = y - model``.

    Rows where ``mask`` is False (the padded tail) are removed with ``jnp.where``, never by
    multiplication, so a non-finite model value on a padded row cannot leak in.
    """

    def stats(
        theta: jax.Array, x: jax.Array, y: jax.Array, mask: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        def f(q: jax.Array) -> jax.Array:
            return model(x, *q)

        r = jnp.where(mask, y - f(theta), 0.0)
        J = jnp.where(mask[:, None], jax.jacfwd(f)(theta), 0.0)
        return 0.5 * jnp.sum(r * r), J.T @ r, J.T @ J

    return jax.jit(stats)


def _block(a: Any, s: int, chunk: int) -> Any:
    """Float64 slice ``a[s:s+chunk]``: a host view if already float64, on device for JAX."""
    if isinstance(a, jax.Array):
        return jnp.asarray(a[s : s + chunk], dtype=jnp.float64)
    return np.asarray(a[s : s + chunk], dtype=np.float64)


def iter_blocks(
    source_x: Any, source_y: Any, source_sigma: Any, chunk: int
) -> Iterator[tuple[Any, Any, Any, jax.Array]]:
    """Yield equal-shape float64 ``(x, y, sigma, mask)`` chunks of ndarray or JAX inputs.

    Host float64 arrays are sliced as views and JAX arrays stay on their device. The tail
    is edge-padded (a padded row has the finiteness of a real row) and masked out.
    """
    if source_sigma is not None:
        raise ValueError(f"sigma {_STAGE2}")
    n = len(source_x)
    full = jnp.ones(chunk, dtype=bool)
    for s in range(0, n, chunk):
        xc, yc = _block(source_x, s, chunk), _block(source_y, s, chunk)
        m = xc.shape[0]
        if m == chunk:
            yield xc, yc, None, full
            continue
        pad = (0, chunk - m)
        yield (
            (jnp if isinstance(xc, jax.Array) else np).pad(xc, pad, mode="edge"),
            (jnp if isinstance(yc, jax.Array) else np).pad(yc, pad, mode="edge"),
            None,
            jnp.arange(chunk) < m,
        )


@jax.jit
def _block_finite(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(x) & jnp.isfinite(y))


def run_pass(
    stats: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    theta: Any,
    source: tuple[Any, Any, Any],
    chunk: int,
    *,
    check_finite: bool = False,
) -> PassStats:
    """One full pass over ``source = (x, y, sigma)``: summed statistics on the host.

    The finite check and the point count are reduced on device alongside the statistics,
    so the only host sync is the final ``device_get``.
    """
    th = jnp.asarray(theta, dtype=jnp.float64)
    acc: tuple[jax.Array, ...] | None = None
    finite = jnp.asarray(True)
    count = jnp.asarray(0)
    for xc, yc, _sc, mask in iter_blocks(*source, chunk):
        out = stats(th, xc, yc, mask)
        acc = (
            out if acc is None else tuple(a + b for a, b in zip(acc, out, strict=True))
        )
        count = count + jnp.sum(mask)
        if check_finite:
            finite = finite & _block_finite(xc, yc)
    if acc is None:
        raise ValueError("x and y must be non-empty")
    c, g, H, ok, n_seen = jax.device_get((*acc, finite, count))
    return PassStats(float(c), np.asarray(g), np.asarray(H), int(n_seen), bool(ok))


def step_termination(
    dF: float,
    F: float,
    rho: float,
    s_norm: float,
    x_norm: float,
    ftol: float,
    xtol: float,
) -> str | None:
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


def _stationary(
    cost: float,
    g: np.ndarray,
    H: np.ndarray,
    free: np.ndarray,
    gtol: float,
    ftol: float,
) -> str | None:
    if cost == 0:  # exact fit: decided before any division
        return "gtol"
    if not free.any() or np.max(np.abs(g[free])) < gtol:
        return "gtol"
    if newton_certificate(g, H, cost, ftol, bounds_active=not free.all()):
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


def _place(x: Any, y: Any) -> tuple[Any, Any]:
    """Move x, y to the accelerator once if they fit in DEVICE_FRACTION of free memory.

    On the CPU backend nothing is copied: chunks are host views (spec, Stage 1).
    """
    if jax.default_backend() == "cpu":
        return x, y
    try:
        mem = jax.devices()[0].memory_stats()
    except Exception as exc:  # backend without memory stats: stream chunks instead
        logger.debug(
            "exact_lm: no device memory stats; streaming chunks", error=str(exc)
        )
        return x, y
    if not mem:
        return x, y
    free = mem.get("bytes_limit", 0) - mem.get("bytes_in_use", 0)
    if 16 * len(x) > DEVICE_FRACTION * free:  # x and y as float64
        return x, y
    return jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)


def _check_inputs(
    x: Any,
    y: Any,
    p0: Sequence[float] | np.ndarray,
    sigma: Any,
    bounds: tuple[Any, Any],
    n_points: int | None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Validate shapes and Stage 1 scope; return ``(n, p, lo, hi)``."""
    if not 0 < LAM0 <= LAMBDA_MAX:
        raise ValueError(f"LAM0 must satisfy 0 < LAM0 <= {LAMBDA_MAX:g}")
    if sigma is not None:
        raise ValueError(f"sigma {_STAGE2}")
    if not hasattr(x, "shape") or x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x and y must be 1-D arrays of equal length")
    n, p = len(x), len(p0)
    if n == 0:
        raise ValueError("x and y must be non-empty")
    if n_points is not None and n_points != n:
        raise ValueError(f"n_points={n_points} does not match len(x)={n}")
    lo = np.broadcast_to(np.asarray(bounds[0], float), (p,))
    hi = np.broadcast_to(np.asarray(bounds[1], float), (p,))
    if np.any(lo >= hi):
        raise ValueError("each lower bound must be strictly less than its upper bound")
    if np.isfinite(lo).any() or np.isfinite(hi).any():
        raise ValueError(f"finite bounds {_STAGE2}")
    return n, p, lo, hi


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
) -> OptimizeResult:
    """Exact full-data Levenberg-Marquardt fit of ``y ~ model(x, *p)``.

    ``max_nfev`` counts full-data passes including the initial one (default ``100*p``).
    Returns an :class:`OptimizeResult` with ``x``, ``cost``, ``pcov``, ``nfev`` (passes),
    ``success``, ``message``, ``termination`` (``"gtol" | "ftol" | "xtol" | "max_nfev" |
    "stalled"``; success only for the first three), ``n_points`` and
    ``equivalent_to_full_batch=True``. A failed fit is returned, not raised.

    Raises ``ValueError`` for non-finite ``x``/``y`` (when ``check_finite``), a non-finite
    cost, gradient or Hessian at ``p0``, and for ``sigma`` or finite ``bounds`` (Stage 2).
    """
    n, p, lo, hi = _check_inputs(x, y, p0, sigma, bounds, n_points)
    if max_nfev is None:
        max_nfev = 100 * p
    chunk = derive_chunk_size(n, p, memory_limit_gb, chunk_size)
    source = (*_place(x, y), None)
    stats = make_chunk_stats(model)

    def evaluate(theta: np.ndarray, nfev: int, first: bool = False) -> PassStats:
        st = run_pass(stats, theta, source, chunk, check_finite=first and check_finite)
        if show_progress:
            logger.info("exact_lm pass", nfev=nfev, cost=st.cost)
        return st

    theta = np.clip(np.asarray(p0, np.float64), lo, hi)
    st = evaluate(theta, 1, first=True)
    if check_finite and not st.data_finite:
        raise ValueError("x and y must be finite (found NaN or inf)")
    if st.n_seen != n:
        raise RuntimeError(f"pass saw {st.n_seen} of {n} points")
    if not _finite(st):
        raise ValueError("model cost, gradient or Hessian is not finite at p0")
    cost, g, H = st.cost, st.g, st.H
    nfev, lam, nu = 1, LAM0, 2.0
    D = np.zeros(p)  # Moré running maximum of diag(H)
    while True:
        # g = J^T r is the descent direction; a variable at a bound with g pointing out is fixed
        free = ~(((theta <= lo) & (g <= 0)) | ((theta >= hi) & (g >= 0)))
        termination = _stationary(cost, g, H, free, gtol, ftol)
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
        step = step_termination(
            actual, cost, rho, np.linalg.norm(s), np.linalg.norm(theta), ftol, xtol
        )
        theta, cost, g, H = new, trial.cost, trial.g, trial.H
        if step:
            termination = step
            break
    message = _MESSAGES[termination]
    if termination == "stalled" and _scaled_eig(g, H)[3]:
        message += "; rank cannot be certified from normal equations"
    s2 = 1.0 if absolute_sigma else 2.0 * cost / max(n - p, 1)
    try:
        pcov = s2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        pcov = np.full((p, p), np.inf)
    return OptimizeResult(
        x=theta,
        cost=cost,
        pcov=pcov,
        nfev=nfev,
        success=termination in ("gtol", "ftol", "xtol"),
        message=message,
        termination=termination,
        n_points=n,
        equivalent_to_full_batch=True,
    )
