"""Exact chunk-accumulated Levenberg-Marquardt for datasets of any length.

One pass over the data per trial point accumulates ``F = 0.5*sum r^2``, ``g = J^T r`` and
``H = J^T J`` in float64, so the solver state is O(p^2) and the result equals a full-batch
fit regardless of how the data is chunked. The accepted trial's statistics are reused as the
next iterate's, so no pass is spent re-evaluating an accepted point.

Ported from ``benchmarks/prototypes/exact_accumulated_lm.py`` (Stage 0, kept frozen as the
independent baseline). Stage 2a adds 1-D ``sigma`` (weights ``1/sigma`` on residuals and
Jacobian rows); finite ``bounds`` raise ``ValueError`` until Stage 2b.

shortcut: normal equations square cond(J); the Stage 3 TSQR study decides whether to upgrade.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nlsq.config import JAXConfig
from nlsq.result import OptimizeResult
from nlsq.result.optimize_warning import OptimizeWarning
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
    sigma_finite: bool = True  # all sigma finite (True when unchecked or unweighted)
    sigma_positive: bool = (
        True  # all finite sigma > 0 (True when unchecked or unweighted)
    )


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
    model: Callable[..., Any], weighted: bool = False
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    """Jitted ``(theta, x, y, mask) -> (F, g, H)`` for one chunk; ``r = y - model``.

    With ``weighted=True`` the signature is ``(theta, x, y, sigma, mask)`` and residuals
    and Jacobian rows are divided by ``sigma``. Rows where ``mask`` is False (the padded
    tail) are removed with ``jnp.where`` after weighting, never by multiplication, so a
    non-finite model value or weight on a padded row cannot leak in.
    """
    if weighted:
        # separate kernel: the unweighted one below stays free of any per-row weight

        def wstats(
            theta: jax.Array, x: jax.Array, y: jax.Array, s: jax.Array, mask: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            def f(q: jax.Array) -> jax.Array:
                return model(x, *q)

            r = jnp.where(mask, (y - f(theta)) / s, 0.0)
            J = jnp.where(mask[:, None], jax.jacfwd(f)(theta) / s[:, None], 0.0)
            return 0.5 * jnp.sum(r * r), J.T @ r, J.T @ J

        return jax.jit(wstats)

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
    is edge-padded (a padded row has the finiteness of a real row) and masked out; a
    ``sigma`` tail is padded with 1, which is valid, so the sigma checks need no mask.
    ``sigma`` is ``None`` (unweighted) or a 1-D array of the same length.
    """
    n = len(source_x)
    full = jnp.ones(chunk, dtype=bool)
    for s in range(0, n, chunk):
        xc, yc = _block(source_x, s, chunk), _block(source_y, s, chunk)
        sc = None if source_sigma is None else _block(source_sigma, s, chunk)
        m = xc.shape[0]
        if m == chunk:
            yield xc, yc, sc, full
            continue
        pad = (0, chunk - m)
        yield (
            (jnp if isinstance(xc, jax.Array) else np).pad(xc, pad, mode="edge"),
            (jnp if isinstance(yc, jax.Array) else np).pad(yc, pad, mode="edge"),
            None
            if sc is None
            else (jnp if isinstance(sc, jax.Array) else np).pad(
                sc, pad, constant_values=1.0
            ),
            jnp.arange(chunk) < m,
        )


@jax.jit
def _block_finite(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(x) & jnp.isfinite(y))


@jax.jit
def _sigma_ok(s: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(all finite, all finite values > 0)``: NaN/inf is reported as non-finite only."""
    fin = jnp.isfinite(s)
    return jnp.all(fin), jnp.all(jnp.where(fin, s > 0, True))


def run_pass(
    stats: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    theta: Any,
    source: tuple[Any, Any, Any],
    chunk: int,
    *,
    check_finite: bool = False,
    check_sigma: bool = False,
) -> PassStats:
    """One full pass over ``source = (x, y, sigma)``: summed statistics on the host.

    ``stats`` must be the weighted kernel when ``sigma`` is not None. The finite and
    sigma checks and the point count are reduced on device alongside the statistics, so
    the only host sync is the final ``device_get``.
    """
    th = jnp.asarray(theta, dtype=jnp.float64)
    acc: tuple[jax.Array, ...] | None = None
    finite = s_fin = s_pos = jnp.asarray(True)
    count = jnp.asarray(0)
    for xc, yc, sc, mask in iter_blocks(*source, chunk):
        out = stats(th, xc, yc, mask) if sc is None else stats(th, xc, yc, sc, mask)
        acc = (
            out if acc is None else tuple(a + b for a, b in zip(acc, out, strict=True))
        )
        count = count + jnp.sum(mask)
        if check_finite:
            finite = finite & _block_finite(xc, yc)
        if check_sigma and sc is not None:
            f_ok, p_ok = _sigma_ok(sc)
            s_fin, s_pos = s_fin & f_ok, s_pos & p_ok
    if acc is None:
        raise ValueError("x and y must be non-empty")
    c, g, H, ok, n_seen, sf, sp = jax.device_get((*acc, finite, count, s_fin, s_pos))
    return PassStats(
        float(c),
        np.asarray(g),
        np.asarray(H),
        int(n_seen),
        bool(ok),
        bool(sf),
        bool(sp),
    )


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


def _place(x: Any, y: Any, sigma: Any) -> tuple[Any, Any, Any]:
    """Move x, y (and sigma) to the accelerator once if they fit in DEVICE_FRACTION of free
    memory.

    On the CPU backend nothing is copied: chunks are host views (spec, Stage 1).
    """
    if jax.default_backend() == "cpu":
        return x, y, sigma
    try:
        mem = jax.devices()[0].memory_stats()
    except Exception as exc:  # backend without memory stats: stream chunks instead
        logger.debug(
            "exact_lm: no device memory stats; streaming chunks", error=str(exc)
        )
        return x, y, sigma
    if not mem:
        return x, y, sigma
    free = mem.get("bytes_limit", 0) - mem.get("bytes_in_use", 0)
    arrays = (x, y) if sigma is None else (x, y, sigma)
    if 8 * len(arrays) * len(x) > DEVICE_FRACTION * free:  # float64
        return x, y, sigma
    placed = [jnp.asarray(a, dtype=jnp.float64) for a in arrays]
    return placed[0], placed[1], (None if sigma is None else placed[2])


def check_model_contract(
    model: Callable[..., Any], theta: Sequence[float] | np.ndarray, x_chunk: Any
) -> None:
    """Guard (not a proof) that ``model`` is pointwise: output ``i`` depends only on ``x[i]``.

    Evaluates the model on ``x_chunk``, its first half and its reversed copy, and requires
    ``f(x[:k]) == f(x)[:k]`` and ``f(x[::-1]) == f(x)[::-1]`` to relative 1e-12. Values
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
        (f(x_chunk[::-1]), full[::-1]),
    ):
        if got.shape != want.shape:
            raise bad
        same = same & jnp.allclose(
            got, want, rtol=1e-12, atol=1e-12 * scale, equal_nan=True
        )
    if not bool(same):
        raise bad


def _check_inputs(
    x: Any,
    y: Any,
    p0: Sequence[float] | np.ndarray,
    sigma: Any,
    bounds: tuple[Any, Any],
    n_points: int | None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Validate shapes (from ``shape`` only, no data read) and scope; return ``(n, p, lo, hi)``."""
    if not 0 < LAM0 <= LAMBDA_MAX:
        raise ValueError(f"LAM0 must satisfy 0 < LAM0 <= {LAMBDA_MAX:g}")
    if not hasattr(x, "shape") or x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x and y must be 1-D arrays of equal length")
    n, p = len(x), len(p0)
    if sigma is not None:
        if sigma.ndim != 1:
            raise ValueError(
                f"sigma must be 1-D, got shape {sigma.shape} (correlated 2-D sigma is "
                "not supported by exact_streaming)"
            )
        if sigma.shape[0] != n:
            raise ValueError(f"sigma has length {sigma.shape[0]}, expected {n}")
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


COND_WARN = 1e4  # warn when sqrt(cond(H)), the cond(J) estimate, exceeds this


def compute_covariance(
    H: np.ndarray, cost: float, n: int, p: int, absolute_sigma: bool
) -> np.ndarray:
    """Parameter covariance from ``H = J^T J`` with the ``nlsq.curve_fit`` contract.

    Pseudo-inverse over the eigen-directions with ``lambda_i > p*eps*lambda_max`` (dropped
    directions give zero rows and columns), scaled by ``2*cost/(n-p)`` unless
    ``absolute_sigma``. All-``inf`` with an :class:`OptimizeWarning` if no direction
    survives, any value is non-finite, or ``n <= p`` without ``absolute_sigma``.
    """
    H = np.asarray(H, dtype=np.float64)
    pcov = np.full((p, p), np.inf)
    if np.isfinite(H).all():
        # eigh on H/s with s a power of two (scaling is exact, entries < 2): H + H.T would
        # overflow for |H| > 1e308 / 2
        top = np.max(np.abs(H))
        s = np.ldexp(1.0, int(np.frexp(top)[1]) - 1) if top > 0 else 1.0
        Hs = H / s
        w, V = np.linalg.eigh((Hs + Hs.T) / 2)  # ascending
        keep = w > p * np.finfo(np.float64).eps * w[-1]
        if keep.any():
            lam_min = w[0] if w[0] > 0 else 0.0
            with np.errstate(
                over="ignore", divide="ignore"
            ):  # cond may overflow to inf
                ill = lam_min == 0.0 or np.sqrt(w[-1] / lam_min) > COND_WARN
            if ill:
                warnings.warn(
                    f"normal-equation Hessian is ill-conditioned (sqrt(cond(H)) > "
                    f"{COND_WARN:.0e}); the covariance may be inaccurate",
                    UserWarning,
                    stacklevel=3,
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
    if np.isinf(pcov).any():
        warnings.warn(
            "Covariance of the parameters could not be estimated",
            OptimizeWarning,
            stacklevel=3,
        )
    return pcov


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

    ``sigma`` (optional, 1-D, length ``n``) gives per-point standard deviations: residuals
    and Jacobian rows are divided by it, so ``cost = 0.5*sum(((y - f)/sigma)^2)`` and
    ``pcov`` is scaled by ``2*cost/(n - p)`` unless ``absolute_sigma``, as in
    ``nlsq.curve_fit``. Its length and dimension are checked from its shape; finiteness and
    positivity are reduced on device during the first pass (always, regardless of
    ``check_finite``, because a negative ``sigma`` would otherwise square away silently).

    Raises ``ValueError`` for non-finite ``x``/``y`` (when ``check_finite``), a ``sigma``
    that is not 1-D, of the wrong length, non-finite or non-positive, a non-finite cost,
    gradient or Hessian at ``p0``, and for finite ``bounds`` (Stage 2b).
    """
    # lists and tuples have no .shape; array-likes (ndarray, JAX, memmap, ...) are never copied
    x = x if hasattr(x, "shape") else np.asarray(x)
    y = y if hasattr(y, "shape") else np.asarray(y)
    if sigma is not None and not hasattr(sigma, "shape"):
        sigma = np.asarray(sigma)
    n, p, lo, hi = _check_inputs(x, y, p0, sigma, bounds, n_points)
    if max_nfev is None:
        max_nfev = 100 * p
    chunk = derive_chunk_size(n, p, memory_limit_gb, chunk_size)
    source = _place(x, y, sigma)
    stats = make_chunk_stats(model, weighted=sigma is not None)

    def evaluate(theta: np.ndarray, nfev: int, first: bool = False) -> PassStats:
        st = run_pass(
            stats,
            theta,
            source,
            chunk,
            check_finite=first and check_finite,
            check_sigma=first,
        )
        if show_progress:
            logger.info("exact_lm pass", nfev=nfev, cost=st.cost)
        return st

    theta = np.clip(np.asarray(p0, np.float64), lo, hi)
    check_model_contract(model, theta, _block(x, 0, chunk))
    st = evaluate(theta, 1, first=True)
    if check_finite and not st.data_finite:
        raise ValueError("x and y must be finite (found NaN or inf)")
    if not st.sigma_finite:
        raise ValueError("sigma must be finite (found NaN or inf)")
    if not st.sigma_positive:
        raise ValueError("sigma must be positive (found a zero or negative value)")
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
    pcov = compute_covariance(H, cost, n, p, absolute_sigma)
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
