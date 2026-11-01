"""Exact chunk-accumulated Levenberg-Marquardt for datasets of any length.

One pass over the data per trial point accumulates ``F = 0.5*sum r^2``, ``g = J^T r`` and
``H = J^T J`` in float64, so the solver state is O(p^2) and the result equals a full-batch
fit regardless of how the data is chunked. The accepted trial's statistics are reused as the
next iterate's, so no pass is spent re-evaluating an accepted point.

Ported from ``benchmarks/prototypes/exact_accumulated_lm.py`` (Stage 0, kept frozen as the
independent baseline). Stage 2a adds 1-D ``sigma`` (weights ``1/sigma`` on residuals and
Jacobian rows); Stage 2b adds box ``bounds`` by an active set (not Coleman-Li TRF): the
result is a KKT point of the box-constrained problem, reached by a different path than TRF.

shortcut: normal equations square cond(J); the Stage 3 TSQR study decides whether to upgrade.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

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


def _source_len(a: Any) -> int:
    """Length of a 1-D source from ``shape`` (zarr v3 arrays have no ``__len__``)."""
    return int(a.shape[0]) if hasattr(a, "shape") else len(a)


def _block(a: Any, s: int, e: int) -> Any:
    """Float64 slice ``a[s:e]``: a host view if already float64, on device for JAX.

    Callers keep ``e <= len(a)``: a strict array-like may raise on a slice past the end.
    """
    if isinstance(a, jax.Array):
        return jnp.asarray(a[s:e], dtype=jnp.float64)
    return np.asarray(a[s:e], dtype=np.float64)


def iter_blocks(
    source_x: Any, source_y: Any, source_sigma: Any, chunk: int
) -> Iterator[tuple[Any, Any, Any, jax.Array]]:
    """Yield equal-shape float64 ``(x, y, sigma, mask)`` chunks of array-like inputs.

    Every source is read by slice ``a[s:min(s+chunk, n)]``, in order, never whole and never
    shuffled: host float64 arrays give views, lazy sources (memmap, HDF5, Zarr) read one
    chunk, and JAX arrays stay on their device. The tail
    is edge-padded (a padded row has the finiteness of a real row) and masked out; a
    ``sigma`` tail is padded with 1, which is valid, so the sigma checks need no mask.
    ``sigma`` is ``None`` (unweighted) or a 1-D array of the same length.
    """
    n = _source_len(source_x)
    full = jnp.ones(chunk, dtype=bool)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xc, yc = _block(source_x, s, e), _block(source_y, s, e)
        sc = None if source_sigma is None else _block(source_sigma, s, e)
        m = xc.shape[0]
        if m == chunk:
            yield xc, yc, sc, full
            continue
        yield _padded(xc, yc, sc, chunk)


def _padded(xc: Any, yc: Any, sc: Any, chunk: int) -> tuple[Any, Any, Any, jax.Array]:
    """A short block edge-padded to ``chunk`` rows (sigma with 1) plus its mask."""
    m = xc.shape[0]
    pad = (0, chunk - m)
    return (
        (jnp if isinstance(xc, jax.Array) else np).pad(xc, pad, mode="edge"),
        (jnp if isinstance(yc, jax.Array) else np).pad(yc, pad, mode="edge"),
        None
        if sc is None
        else (jnp if isinstance(sc, jax.Array) else np).pad(
            sc, pad, constant_values=1.0
        ),
        jnp.arange(chunk) < m,
    )


def _as64(a: Any) -> Any:
    """Float64 block: on device for a JAX array, else a host array (a view if float64)."""
    if isinstance(a, jax.Array):
        return jnp.asarray(a, dtype=jnp.float64)
    return np.asarray(a, dtype=np.float64)


_END = object()  # end of a factory pass; a yielded None is a bad block, not the end


class FactorySource:
    """A replayable chunk factory ``factory(chunk) -> Iterator[(x, y) | (x, y, sigma)]``
    as a :func:`run_pass` source.

    The factory is called once per full data pass. Pass 1 is opened at construction and
    its leading blocks are validated and kept until they hold at least 2 points (or the
    stream ends): ``first_x``, their ``x`` joined and cut to ``chunk``, feeds the model
    probe, which cannot detect a non-pointwise model from one point; the first block's
    arity (``sigma`` or not) fixes the kernel. Pass 1 then replays those blocks in order
    before reading on, so the probe costs no extra factory call.
    Each block is validated as it arrives: a tuple of 2 or 3 parts, the same number as
    the first block, every part 1-D with equal lengths, ``0 < len <= chunk``. Short blocks
    are padded to ``chunk`` and masked, so every block has the compiled shape. At the end
    of every pass the point count is checked against ``n`` (``n_points`` or the count of
    pass 1): a factory that replays different data raises ``ValueError``.
    """

    def __init__(self, factory: Callable[[int], Any], chunk: int, n: int | None):
        self.factory, self.chunk, self.n = factory, chunk, n
        self._from_n_points = n is not None
        self.passes = 0
        self._pending: Iterator[Any] | None = iter(factory(chunk))
        first = next(self._pending, _END)
        if first is _END:
            raise ValueError("chunk factory yielded no blocks")
        self.weighted = isinstance(first, tuple) and len(first) == 3
        self._head: list[tuple[Any, ...]] = [self._check(first)]
        seen = self._head[0][0].shape[0]
        while seen < 2:  # a 1-point probe cannot see a non-pointwise model
            item = next(self._pending, _END)
            if item is _END:  # < 2 points in all: the fit raises after pass 1
                break
            self._head.append(self._check(item))
            seen += self._head[-1][0].shape[0]
        xs = [b[0] for b in self._head]
        self.first_x: Any = (
            xs[0]
            if len(xs) == 1
            else jnp.concatenate([jnp.asarray(a) for a in xs])[:chunk]
        )

    def _check(self, item: Any) -> tuple[Any, ...]:
        if not isinstance(item, tuple) or len(item) not in (2, 3):
            raise ValueError(
                "chunk factory must yield (x, y) or (x, y, sigma) tuples, got "
                f"{type(item).__name__}"
                + (f" of length {len(item)}" if isinstance(item, tuple) else "")
            )
        if (len(item) == 3) != self.weighted:
            raise ValueError(
                f"chunk factory block has {len(item)} parts; every block must have the "
                f"same parts as the first ({3 if self.weighted else 2})"
            )
        parts = tuple(_as64(a) for a in item)
        for name, a in zip(("x", "y", "sigma"), parts, strict=False):
            if a.ndim != 1:
                raise ValueError(f"block {name} must be 1-D, got shape {a.shape}")
        lengths = [a.shape[0] for a in parts]
        if len(set(lengths)) != 1:
            raise ValueError(f"block parts must have equal lengths, got {lengths}")
        if lengths[0] == 0:
            raise ValueError("chunk factory yielded an empty block")
        if lengths[0] > self.chunk:
            raise ValueError(
                f"chunk factory yielded a block of {lengths[0]} points, more than "
                f"chunk_size={self.chunk}"
            )
        return parts

    def blocks(self) -> Iterator[tuple[Any, Any, Any, jax.Array]]:
        """One full pass: validated, padded ``(x, y, sigma, mask)`` blocks."""
        if self._pending is None:
            items: Iterator[Any] = iter(self.factory(self.chunk))
            head: list[tuple[Any, ...]] = []
        else:  # pass 1, opened at construction: replay the probed blocks first
            items, head, self._pending = self._pending, self._head, None
            self._head, self.first_x = [], None  # not retained for the whole fit
        self.passes += 1
        full = jnp.ones(self.chunk, dtype=bool)
        count = 0
        while True:
            if head:
                parts = head.pop(0)
            else:
                item = next(items, _END)
                if item is _END:
                    break
                parts = self._check(item)
            xc, yc = parts[0], parts[1]
            sc = parts[2] if self.weighted else None
            m = xc.shape[0]
            count += m
            yield (
                (xc, yc, sc, full)
                if m == self.chunk
                else _padded(xc, yc, sc, self.chunk)
            )
        if self.n is None:
            self.n = count
        elif count != self.n:
            origin = "n_points" if self._from_n_points else "pass 1"
            raise ValueError(
                f"chunk factory yielded {count} points on pass {self.passes}, expected "
                f"{self.n} ({origin}); the factory must replay identical data on every "
                "call"
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
    source: tuple[Any, Any, Any] | FactorySource,
    chunk: int,
    *,
    check_finite: bool = False,
    check_sigma: bool = False,
) -> PassStats:
    """One full pass over ``source = (x, y, sigma)`` or a :class:`FactorySource`.

    Returns the summed statistics on the host.

    ``stats`` must be the weighted kernel when ``sigma`` is not None. The finite and
    sigma checks and the point count are reduced on device alongside the statistics, so
    the only host sync is the final ``device_get``.
    """
    th = jnp.asarray(theta, dtype=jnp.float64)
    acc: tuple[jax.Array, ...] | None = None
    finite = s_fin = s_pos = jnp.asarray(True)
    count = jnp.asarray(0)
    blocks = (
        source.blocks()
        if isinstance(source, FactorySource)
        else iter_blocks(*source, chunk)
    )
    for xc, yc, sc, mask in blocks:
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
) -> str | None:
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
    need = 8 * len(arrays) * _source_len(x)  # float64
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
    if not 0 < LAM0 <= LAMBDA_MAX:
        raise ValueError(f"LAM0 must satisfy 0 < LAM0 <= {LAMBDA_MAX:g}")
    if hasattr(x, "shape") and len(x.shape) > 1:
        raise ValueError(
            f"multi-feature (2-D) xdata is not supported by exact_streaming: x must be "
            f"1-D, got shape {tuple(x.shape)}"
        )
    if not hasattr(x, "shape") or len(x.shape) != 1 or tuple(y.shape) != tuple(x.shape):
        raise ValueError("x and y must be 1-D arrays of equal length")
    n, p = _source_len(x), len(p0)
    if sigma is not None:
        if len(sigma.shape) != 1:
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
    lo, hi = check_bounds(bounds, p)
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


def _check_factory_inputs(
    y: Any,
    p0: Sequence[float] | np.ndarray,
    sigma: Any,
    bounds: tuple[Any, Any],
    n_points: int | None,
) -> tuple[int | None, int, np.ndarray, np.ndarray]:
    """Argument checks for a chunk factory source; return ``(n_points, p, lo, hi)``."""
    if not 0 < LAM0 <= LAMBDA_MAX:
        raise ValueError(f"LAM0 must satisfy 0 < LAM0 <= {LAMBDA_MAX:g}")
    if y is not None:
        raise ValueError(
            "ydata must be None when xdata is a chunk factory (callable): the factory "
            "yields (x, y) or (x, y, sigma) blocks"
        )
    if sigma is not None:
        raise ValueError(
            "sigma must be None with a chunk factory: yield (x, y, sigma) blocks instead"
        )
    if n_points is not None and (
        isinstance(n_points, bool)
        or not isinstance(n_points, int | np.integer)
        or n_points < 1
    ):
        raise ValueError(f"n_points must be a positive integer, got {n_points!r}")
    p = len(p0)
    lo, hi = check_bounds(bounds, p)
    return (None if n_points is None else int(n_points)), p, lo, hi


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
    block. ``chunk`` is :func:`derive_chunk_size` of ``n_points``, or, with ``n_points``
    None, of an unbounded length (``chunk_size`` as given, else ``1 << 22`` reduced to the
    budget, never clamped to the unknown length). ``n_points`` is then optional: the count
    of pass 1 is used when it is None, and every pass verifies the count (``ValueError``
    naming both counts on a mismatch). With an array source ``n_points`` must match its
    length. Without ``n_points`` and ``chunk_size`` every block is padded to the derived
    chunk (default 4,194,304 rows), so pass one of them for small factories. The model
    contract probe runs on the leading blocks of pass 1, joined until they hold at least 2
    points; a factory yielding fewer than 2 points in total raises ``ValueError``.

    Raises ``ValueError`` for non-finite ``x``/``y`` (when ``check_finite``), a ``sigma``
    that is not 1-D, of the wrong length, non-finite or non-positive, a non-finite cost,
    gradient or Hessian at ``p0``, and for ``bounds`` of the wrong shape or with
    ``lb >= ub`` anywhere (equal, inverted or NaN bounds).
    """
    factory = callable(x) and not hasattr(x, "shape")
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
        if show_progress:
            logger.info("exact_lm pass", nfev=nfev, cost=st.cost)
        return st

    theta = np.clip(np.asarray(p0, np.float64), lo, hi)
    if factory:
        source = FactorySource(x, chunk, n)  # opens pass 1, reads its leading blocks
        check_model_contract(model, theta, source.first_x)
        weighted = source.weighted
    else:
        source = _place(x, y, sigma, chunk * p * 8 * WORKSPACE_FACTOR, memory_limit_gb)
        check_model_contract(model, theta, _block(x, 0, chunk))
        weighted = sigma is not None
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
