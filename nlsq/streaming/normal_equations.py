"""Chunked normal-equation accumulation: ``F = 0.5 sum r^2``, ``g = J^T r``, ``H = J^T J``.

The jitted per-chunk kernel (:func:`make_chunk_stats`) and the data pass
(:func:`run_pass`) are shared by the exact streaming LM solver
(:mod:`nlsq.streaming.exact_lm`, which calls them directly and re-exports the moved
helpers for import compatibility), the hybrid optimizer
(:mod:`nlsq.streaming.adaptive_hybrid`) and the Gauss-Newton phase
(:mod:`nlsq.streaming.phases.gauss_newton`); the latter two use :class:`NormalEquations`
through :func:`cached_kernel` and :meth:`NormalEquations.add_chunk`. A data pass sums the
per-chunk statistics over equal-shape, masked chunks read from array-likes or a
replayable chunk factory; :class:`NormalEquations` bundles the kernel and the pass.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, NamedTuple, TypeGuard

import jax
import jax.numpy as jnp
import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()  # enables jax_enable_x64, as every nlsq module does


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


# Contraction choice for J^T r and J^T J of a masked (n, p) Jacobian, by backend.
# Inside jit, XLA folds the transpose into the dot; on CPU that is slow for skinny J.
# Measured (min of 12 repeats, float64, n = 1e5 and 1e6, ms; one RTX 4090 Laptop GPU and
# this CPU), fused multiply-and-sum vs the jitted dot vs the eager dot of the pre-jit code:
#   CPU n=1e6: p=3 fused 2.6 / jit dot 13.6 / eager 5.5; p=8 3.9 / 19.2 / 24.5;
#              p=30 33.9 / 70.8 / 126; p=50 95.8 / 120 / 247.
#   CPU n=1e5: p=3 0.23 / 1.26 / 0.37; p=8 0.39 / 0.46 / 1.35; p=12 1.58 / 1.23 / 2.9;
#              p=50 10.1 / 18.8 / 25.2.
#   -> on CPU fused is faster than the eager dot at every p in 1..50 and than the jitted
#      dot except at p=1 and p=12 (n=1e5, 1.3-1.9x of a <=1.6 ms term): always fused.
#   GPU n=1e6: p=3 fused 0.19 / dot 0.48; p=5 0.69 / 0.63; p=12 4.0 / 2.8; p=30 21.6 / 5.8.
#   GPU n=1e5: p=3 0.07 / 0.16; p=5 0.20 / 0.16; p=30 1.4 / 1.1; p=50 7.0 / 2.6.
#   -> on GPU fused wins up to p=3-4 and loses from p=5: fused for p < 5.
_GPU_FUSED_BELOW_P = 5


def _fused_contractions(J: jax.Array, r: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(J^T r, J^T J)`` as multiply-and-sum over the rows (no ``(n, p, p)`` buffer)."""
    return (
        jnp.sum(J * r[:, None], axis=0),
        jnp.sum(J[:, :, None] * J[:, None, :], axis=0),
    )


def _dot_contractions(J: jax.Array, r: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(J^T r, J^T J)`` as matrix products."""
    return J.T @ r, J.T @ J


def _use_fused(p: int) -> bool:
    """Whether the fused form is used for ``p`` parameters on the current backend.

    Decided at trace time (the kernel is traced for the default backend).
    """
    return jax.default_backend() == "cpu" or p < _GPU_FUSED_BELOW_P


def _gradient_and_gauss_newton_hessian(
    J: jax.Array, r: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """``(J^T r, J^T J)`` of a masked ``(n, p)`` Jacobian and its residual.

    Either form sums the same products; they differ only in summation order (the fused
    one is the more accurate, about 1e-16 relative against 1e-14 for the dot at n=1e6).
    """
    contract = _fused_contractions if _use_fused(J.shape[1]) else _dot_contractions
    return contract(J, r)


def make_chunk_stats(
    model: Callable[..., Any],
    weighted: bool = False,
    *,
    jacobian: Callable[[Any, Any], Any] | None = None,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    """Jitted ``(theta, x, y, mask) -> (F, g, H)`` for one chunk; ``r = y - model``.

    With ``weighted=True`` the signature is ``(theta, x, y, sigma, mask)`` and residuals
    and Jacobian rows are divided by ``sigma``. Rows where ``mask`` is False (the padded
    tail) are removed with ``jnp.where`` after weighting, never by multiplication, so a
    non-finite model value or weight on a padded row cannot leak in.

    ``jacobian=None`` differentiates the chunk with ``jax.jacfwd``; a callable
    ``jacobian(theta, x_chunk) -> (n_chunk, p)`` replaces it (``d model / d theta``, for
    models ``jacfwd`` cannot differentiate), with the same masking and weighting.
    """

    def jac(f: Callable[[jax.Array], jax.Array], theta: jax.Array, x: jax.Array) -> Any:
        return jax.jacfwd(f)(theta) if jacobian is None else jacobian(theta, x)

    if weighted:
        # separate kernel: the unweighted one below stays free of any per-row weight

        def wstats(
            theta: jax.Array, x: jax.Array, y: jax.Array, s: jax.Array, mask: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            def f(q: jax.Array) -> jax.Array:
                return model(x, *q)

            r = jnp.where(mask, (y - f(theta)) / s, 0.0)
            J = jnp.where(mask[:, None], jac(f, theta, x) / s[:, None], 0.0)
            return 0.5 * jnp.sum(r * r), *_gradient_and_gauss_newton_hessian(J, r)

        return jax.jit(wstats)

    def stats(
        theta: jax.Array, x: jax.Array, y: jax.Array, mask: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        def f(q: jax.Array) -> jax.Array:
            return model(x, *q)

        r = jnp.where(mask, y - f(theta), 0.0)
        J = jnp.where(mask[:, None], jac(f, theta, x), 0.0)
        return 0.5 * jnp.sum(r * r), *_gradient_and_gauss_newton_hessian(J, r)

    return jax.jit(stats)


def _source_len(a: Any) -> int:
    """Length of a 1-D source from ``shape`` (zarr v3 arrays have no ``__len__``)."""
    return int(a.shape[0]) if hasattr(a, "shape") else len(a)


def _block(a: Any, s: int, e: int) -> Any:
    """Float64 slice ``a[s:e]``: a host view if already float64, on device for JAX.

    Callers keep ``e <= len(a)``: a strict array-like may raise on a slice past the end.
    """
    return _as64(a[s:e])


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
        got = {"x": m, "y": yc.shape[0]} | (
            {} if sc is None else {"sigma": sc.shape[0]}
        )
        if any(v != e - s for v in got.values()):  # a short slice would be edge-padded
            raise ValueError(
                f"source slice [{s}:{e}] returned {got} rows, expected {e - s} each: "
                "x, y and sigma must all have the same length on every pass"
            )
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


def is_chunk_factory(
    x: Any,
) -> TypeGuard[Callable[[int], Iterator[tuple[Any, ...]]]]:
    """True for a replayable chunk factory: a callable that is not an array-like."""
    return callable(x) and not hasattr(x, "shape")


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


class NormalEquations:
    """Chunked accumulator of ``F = 0.5 sum r^2``, ``g = J^T r`` and ``H = J^T J``.

    Holds the jitted per-chunk kernel of :func:`make_chunk_stats` for ``model(x, *theta)``
    (``r = y - model``) and the chunk length every pass uses. ``jacobian`` is an optional
    ``(theta, x_chunk) -> (n_chunk, p)`` callable replacing the default chunk-wide
    ``jax.jacfwd``; masking, weighting and accumulation are identical either way.
    """

    def __init__(
        self,
        model: Callable[..., Any],
        *,
        weighted: bool = False,
        chunk: int,
        jacobian: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        self.chunk = chunk
        self._stats = make_chunk_stats(model, weighted, jacobian=jacobian)

    def chunk_stats(
        self,
        theta: jax.Array,
        x: jax.Array,
        y: jax.Array,
        mask: jax.Array,
        sigma: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """``(F, g, H)`` of one chunk; ``sigma`` only for a ``weighted=True`` instance."""
        if sigma is None:
            return self._stats(theta, x, y, mask)
        return self._stats(theta, x, y, sigma, mask)

    def accumulate(
        self,
        theta: Any,
        source: tuple[Any, Any, Any] | FactorySource,
        *,
        check_finite: bool = False,
        check_sigma: bool = False,
    ) -> PassStats:
        """One full pass over ``source``; see :func:`run_pass`."""
        return run_pass(
            self._stats,
            theta,
            source,
            self.chunk,
            check_finite=check_finite,
            check_sigma=check_sigma,
        )

    def add_chunk(
        self,
        theta: jax.Array,
        x: jax.Array,
        y: jax.Array,
        H_prev: jax.Array,
        g_prev: jax.Array,
        valid_size: int | None = None,
        sigma: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, float]:
        """Fold one chunk into running ``(H, g)``; returns ``(H, g, sum r^2)``.

        ``valid_size`` is the number of real leading points of a padded chunk
        (``None``: all points are real). ``sigma`` is required for, and only valid
        with, a ``weighted=True`` instance (see :meth:`chunk_stats`); the returned
        sum is then the weighted ``sum (r/sigma)^2``.
        """
        n = len(x)
        mask = (
            jnp.ones(n, dtype=bool)
            if valid_size is None
            else jnp.arange(n) < valid_size
        )
        F, g, H = self.chunk_stats(theta, x, y, mask, sigma)
        # kernel cost is 0.5*sum r^2; callers want sum r^2
        return H_prev + H, g_prev + g, float(2.0 * F)


def cached_kernel(owner: Any) -> NormalEquations:
    """Chunk kernel for ``owner.normalized_model`` using ``owner._compute_jacobian_chunk``.

    Cached on ``owner._normal_eq`` as ``(model, jacobian_fn, kernel)`` and rebuilt
    whenever ``owner.normalized_model`` or ``owner._jacobian_fn_compiled`` is a
    different object than at the last build (the jitted kernel bakes both in).
    """
    model, jac = owner.normalized_model, owner._jacobian_fn_compiled
    cached = owner._normal_eq
    if cached is not None and cached[0] is model and cached[1] is jac:
        return cached[2]
    ne = NormalEquations(
        model,
        chunk=owner.config.chunk_size,  # only sizes run_pass; chunk_stats takes any length
        jacobian=lambda theta, x: owner._compute_jacobian_chunk(x, theta),
    )
    owner._normal_eq = (model, jac, ne)
    return ne
