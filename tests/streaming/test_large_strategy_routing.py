"""Routing table for the two large-dataset engines.

Pins, for each of the 13 places that construct ``AdaptiveHybridStreamingOptimizer``
(the "hybrid" engine) or ``LargeDatasetFitter`` (the "chunked" engine), what a
given (entry point, kwargs) builds: which class, the whole constructor argument
set, the config values (forwarded or dropped tolerances), how the engine's fit
method is called, and what the entry point returns. It guards the routing of those
construction sites through ``build_large_strategy``: the rows were written against
the pre-migration code and must keep passing unchanged.

Recording patches ``__init__``/``fit``/``fit_with_progress`` on the two real
classes (not module attributes), so it sees construction wherever the classes
are imported from. Arguments are normalised through the real signatures, so a
positional-vs-keyword change at a call site does not matter, but an added,
dropped or changed value does. Rows that would run a long fit stub the engine's
fit with sentinels whose fields all differ (``x`` vs ``popt``, hybrid vs chunked
``pcov``), so a wrong extraction fails; ``*real*`` rows run the real fit.

What a row catches in a refactor: a different class or a class built twice; a
hardcoded or dropped forwarded kwarg (``absolute_sigma``, ``sigma``, ``callback``,
bounds, tolerances, ``check_finite``, ``method``, extra solver kwargs); a config
swapped for a copy (or the reverse) or mutated in place (configs are
deep-copied when recorded and the caller's config is compared before/after);
list/tuple/ndarray conversions (``same`` is strict about container types); and
dropped result metadata or identity of results the production code returns as-is.

Public vs direct: every ``WORKFLOW_PRESETS`` key is rejected before
``_fit_with_preset`` (see ``test_legacy_preset_keys_rejected_publicly``), so its
two constructors are driven through the module-level helper directly.
``LargeDatasetFitter._fit_with_streaming`` has no caller anywhere; it is driven
directly and ``test_fitter_fit_never_builds_hybrid`` pins that it stays unwired.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq.config import get_memory_config
from nlsq.core import minpack
from nlsq.core.workflow import (
    MemoryBudgetSelector,
    OptimizationGoal,
    calculate_adaptive_tolerances,
)
from nlsq.result import CurveFitResult, OptimizeResult
from nlsq.streaming import adaptive_hybrid, large_dataset
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.large_dataset import LDMemoryConfig
from nlsq.streaming.specs import Chunked

HYBRID = "AdaptiveHybridStreamingOptimizer"
FITTER = "LargeDatasetFitter"
_REAL = {
    HYBRID: adaptive_hybrid.AdaptiveHybridStreamingOptimizer,
    FITTER: large_dataset.LargeDatasetFitter,
}
_METHODS = {
    HYBRID: ("__init__", "fit"),
    FITTER: ("__init__", "fit", "fit_with_progress"),
}

N = 600
X = np.linspace(0.0, 4.0, N)
Y = 2.5 * np.exp(-1.3 * X) + 0.01 * np.random.default_rng(0).standard_normal(N)
SIG = np.full(N, 0.02)
P0 = [2.0, 1.0]
TRUE = np.array([2.5, 1.3])
FINITE_BOUNDS = ([0.0, 0.0], [10.0, 5.0])
FINITE_BOUNDS_ARR = (np.array(FINITE_BOUNDS[0]), np.array(FINITE_BOUNDS[1]))
INF_BOUNDS = (-np.inf, np.inf)
_LOGGER = logging.getLogger("test_large_strategy_routing")


def _CALLBACK(*args: Any) -> None:
    pass


# Memory limits that force a MemoryBudgetSelector decision for N=600, 2 params:
# the 0.1 GB solver overhead in the peak estimate exceeds the limit -> chunked;
# a limit below the ~9e-6 GB of data -> streaming.
CHUNKED_LIMIT_GB = 0.1
STREAMING_LIMIT_GB = 1e-6
# Fast real hybrid fit (the default 200 warmup iterations takes ~60 s).
FAST_HYBRID = {
    "warmup_iterations": 5,
    "max_warmup_iterations": 10,
    "gauss_newton_max_iterations": 5,
}

# Sentinels: every field of the two engines' stub results differs, so taking
# x where popt is expected (or the reverse), or the wrong pcov, fails.
HYB_X = np.array([7.0, 8.0])
HYB_PCOV = np.array([[13.0, 1.0], [1.0, 17.0]])
HYB_FUN = np.array([0.5, -0.25, 0.125])
HYB_PERR = np.array([3.5, 4.5])
HYB_META = {
    "success": True,
    "message": "hybrid-stub",
    "fun": HYB_FUN,
    "perr": HYB_PERR,
    "streaming_diagnostics": {"marker": "hybrid-diag"},
    "nfev": 42,
}
FIT_X = np.array([21.0, 22.0])
FIT_POPT = np.array([23.0, 24.0])
FIT_PCOV = np.array([[29.0, 2.0], [2.0, 31.0]])
FIT_META = {
    "success": True,
    "message": "fitter-stub",
    "marker": "fitter-meta",
    "execution_time": 0.123,
}


def MODEL(x, a, b):
    return a * jnp.exp(-b * x)


# --------------------------------------------------------------------------
# Recording harness
# --------------------------------------------------------------------------


@dataclass
class Call:
    engine: str
    method: str
    named: dict[str, Any]  # live references, exactly as the call site passed them
    extra: dict[str, Any]  # the **kwargs the call site forwarded
    configs: dict[str, Any]  # deep copies of dataclass args taken at call time
    ambient: dict[str, Any]  # nlsq memory config in effect at call time


@dataclass
class Recorder:
    calls: list[Call] = field(default_factory=list)
    stubs: dict[str, Callable[[], Any]] = field(default_factory=dict)
    last_result: Any = None  # what the stubbed engine fit last returned

    def ctors(self, engine: str | None = None) -> list[Call]:
        return [
            c
            for c in self.calls
            if c.method == "__init__" and engine in (None, c.engine)
        ]

    def fits(self, engine: str, method: str = "fit") -> list[Call]:
        return [c for c in self.calls if c.engine == engine and c.method == method]

    def only_fit(self, engine: str, method: str = "fit") -> Call:
        (call,) = self.fits(engine, method)
        return call


def _wrap(rec: Recorder, engine: str, method: str, real: Callable) -> Callable:
    sig = inspect.signature(real)

    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        named: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for name, value in list(bound.arguments.items())[1:]:
            if sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                extra.update(value)
            else:
                named[name] = value
        configs = {
            k: copy.deepcopy(v)
            for k, v in named.items()
            if dataclasses.is_dataclass(v) and not isinstance(v, type)
        }
        ambient = dataclasses.asdict(get_memory_config())
        rec.calls.append(Call(engine, method, named, extra, configs, ambient))
        if method != "__init__" and engine in rec.stubs:
            rec.last_result = rec.stubs[engine]()  # a stub may raise
            return rec.last_result
        return real(self, *args, **kwargs)

    return wrapper


@pytest.fixture
def rec(monkeypatch) -> Recorder:
    recorder = Recorder()
    for engine, cls in _REAL.items():
        for method in _METHODS[engine]:
            real = getattr(cls, method)
            monkeypatch.setattr(cls, method, _wrap(recorder, engine, method, real))
    return recorder


def hybrid_ok() -> dict[str, Any]:
    return {
        "x": HYB_X.copy(),
        "pcov": HYB_PCOV.copy(),
        **copy.deepcopy(HYB_META),
    }


def hybrid_no_pcov() -> dict[str, Any]:
    result = hybrid_ok()
    del result["pcov"]
    return result


def hybrid_failed() -> dict[str, Any]:
    return {**hybrid_ok(), "success": False}


def fitter_ok() -> OptimizeResult:
    return OptimizeResult(
        x=FIT_X.copy(), popt=FIT_POPT.copy(), pcov=FIT_PCOV.copy(), **FIT_META
    )


def fitter_x_only() -> OptimizeResult:
    return OptimizeResult(x=FIT_X.copy(), success=True, message="fitter-stub")


def fitter_curve_fit_result() -> CurveFitResult:
    return CurveFitResult(fitter_ok())


def fitter_failed() -> OptimizeResult:
    return OptimizeResult(x=FIT_X.copy(), success=False, message="boom")


def same(a: Any, b: Any) -> bool:
    """Strict equality: arrays by value, but list/tuple/ndarray and scalar types
    must match, so a changed conversion shows up."""
    if a is None or b is None:
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(same(i, j) for i, j in zip(a, b, strict=True))
        )
    if hasattr(a, "shape") or hasattr(b, "shape"):
        return (
            hasattr(a, "shape")
            and hasattr(b, "shape")
            and np.shape(a) == np.shape(b)
            and bool(np.array_equal(np.asarray(a), np.asarray(b)))
        )
    return type(a) is type(b) and bool(a == b)


def _short(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_short(v)}" for k, v in value.items()) + "}"
    if hasattr(value, "shape") and np.size(value) > 6:
        return f"<array shape={np.shape(value)} first={np.ravel(value)[:3]}>"
    return f"{value!r}:{type(value).__name__}"


def expect(actual: Any, expected: Any, what: str) -> None:
    assert same(actual, expected), (
        f"{what}: got {_short(actual)}, expected {_short(expected)}"
    )


def expect_config(actual: Any, expected: Any, what: str) -> None:
    """Whole-config equality: any value changed, added or dropped fails."""
    assert type(actual) is type(expected), f"{what}: {type(actual)} vs {type(expected)}"
    expect(dataclasses.asdict(actual), dataclasses.asdict(expected), what)


def snap(config: Any) -> dict[str, Any]:
    return copy.deepcopy(dataclasses.asdict(config))


def expect_unchanged(config: Any, snapshot: dict[str, Any], what: str) -> None:
    expect(dataclasses.asdict(config), snapshot, f"{what} (caller's config mutated)")


def expect_hybrid_ctor(rec: Recorder, expected_config: Any, what: str) -> Call:
    (ctor,) = rec.ctors(HYBRID)
    assert set(ctor.named) == {"config"}
    expect_config(ctor.configs["config"], expected_config, what)
    return ctor


_FITTER_CTOR_DEFAULTS = {
    "curve_fit_class": None,
    "logger": None,
    "multistart": False,
    "n_starts": 10,
    "sampler": "lhs",
}


def expect_fitter_ctor(
    rec: Recorder, *, memory_limit_gb: Any, config: Any, **overrides: Any
) -> Call:
    """The whole recorded constructor argument set (config compared by value)."""
    (ctor,) = rec.ctors(FITTER)
    named = dict(ctor.named)
    named.pop("config")
    expected = {
        "memory_limit_gb": memory_limit_gb,
        **_FITTER_CTOR_DEFAULTS,
        **overrides,
    }
    expect(named, expected, "LargeDatasetFitter ctor args")
    if config is None:
        assert ctor.named["config"] is None
    else:
        expect_config(ctor.configs["config"], config, "LargeDatasetFitter config")
    return ctor


def expect_hybrid_fit(
    call: Call,
    *,
    p0: Any,
    bounds: Any,
    sigma: Any,
    sigma_is: bool = False,
    absolute_sigma: bool = False,
    callback: Any = None,
    verbose: int = 1,
) -> None:
    n = call.named
    assert set(n) == {
        "data_source",
        "func",
        "p0",
        "bounds",
        "sigma",
        "absolute_sigma",
        "callback",
        "verbose",
    }
    assert isinstance(n["data_source"], tuple) and len(n["data_source"]) == 2
    expect(n["data_source"][0], X, "hybrid data_source x")
    expect(n["data_source"][1], Y, "hybrid data_source y")
    assert n["func"] is MODEL
    expect(n["p0"], p0, "hybrid p0")
    expect(n["bounds"], bounds, "hybrid bounds")
    expect(n["sigma"], sigma, "hybrid sigma")
    if sigma_is:
        assert n["sigma"] is sigma
    assert n["absolute_sigma"] is absolute_sigma
    assert n["callback"] is callback
    assert n["verbose"] == verbose
    assert call.extra == {}  # nothing else (no ftol/xtol/gtol) reaches the engine


def expect_fitter_fit(
    call: Call,
    *,
    p0: Any,
    bounds: Any,
    method: str,
    extra: dict[str, Any],
    p0_is: bool = False,
    bounds_is: bool = False,
    solver: str = "auto",
) -> None:
    n = call.named
    assert set(n) == {"f", "xdata", "ydata", "p0", "bounds", "method", "solver"}
    assert n["f"] is MODEL
    expect(n["xdata"], X, "fitter xdata")
    expect(n["ydata"], Y, "fitter ydata")
    expect(n["p0"], p0, "fitter p0")
    expect(n["bounds"], bounds, "fitter bounds")
    if p0_is:
        assert n["p0"] is p0
    if bounds_is:
        assert n["bounds"] is bounds
    assert n["method"] == method
    assert n["solver"] == solver
    expect(call.extra, extra, "forwarded fit kwargs")
    for key in ("sigma", "gtol"):  # identity of array/py objects passed through
        if key in extra and isinstance(extra[key], np.ndarray):
            assert call.extra[key] is extra[key]


def expect_curve_fit_result(
    result: Any, *, x: Any, pcov: Any, meta: dict[str, Any], model: bool = True
) -> None:
    """Hand-off of the _fit_with_* helpers: engine result + model/xdata/ydata."""
    assert isinstance(result, CurveFitResult)
    if model:
        assert result["model"] is MODEL
        expect(result["xdata"], X, "result xdata")
        expect(result["ydata"], Y, "result ydata")
    else:
        assert "model" not in result
    expect(result["x"], x, "result x")
    expect(result["pcov"], pcov, "result pcov")
    for key, value in meta.items():  # engine metadata survives the hand-off
        expect(result[key], value, f"result[{key!r}]")


def close_to_truth(popt: Any) -> None:
    np.testing.assert_allclose(np.asarray(popt), TRUE, atol=0.05)


# --------------------------------------------------------------------------
# The routing table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    id: str
    site: str  # construction site (enclosing function) this row exercises
    engine: str | None  # class that must be built exactly once; None = neither
    drive: Callable[[SimpleNamespace], Any]
    check: Callable[[Recorder, Any, SimpleNamespace], None]
    stubs: dict[str, Callable[[], Any]] = field(default_factory=dict)
    # Constructions the drive function itself performs before the routed one.
    prebuilt: tuple[str, ...] = ()
    marks: tuple[Any, ...] = ()


SLOW = (pytest.mark.slow,)
# Enters nlsq.config.memory_context/large_dataset_context (process-global state).
SERIAL = (pytest.mark.serial,)


def _call_preset(key: str, *, n_points: int = N, **overrides: Any) -> Any:
    args: dict[str, Any] = {
        "f": MODEL,
        "xdata": X,
        "ydata": Y,
        "p0": P0,
        "sigma": None,
        "absolute_sigma": False,
        "check_finite": True,
        "bounds": INF_BOUNDS,
        "method": None,
        "preset": minpack.WORKFLOW_PRESETS[key],
        "goal": None,
        "n_points": n_points,
    }
    args.update(overrides)
    return minpack._fit_with_preset(**args)


# ---- nlsq.fit (preset="streaming") ----------------------------------------


def _drive_fit_preset_streaming(ctx: SimpleNamespace, **kw: Any) -> Any:
    with pytest.warns(FutureWarning, match="preset"):
        if "ftol" in kw or "xtol" in kw:
            # no hybrid equivalent: warned about, not silently dropped
            with pytest.warns(UserWarning, match="ftol"):
                return nlsq.fit(MODEL, X, Y, p0=P0, preset="streaming", **kw)
        return nlsq.fit(MODEL, X, Y, p0=P0, preset="streaming", **kw)


def _drive_fit_preset_streaming_full(ctx: SimpleNamespace) -> Any:
    return _drive_fit_preset_streaming(
        ctx,
        sigma=SIG,
        absolute_sigma=True,
        bounds=FINITE_BOUNDS,
        callback=_CALLBACK,
        verbose=0,
        multistart=True,
        n_starts=4,
        sampler="sobol",
        gtol=1e-3,
        ftol=1e-3,
        xtol=1e-3,
    )


def _check_fit_preset_streaming(rec: Recorder, result: Any, ctx: Any) -> None:
    # gtol maps to gauss_newton_tol; ftol/xtol have no hybrid equivalent (warned,
    # not forwarded). Multistart settings reach the config, and
    # sigma/absolute_sigma/callback/verbose/bounds reach fit().
    expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(
            enable_multistart=True,
            n_starts=4,
            multistart_sampler="sobol",
            gauss_newton_tol=1e-3,
        ),
        "fit(preset=streaming) config",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        sigma_is=True,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
    )
    # No model/xdata/ydata on this route; engine metadata is kept.
    expect_curve_fit_result(result, x=HYB_X, pcov=HYB_PCOV, meta=HYB_META, model=False)
    expect(
        result["multistart_diagnostics"],
        {"n_starts_configured": 4, "bypassed": False, "preset": "streaming"},
        "multistart_diagnostics",
    )


def _check_fit_preset_streaming_off(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(enable_multistart=False, n_starts=10),
        "multistart=False config",
    )
    expect_hybrid_fit(rec.only_fit(HYBRID), p0=np.array(P0), bounds=None, sigma=None)
    expect(
        result["multistart_diagnostics"],
        {"n_starts_configured": 10, "bypassed": True, "preset": "streaming"},
        "multistart_diagnostics",
    )


# ---- nlsq.curve_fit_large --------------------------------------------------


def _drive_clf_hybrid(ctx: SimpleNamespace, **kw: Any) -> Any:
    return nlsq.curve_fit_large(
        MODEL, X, Y, p0=P0, method="hybrid_streaming", verbose=0, **FAST_HYBRID, **kw
    )


def _check_clf_hybrid_real(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_hybrid_ctor(
        rec, HybridStreamingConfig(**FAST_HYBRID), "curve_fit_large hybrid config"
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=None,
        sigma=None,
        verbose=0,
    )
    # A (popt, pcov) tuple of arrays, not a result object.
    assert type(result) is tuple and len(result) == 2
    close_to_truth(result[0])
    assert np.asarray(result[1]).shape == (2, 2) and np.all(np.isfinite(result[1]))


def _drive_clf_hybrid_full(ctx: SimpleNamespace) -> Any:
    return _drive_clf_hybrid(ctx, sigma=SIG, absolute_sigma=True, bounds=FINITE_BOUNDS)


def _check_clf_hybrid_stub(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_hybrid_ctor(
        rec, HybridStreamingConfig(**FAST_HYBRID), "curve_fit_large hybrid config"
    )
    # callback is not accepted on this route, so the engine always gets None.
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        sigma_is=True,
        absolute_sigma=True,
        verbose=0,
    )
    assert type(result) is tuple and len(result) == 2
    assert result[0] is rec.last_result["x"]  # engine's arrays, unwrapped
    assert result[1] is rec.last_result["pcov"]


def _check_clf_hybrid_no_pcov(rec: Recorder, result: Any, ctx: Any) -> None:
    assert type(result) is tuple and len(result) == 2
    assert result[0] is rec.last_result["x"]
    expect(result[1], np.full((2, 2), np.inf), "pcov fallback is all-inf")


def _drive_clf_chunked(ctx: SimpleNamespace, **kw: Any) -> Any:
    return nlsq.curve_fit_large(MODEL, X, Y, p0=P0, size_threshold=100, **kw)


def _drive_clf_chunked_pinned(
    ctx: SimpleNamespace, chunk_size: int | None = None, **kw: Any
) -> Any:
    # The default large path is exact streaming; Chunked() selects the chunked engine.
    return nlsq.curve_fit_large(
        MODEL, X, Y, p0=P0, strategy=Chunked(chunk_size=chunk_size), **kw
    )


def _expect_clf_ambient(call: Call, *, size: int, progress: bool) -> None:
    # The fitter is built and run inside memory_context(chunk bounds, progress).
    assert call.ambient["min_chunk_size"] == size
    assert call.ambient["max_chunk_size"] == size
    assert call.ambient["progress_reporting"] is progress


def _check_clf_chunked_real(rec: Recorder, result: Any, ctx: Any) -> None:
    # Ctor gets the limit and a config only: multistart & co. go to fit().
    ctor = expect_fitter_ctor(
        rec,
        memory_limit_gb=2.0,
        config=LDMemoryConfig(
            memory_limit_gb=2.0, min_chunk_size=600, max_chunk_size=600
        ),
    )
    call = rec.only_fit(FITTER)
    _expect_clf_ambient(ctor, size=600, progress=False)
    _expect_clf_ambient(call, size=600, progress=False)
    expect_fitter_fit(
        call,
        p0=P0,
        p0_is=True,
        bounds=INF_BOUNDS,
        method="trf",
        extra={
            "absolute_sigma": False,
            "check_finite": True,
            "multistart": False,
            "n_starts": 10,
            "sampler": "lhs",
            "center_on_p0": True,
            "scale_factor": 1.0,
        },
    )
    assert type(result) is tuple and len(result) == 2
    close_to_truth(result[0])
    assert np.asarray(result[1]).shape == (2, 2)


def _drive_clf_chunked_args(ctx: SimpleNamespace) -> Any:
    # method="lm" and multistart reach chunked through the fallback, which warns
    with pytest.warns(UserWarning, match="exact streaming not used"):
        return _drive_clf_chunked(
            ctx,
            memory_limit_gb=3.0,
            chunk_size=300,
            method="lm",
            sigma=SIG,
            absolute_sigma=True,
            check_finite=False,
            bounds=FINITE_BOUNDS,
            multistart=True,
            n_starts=5,
            sampler="sobol",
            center_on_p0=False,
            scale_factor=2.0,
            gtol=1e-3,
            max_nfev=7,
        )


def _check_clf_chunked_args(rec: Recorder, result: Any, ctx: Any) -> None:
    ctor = expect_fitter_ctor(
        rec,
        memory_limit_gb=3.0,
        config=LDMemoryConfig(
            memory_limit_gb=3.0, min_chunk_size=300, max_chunk_size=300
        ),
    )
    call = rec.only_fit(FITTER)
    _expect_clf_ambient(ctor, size=300, progress=False)
    _expect_clf_ambient(call, size=300, progress=False)
    expect_fitter_fit(
        call,
        p0=P0,
        p0_is=True,
        bounds=FINITE_BOUNDS,
        bounds_is=True,
        method="lm",
        extra={
            "sigma": SIG,
            "absolute_sigma": True,
            "check_finite": False,
            "multistart": True,
            "n_starts": 5,
            "sampler": "sobol",
            "center_on_p0": False,
            "scale_factor": 2.0,
            "gtol": 1e-3,
            "max_nfev": 7,
        },
    )
    assert call.extra["sigma"] is SIG
    assert type(result) is tuple and len(result) == 2
    assert result[0] is rec.last_result.popt  # popt, not x
    assert result[1] is rec.last_result.pcov


def _drive_clf_chunked_x_only(ctx: SimpleNamespace) -> Any:
    # The table loop runs drive() before check(): hand the error to the check.
    try:
        return _drive_clf_chunked_pinned(ctx, memory_limit_gb=2.0)
    except RuntimeError as exc:
        return exc


def _check_clf_chunked_x_only(rec: Recorder, result: Any, ctx: Any) -> None:
    assert isinstance(result, RuntimeError), (
        "no fabricated pcov when the result has none"
    )


def _drive_clf_chunked_progress(ctx: SimpleNamespace) -> Any:
    return _drive_clf_chunked_pinned(
        ctx, memory_limit_gb=2.0, show_progress=True, chunk_size=450
    )


def _check_clf_chunked_progress(rec: Recorder, result: Any, ctx: Any) -> None:
    assert rec.fits(FITTER, "fit") == []
    ctor = expect_fitter_ctor(
        rec,
        memory_limit_gb=2.0,
        config=LDMemoryConfig(
            memory_limit_gb=2.0, min_chunk_size=450, max_chunk_size=450
        ),
    )
    call = rec.only_fit(FITTER, "fit_with_progress")
    _expect_clf_ambient(ctor, size=450, progress=True)
    _expect_clf_ambient(call, size=450, progress=True)
    expect_fitter_fit(
        call,
        p0=P0,
        p0_is=True,
        bounds=INF_BOUNDS,
        method="trf",
        extra={
            "absolute_sigma": False,
            "check_finite": True,
            "multistart": False,
            "n_starts": 10,
            "sampler": "lhs",
            "center_on_p0": True,
            "scale_factor": 1.0,
        },
    )
    assert result[0] is rec.last_result.popt
    assert result[1] is rec.last_result.pcov


def _check_clf_small(rec: Recorder, result: Any, ctx: Any) -> None:
    # Below size_threshold the engine is bypassed entirely.
    assert rec.calls == []
    assert isinstance(result, CurveFitResult)
    close_to_truth(result.x)


# ---- nlsq.streaming.large_dataset -----------------------------------------


def _drive_fit_large_dataset(ctx: SimpleNamespace) -> Any:
    return large_dataset.fit_large_dataset(
        MODEL, X, Y, p0=P0, memory_limit_gb=2.0, multistart=False, n_starts=3
    )


def _check_fit_large_dataset_real(rec: Recorder, result: Any, ctx: Any) -> None:
    # Loose limits, no config object; multistart is a ctor arg here.
    expect_fitter_ctor(rec, memory_limit_gb=2.0, config=None, n_starts=3)
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=P0,
        p0_is=True,
        bounds=INF_BOUNDS,
        method="trf",
        extra={},
    )
    assert isinstance(result, OptimizeResult) and result.success
    close_to_truth(result.x)


def _drive_fit_large_dataset_stub(ctx: SimpleNamespace) -> Any:
    return large_dataset.fit_large_dataset(
        MODEL,
        X,
        Y,
        p0=P0,
        memory_limit_gb=2.0,
        logger=_LOGGER,
        multistart=True,
        n_starts=3,
        sampler="halton",
        method="lm",
        bounds=FINITE_BOUNDS,
        ftol=1e-4,
    )


def _check_fit_large_dataset_stub(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_fitter_ctor(
        rec,
        memory_limit_gb=2.0,
        config=None,
        logger=_LOGGER,
        multistart=True,
        n_starts=3,
        sampler="halton",
    )
    (ctor,) = rec.ctors(FITTER)
    assert ctor.named["logger"] is _LOGGER
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=P0,
        p0_is=True,
        bounds=FINITE_BOUNDS,
        bounds_is=True,
        method="lm",
        extra={"ftol": 1e-4},
    )
    assert result is rec.last_result  # the engine's result, returned as-is


def _drive_fit_large_dataset_progress(ctx: SimpleNamespace) -> Any:
    return large_dataset.fit_large_dataset(
        MODEL,
        X,
        Y,
        p0=P0,
        memory_limit_gb=2.0,
        show_progress=True,
        bounds=FINITE_BOUNDS,
    )


def _check_fit_large_dataset_progress(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_fitter_ctor(rec, memory_limit_gb=2.0, config=None)
    assert rec.fits(FITTER, "fit") == []
    expect_fitter_fit(
        rec.only_fit(FITTER, "fit_with_progress"),
        p0=P0,
        p0_is=True,
        bounds=FINITE_BOUNDS,
        bounds_is=True,
        method="trf",
        extra={},
    )
    assert result is rec.last_result


def _drive_fitter_streaming(ctx: SimpleNamespace) -> Any:
    # Built by hand: LargeDatasetFitter.fit never reaches _fit_with_streaming.
    fitter = _REAL[FITTER](
        config=LDMemoryConfig(
            memory_limit_gb=1.0, streaming_batch_size=123, streaming_max_epochs=7
        )
    )
    return fitter._fit_with_streaming(
        MODEL,
        X,
        Y,
        [2.0, 1.0],
        FINITE_BOUNDS,
        "trf",
        "auto",
        False,
        sigma=SIG,  # accepted via **kwargs and silently unused
        absolute_sigma=True,
        callback=_CALLBACK,
    )


def _check_fitter_streaming(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(chunk_size=123, gauss_newton_max_iterations=7),
        "_fit_with_streaming config",
    )
    # sigma/absolute_sigma/callback are not forwarded; bounds are passed through
    # unprepared; verbose is 1.
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS,
        sigma=None,
        verbose=1,
    )
    assert rec.only_fit(HYBRID).named["bounds"] is FINITE_BOUNDS
    engine = rec.last_result
    assert isinstance(result, OptimizeResult)
    assert result.x is engine["x"]
    assert result.popt is engine["x"]
    assert result.pcov is engine["pcov"]
    assert result.fun is engine["fun"]
    assert result.message == "hybrid-stub"
    assert result.success is True and result.nfev == N


def _check_fitter_streaming_failure(rec: Recorder, result: Any, ctx: Any) -> None:
    assert isinstance(result, OptimizeResult) and result.success is False
    assert "Streaming fit failed: kaboom" in result.message
    expect(result.x, np.array(P0), "falls back to p0")
    expect(result.pcov, np.full((2, 2), np.inf), "pcov is all-inf, not identity")


def _raise_kaboom() -> Any:
    raise RuntimeError("kaboom")


# ---- nlsq.core.minpack._fit_with_config -----------------------------------


def _drive_workflow_hybrid_config(ctx: SimpleNamespace, **kw: Any) -> Any:
    ctx.config = HybridStreamingConfig(**FAST_HYBRID)
    ctx.before = snap(ctx.config)
    return nlsq.fit(MODEL, X, Y, workflow=ctx.config, verbose=0, **kw)


def _drive_workflow_hybrid_full(ctx: SimpleNamespace) -> Any:
    return _drive_workflow_hybrid_config(
        ctx,
        p0=P0,
        gtol=1e-3,
        ftol=1e-4,
        xtol=1e-5,
        sigma=SIG,
        absolute_sigma=True,
        bounds=FINITE_BOUNDS,
        callback=_CALLBACK,
    )


def _check_workflow_hybrid_config(rec: Recorder, result: Any, ctx: Any) -> None:
    # gtol is mapped into gauss_newton_tol on a *copy*; ftol/xtol have no target.
    ctor = expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(gauss_newton_tol=1e-3, **FAST_HYBRID),
        "workflow=HybridStreamingConfig config",
    )
    assert ctor.named["config"] is not ctx.config
    expect_unchanged(ctx.config, ctx.before, "workflow=HybridStreamingConfig")
    assert ctx.config.gauss_newton_tol == HybridStreamingConfig().gauss_newton_tol
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
    )
    expect_curve_fit_result(result, x=HYB_X, pcov=HYB_PCOV, meta=HYB_META)


def _check_workflow_hybrid_goal(rec: Recorder, result: Any, ctx: Any) -> None:
    gtol = calculate_adaptive_tolerances(N, OptimizationGoal.FAST)["gtol"]
    ctor = expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(gauss_newton_tol=gtol, **FAST_HYBRID),
        "goal-derived gtol reaches gauss_newton_tol",
    )
    assert ctor.named["config"] is not ctx.config
    expect_unchanged(ctx.config, ctx.before, "goal route")
    expect_hybrid_fit(
        rec.only_fit(HYBRID), p0=np.ones(2), bounds=None, sigma=None, verbose=0
    )


def _check_workflow_hybrid_no_gtol(rec: Recorder, result: Any, ctx: Any) -> None:
    # Without gtol the caller's config object is used as-is (same object).
    ctor = expect_hybrid_ctor(
        rec, HybridStreamingConfig(**FAST_HYBRID), "no gtol -> config untouched"
    )
    assert ctor.named["config"] is ctx.config
    expect_unchanged(ctx.config, ctx.before, "no-gtol route")
    expect_curve_fit_result(
        result,
        x=HYB_X,
        pcov=np.full((2, 2), np.inf),
        meta=HYB_META,
    )


def _check_workflow_hybrid_real(rec: Recorder, result: Any, ctx: Any) -> None:
    ctor = expect_hybrid_ctor(
        rec, HybridStreamingConfig(**FAST_HYBRID), "real hybrid config"
    )
    assert ctor.named["config"] is ctx.config
    expect_unchanged(ctx.config, ctx.before, "real hybrid route")
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=None,
        sigma=None,
        absolute_sigma=True,
        verbose=0,
    )
    assert isinstance(result, CurveFitResult)
    close_to_truth(result["x"])
    assert result["model"] is MODEL
    for key in ("fun", "perr", "streaming_diagnostics", "success", "message"):
        assert key in result  # the real engine's metadata survives
    assert np.asarray(result["pcov"]).shape == (2, 2)
    assert np.all(np.isfinite(result["pcov"]))


LD_BOUNDS = FINITE_BOUNDS


def _drive_workflow_ld_config(ctx: SimpleNamespace, **kw: Any) -> Any:
    ctx.config = LDMemoryConfig(
        memory_limit_gb=2.0, min_chunk_size=100, max_chunk_size=300
    )
    ctx.before = snap(ctx.config)
    return nlsq.fit(MODEL, X, Y, p0=P0, workflow=ctx.config, **kw)


def _drive_workflow_ld_full(ctx: SimpleNamespace) -> Any:
    return _drive_workflow_ld_config(
        ctx,
        method="lm",
        gtol=1e-3,
        ftol=1e-4,
        xtol=1e-5,
        sigma=SIG,
        absolute_sigma=True,
        check_finite=False,
        bounds=FINITE_BOUNDS,
        max_nfev=7,
    )


def _check_workflow_ld_config(rec: Recorder, result: Any, ctx: Any) -> None:
    ctor = expect_fitter_ctor(rec, memory_limit_gb=2.0, config=ctx.before_config())
    assert ctor.named["config"] is ctx.config  # used as given, not copied
    expect_unchanged(ctx.config, ctx.before, "workflow=LDMemoryConfig")
    # sigma/absolute_sigma/check_finite/method and all three tolerances are
    # forwarded (unlike the hybrid route). n_starts=None remains a public
    # passthrough; the memory budget is consumed by the constructor/config.
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS,
        bounds_is=True,
        method="lm",
        extra={
            "sigma": SIG,
            "absolute_sigma": True,
            "check_finite": False,
            "n_starts": None,
            "gtol": 1e-3,
            "ftol": 1e-4,
            "xtol": 1e-5,
            "max_nfev": 7,
        },
    )
    # Plain OptimizeResult is wrapped; metadata and model/xdata/ydata attached.
    expect_curve_fit_result(result, x=FIT_X, pcov=FIT_PCOV, meta=FIT_META)


def _check_workflow_ld_passthrough(rec: Recorder, result: Any, ctx: Any) -> None:
    assert result is rec.last_result  # an engine CurveFitResult is returned as-is


def _check_workflow_ld_config_real(rec: Recorder, result: Any, ctx: Any) -> None:
    ctor = expect_fitter_ctor(rec, memory_limit_gb=2.0, config=ctx.before_config())
    assert ctor.named["config"] is ctx.config
    expect_unchanged(ctx.config, ctx.before, "real LD route")
    assert len(rec.ctors()) == 1
    assert isinstance(result, CurveFitResult)
    close_to_truth(result["x"])
    assert result["model"] is MODEL


def _check_auto_chunked(rec: Recorder, result: Any, ctx: Any) -> None:
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=CHUNKED_LIMIT_GB
    )
    assert isinstance(expected, LDMemoryConfig)
    expect_fitter_ctor(rec, memory_limit_gb=expected.memory_limit_gb, config=expected)
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=np.array(P0),
        bounds=INF_BOUNDS,
        method="trf",
        extra={  # memory_limit_gb is consumed by the selector
            "sigma": None,
            "absolute_sigma": False,
            "check_finite": True,
            "n_starts": None,
            "gtol": 1e-3,
        },
    )
    expect_curve_fit_result(result, x=FIT_X, pcov=FIT_PCOV, meta=FIT_META)


def _drive_auto_streaming(ctx: SimpleNamespace) -> Any:
    return nlsq.fit(
        MODEL,
        X,
        Y,
        p0=P0,
        workflow="auto",
        memory_limit_gb=STREAMING_LIMIT_GB,
        gtol=1e-3,
        ftol=1e-4,
        xtol=1e-5,
        sigma=SIG,
        absolute_sigma=True,
        bounds=FINITE_BOUNDS,
        callback=_CALLBACK,
        verbose=0,
    )


def _check_auto_streaming(rec: Recorder, result: Any, ctx: Any) -> None:
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=STREAMING_LIMIT_GB
    )
    assert isinstance(expected, HybridStreamingConfig)
    expect_hybrid_ctor(
        rec,
        dataclasses.replace(expected, gauss_newton_tol=1e-3),
        "selector's streaming config + gtol",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
    )
    expect_curve_fit_result(result, x=HYB_X, pcov=HYB_PCOV, meta=HYB_META)


def _check_auto_standard(rec: Recorder, result: Any, ctx: Any) -> None:
    assert rec.calls == []  # data fits: neither engine is built
    assert isinstance(result, CurveFitResult)
    close_to_truth(result["popt"] if "popt" in result else result["x"])


# ---- nlsq.core.minpack._fit_with_preset (direct helper) --------------------


def _drive_preset_chunked(ctx: SimpleNamespace) -> Any:
    return _call_preset(
        "large_robust",
        n_points=1_000_000,
        method="lm",
        sigma=SIG,
        absolute_sigma=True,
        check_finite=False,
        max_nfev=7,
    )


def _check_preset_chunked(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_fitter_ctor(
        rec,
        memory_limit_gb=8.0,
        config=LDMemoryConfig(
            memory_limit_gb=8.0, min_chunk_size=1000, max_chunk_size=100_000
        ),
    )
    # sigma, absolute_sigma and check_finite are dropped on this route; method,
    # the preset's multistart settings and tolerances and extra kwargs are forwarded.
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=np.array(P0),
        bounds=INF_BOUNDS,
        bounds_is=True,
        method="lm",
        extra={
            "multistart": True,
            "n_starts": 10,
            "gtol": 1e-8,
            "ftol": 1e-8,
            "xtol": 1e-8,
            "max_nfev": 7,
        },
    )
    expect_curve_fit_result(result, x=FIT_X, pcov=FIT_PCOV, meta=FIT_META)


def _drive_preset_chunked_passthrough(ctx: SimpleNamespace) -> Any:
    return _call_preset("large_robust", n_points=1_000_000)


def _drive_preset_streaming(ctx: SimpleNamespace) -> Any:
    return _call_preset(
        "streaming",
        sigma=SIG,
        absolute_sigma=True,
        bounds=FINITE_BOUNDS,
        callback=_CALLBACK,
        verbose=0,
        ftol=1e-4,
        xtol=1e-5,
    )


def _check_preset_streaming(rec: Recorder, result: Any, ctx: Any) -> None:
    # gauss_newton_tol comes from the preset's gtol (1e-7); explicit ftol/xtol are
    # dropped; sigma/absolute_sigma/callback/verbose/bounds reach the engine.
    expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=1e-7,
            enable_checkpoints=False,
            enable_multistart=False,
            n_starts=0,
        ),
        "streaming preset config",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
    )
    expect_curve_fit_result(result, x=HYB_X, pcov=HYB_PCOV, meta=HYB_META)


def _drive_preset_hpc(ctx: SimpleNamespace) -> Any:
    return _call_preset(
        "hpc_distributed",
        sigma=SIG,
        absolute_sigma=True,
        bounds=FINITE_BOUNDS,
        gtol=1e-5,
    )


def _check_preset_hpc(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=1e-5,  # explicit gtol beats the preset's 1e-6
            enable_checkpoints=True,
            enable_multistart=True,
            n_starts=10,
        ),
        "hpc_distributed preset config",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        absolute_sigma=True,
    )


def _check_preset_goal_tolerance(rec: Recorder, result: Any, ctx: Any) -> None:
    gtol = calculate_adaptive_tolerances(N, OptimizationGoal.QUALITY)["gtol"]
    assert gtol != 1e-7
    expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=gtol,
            enable_checkpoints=False,
            enable_multistart=False,
            n_starts=0,
        ),
        "goal-derived gtol overrides the preset's",
    )


# ---- nlsq.core.minpack._fit_global_multistart ------------------------------


def _drive_auto_global(limit: float | None, **kw: Any) -> Any:
    return nlsq.fit(
        MODEL,
        X,
        Y,
        p0=P0,
        workflow="auto_global",
        bounds=FINITE_BOUNDS,
        method="multi-start",
        memory_limit_gb=limit,
        n_starts=6,
        **kw,
    )


def _drive_global_chunked(ctx: SimpleNamespace) -> Any:
    return _drive_auto_global(CHUNKED_LIMIT_GB, absolute_sigma=True, check_finite=False)


def _check_global_chunked(rec: Recorder, result: Any, ctx: Any) -> None:
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=CHUNKED_LIMIT_GB
    )
    # multistart/n_starts go to fit(), not to the constructor.
    expect_fitter_ctor(rec, memory_limit_gb=expected.memory_limit_gb, config=expected)
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        method="trf",
        extra={
            "multistart": True,
            "n_starts": 6,
            "check_finite": False,
            "absolute_sigma": True,
        },
    )
    expect_curve_fit_result(result, x=FIT_X, pcov=FIT_PCOV, meta=FIT_META)


def _drive_global_streaming(ctx: SimpleNamespace) -> Any:
    return _drive_auto_global(
        STREAMING_LIMIT_GB,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
        gtol=1e-4,
        ftol=1e-5,
        xtol=1e-6,
    )


def _check_global_streaming(rec: Recorder, result: Any, ctx: Any) -> None:
    _, base = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=STREAMING_LIMIT_GB
    )
    assert isinstance(base, HybridStreamingConfig)
    # Only gtol is used (ftol/xtol dropped); the selector's chunk_size is kept.
    expect_hybrid_ctor(
        rec,
        dataclasses.replace(
            base,
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=1e-4,
            enable_multistart=True,
            n_starts=6,
        ),
        "multistart streaming config",
    )
    # Unlike the other hybrid sites, bounds are passed on unconditionally.
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
    )
    expect_curve_fit_result(result, x=HYB_X, pcov=HYB_PCOV, meta=HYB_META)


def _call_multistart(strategy: str, **overrides: Any) -> Any:
    args: dict[str, Any] = {
        "f": MODEL,
        "xdata": X,
        "ydata": Y,
        "p0": P0,
        "sigma": None,
        "absolute_sigma": False,
        "check_finite": True,
        "bounds": FINITE_BOUNDS_ARR,
        "strategy": strategy,
        "n_starts": 3,
    }
    args.update(overrides)
    return minpack._fit_global_multistart(**args)


def _drive_multistart_default_chunked(ctx: SimpleNamespace) -> Any:
    return _call_multistart("chunked", memory_config=None, absolute_sigma=True)


def _check_multistart_default_chunked(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_fitter_ctor(
        rec,
        memory_limit_gb=8.0,
        config=LDMemoryConfig(
            memory_limit_gb=8.0, min_chunk_size=1000, max_chunk_size=1_000_000
        ),
    )
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        bounds_is=True,
        method="trf",
        extra={
            "multistart": True,
            "n_starts": 3,
            "check_finite": True,
            "absolute_sigma": True,
        },
    )


def _drive_multistart_given_chunked(ctx: SimpleNamespace) -> Any:
    ctx.config = LDMemoryConfig(
        memory_limit_gb=0.75,
        min_chunk_size=111,
        max_chunk_size=222,
        safety_factor=0.5,
        use_streaming=False,
    )
    ctx.before = snap(ctx.config)
    return _call_multistart("chunked", memory_config=ctx.config)


def _check_multistart_given_chunked(rec: Recorder, result: Any, ctx: Any) -> None:
    ctor = expect_fitter_ctor(rec, memory_limit_gb=0.75, config=ctx.before_config())
    assert ctor.named["config"] is ctx.config  # reused as-is, not copied
    expect_unchanged(ctx.config, ctx.before, "multistart chunked memory_config")


def _drive_multistart_default_streaming(ctx: SimpleNamespace) -> Any:
    return _call_multistart(
        "streaming",
        memory_config=None,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
        ftol=1e-5,
    )


def _check_multistart_default_streaming(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=1e-8,
            enable_multistart=True,
            n_starts=3,
        ),
        "fallback streaming config without memory_config",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
    )


def _drive_multistart_given_streaming(ctx: SimpleNamespace) -> Any:
    ctx.config = HybridStreamingConfig(
        chunk_size=17,
        normalize=False,
        warmup_iterations=11,
        gauss_newton_tol=0.5,
        enable_multistart=False,
        n_starts=2,
    )
    ctx.before = snap(ctx.config)
    return _call_multistart("streaming", memory_config=ctx.config)


def _check_multistart_given_streaming(rec: Recorder, result: Any, ctx: Any) -> None:
    ctor = expect_hybrid_ctor(
        rec,
        dataclasses.replace(
            ctx.config,
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=1e-8,
            enable_multistart=True,
            n_starts=3,
        ),
        "memory_config copy with multistart overrides",
    )
    assert ctor.named["config"] is not ctx.config  # a modified copy
    assert ctor.configs["config"].chunk_size == 17  # the budget's chunk size kept
    expect_unchanged(ctx.config, ctx.before, "multistart streaming memory_config")


# ---- nlsq.core.minpack.CurveFit --------------------------------------------


def _drive_cf_hybrid(ctx: SimpleNamespace, **kw: Any) -> Any:
    return nlsq.curve_fit(
        MODEL, X, Y, p0=P0, method="hybrid_streaming", verbose=0, **FAST_HYBRID, **kw
    )


def _drive_cf_hybrid_full(ctx: SimpleNamespace) -> Any:
    return _drive_cf_hybrid(
        ctx,
        sigma=SIG,
        absolute_sigma=True,
        bounds=FINITE_BOUNDS,
        callback=_CALLBACK,
    )


def _check_cf_hybrid_real(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_hybrid_ctor(
        rec,
        HybridStreamingConfig(**FAST_HYBRID),
        "curve_fit(hybrid_streaming) config from kwargs",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID), p0=np.array(P0), bounds=None, sigma=None, verbose=0
    )
    assert isinstance(result, CurveFitResult)
    close_to_truth(result.x)
    assert result["model"] is MODEL and np.asarray(result["pcov"]).shape == (2, 2)


def _check_cf_hybrid_full(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_hybrid_ctor(rec, HybridStreamingConfig(**FAST_HYBRID), "config from kwargs")
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
    )
    expect_curve_fit_result(result, x=HYB_X, pcov=HYB_PCOV, meta=HYB_META)


def _drive_cf_hybrid_config_object(ctx: SimpleNamespace) -> Any:
    ctx.config = HybridStreamingConfig(**FAST_HYBRID)
    ctx.before = snap(ctx.config)
    return nlsq.curve_fit(
        MODEL, X, Y, p0=P0, method="hybrid_streaming", config=ctx.config, verbose=0
    )


def _check_cf_hybrid_config_object(rec: Recorder, result: Any, ctx: Any) -> None:
    ctor = expect_hybrid_ctor(
        rec, HybridStreamingConfig(**FAST_HYBRID), "config= object"
    )
    assert ctor.named["config"] is ctx.config  # used as given
    expect_unchanged(ctx.config, ctx.before, "curve_fit(config=)")


def _check_cf_hybrid_eval(rec: Recorder, result: Any, ctx: Any) -> None:
    assert type(result) is tuple and len(result) == 3
    expect(result[0], HYB_X, "popt")
    expect(result[1], HYB_PCOV, "pcov")
    expect(result[2], np.asarray(MODEL(X, *HYB_X)), "f(xdata, *popt)")


def _check_cf_hybrid_full_output(rec: Recorder, result: Any, ctx: Any) -> None:
    assert type(result) is tuple and len(result) == 5
    expect(result[0], HYB_X, "popt")
    expect(result[1], HYB_PCOV, "pcov")
    expect(result[2], {"nfev": 42, "fvec": HYB_FUN}, "infodict from nfev and fun")
    assert result[3] == "hybrid-stub"
    assert result[4] == 1  # success -> ier 1


def _check_cf_hybrid_full_output_failed(rec: Recorder, result: Any, ctx: Any) -> None:
    assert type(result) is tuple and len(result) == 5
    assert result[4] == 5  # failure -> ier 5


def _drive_cf_auto_streaming(ctx: SimpleNamespace) -> Any:
    return nlsq.curve_fit(
        MODEL,
        X,
        Y,
        p0=P0,
        method="auto",
        memory_limit_gb=STREAMING_LIMIT_GB,
        sigma=SIG,
        absolute_sigma=True,
        bounds=FINITE_BOUNDS,
        callback=_CALLBACK,
        verbose=0,
    )


def _check_cf_auto_streaming(rec: Recorder, result: Any, ctx: Any) -> None:
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=STREAMING_LIMIT_GB
    )
    # The selector's config is handed through (not a fresh default).
    expect_hybrid_ctor(rec, expected, "method=auto streaming config")
    assert expected.chunk_size != HybridStreamingConfig().chunk_size
    expect_hybrid_fit(
        rec.only_fit(HYBRID),
        p0=np.array(P0),
        bounds=FINITE_BOUNDS_ARR,
        sigma=SIG,
        absolute_sigma=True,
        callback=_CALLBACK,
        verbose=0,
    )
    expect_curve_fit_result(result, x=HYB_X, pcov=HYB_PCOV, meta=HYB_META)


def _drive_cf_auto_chunked(ctx: SimpleNamespace, **kw: Any) -> Any:
    return nlsq.curve_fit(
        MODEL,
        X,
        Y,
        p0=P0,
        method="auto",
        memory_limit_gb=CHUNKED_LIMIT_GB,
        **kw,
    )


def _drive_cf_auto_chunked_full(ctx: SimpleNamespace) -> Any:
    return _drive_cf_auto_chunked(
        ctx, absolute_sigma=True, check_finite=False, bounds=FINITE_BOUNDS
    )


def _check_cf_auto_chunked(rec: Recorder, result: Any, ctx: Any) -> None:
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=CHUNKED_LIMIT_GB
    )
    # The raw memory_limit_gb kwarg (not the config's) is the constructor limit.
    assert expected.memory_limit_gb != CHUNKED_LIMIT_GB
    expect_fitter_ctor(rec, memory_limit_gb=CHUNKED_LIMIT_GB, config=expected)
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=P0,
        p0_is=True,
        bounds=FINITE_BOUNDS,
        bounds_is=True,
        method="trf",  # engine default; no method is forwarded
        extra={"check_finite": False, "absolute_sigma": True},
    )
    assert isinstance(result, CurveFitResult)
    expect_curve_fit_result(result, x=FIT_X, pcov=FIT_PCOV, meta=FIT_META)
    assert result["xdata"] is X and result["ydata"] is Y


def _check_cf_auto_chunked_defaults(rec: Recorder, result: Any, ctx: Any) -> None:
    expect_fitter_ctor(
        rec,
        memory_limit_gb=CHUNKED_LIMIT_GB,
        config=MemoryBudgetSelector().select(
            n_points=N, n_params=2, memory_limit_gb=CHUNKED_LIMIT_GB
        )[1],
    )
    expect_fitter_fit(
        rec.only_fit(FITTER),
        p0=P0,
        p0_is=True,
        bounds=INF_BOUNDS,
        method="trf",
        extra={"check_finite": True, "absolute_sigma": False},
    )


def _check_cf_auto_chunked_eval(rec: Recorder, result: Any, ctx: Any) -> None:
    assert type(result) is tuple and len(result) == 3
    assert result[0] is rec.last_result.x
    assert result[1] is rec.last_result.pcov
    expect(result[2], np.asarray(MODEL(X, *FIT_X)), "f(xdata, *popt)")


def _check_cf_auto_chunked_timeit(rec: Recorder, result: Any, ctx: Any) -> None:
    assert type(result) is tuple and len(result) == 5
    assert result[0] is rec.last_result.x
    assert result[1] is rec.last_result.pcov
    assert result[2] is rec.last_result  # the engine's result object
    assert result[3] == 0
    assert result[4] == 0.123  # engine's execution_time


def _check_cf_auto_chunked_real(rec: Recorder, result: Any, ctx: Any) -> None:
    assert len(rec.ctors()) == 1 and len(rec.ctors(FITTER)) == 1
    assert isinstance(result, CurveFitResult)
    close_to_truth(result.x)


ROWS: list[Row] = [
    # --- nlsq.fit (preset=streaming) --------------------------------------
    Row(
        "init.fit[preset=streaming]",
        "nlsq.fit",
        HYBRID,
        _drive_fit_preset_streaming_full,
        _check_fit_preset_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "init.fit[preset=streaming,multistart=False]",
        "nlsq.fit",
        HYBRID,
        lambda ctx: _drive_fit_preset_streaming(ctx, multistart=False),
        _check_fit_preset_streaming_off,
        {HYBRID: hybrid_ok},
    ),
    # --- nlsq.curve_fit_large ---------------------------------------------
    Row(
        "init.curve_fit_large[hybrid_streaming,real]",
        "nlsq.curve_fit_large",
        HYBRID,
        _drive_clf_hybrid,
        _check_clf_hybrid_real,
        marks=SLOW,
    ),
    Row(
        "init.curve_fit_large[hybrid_streaming,stub]",
        "nlsq.curve_fit_large",
        HYBRID,
        _drive_clf_hybrid_full,
        _check_clf_hybrid_stub,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "init.curve_fit_large[hybrid_streaming,no-pcov]",
        "nlsq.curve_fit_large",
        HYBRID,
        _drive_clf_hybrid,
        _check_clf_hybrid_no_pcov,
        {HYBRID: hybrid_no_pcov},
    ),
    Row(
        "init.curve_fit_large[chunked,real]",
        "nlsq.curve_fit_large",
        FITTER,
        lambda ctx: _drive_clf_chunked_pinned(ctx, memory_limit_gb=2.0),
        _check_clf_chunked_real,
        marks=SERIAL,
    ),
    Row(
        "init.curve_fit_large[chunked,args]",
        "nlsq.curve_fit_large",
        FITTER,
        _drive_clf_chunked_args,
        _check_clf_chunked_args,
        {FITTER: fitter_ok},
        marks=SERIAL,
    ),
    Row(
        "init.curve_fit_large[chunked,x-only-result]",
        "nlsq.curve_fit_large",
        FITTER,
        _drive_clf_chunked_x_only,
        _check_clf_chunked_x_only,
        {FITTER: fitter_x_only},
        marks=SERIAL,
    ),
    Row(
        "init.curve_fit_large[chunked,show_progress]",
        "nlsq.curve_fit_large",
        FITTER,
        _drive_clf_chunked_progress,
        _check_clf_chunked_progress,
        {FITTER: fitter_ok},
        marks=SERIAL,
    ),
    Row(
        "init.curve_fit_large[small-data,no-engine]",
        "nlsq.curve_fit_large (bypass)",
        None,
        lambda ctx: nlsq.curve_fit_large(MODEL, X, Y, p0=P0),
        _check_clf_small,
    ),
    # --- nlsq.streaming.large_dataset ---------------------------------------
    Row(
        "large_dataset.fit_large_dataset[real]",
        "fit_large_dataset",
        FITTER,
        _drive_fit_large_dataset,
        _check_fit_large_dataset_real,
    ),
    Row(
        "large_dataset.fit_large_dataset[args,stub]",
        "fit_large_dataset",
        FITTER,
        _drive_fit_large_dataset_stub,
        _check_fit_large_dataset_stub,
        {FITTER: fitter_ok},
    ),
    Row(
        "large_dataset.fit_large_dataset[progress,stub]",
        "fit_large_dataset",
        FITTER,
        _drive_fit_large_dataset_progress,
        _check_fit_large_dataset_progress,
        {FITTER: fitter_ok},
    ),
    Row(
        "large_dataset.LargeDatasetFitter._fit_with_streaming[unwired,direct]",
        "LargeDatasetFitter._fit_with_streaming",
        HYBRID,
        _drive_fitter_streaming,
        _check_fitter_streaming,
        {HYBRID: hybrid_ok},
        prebuilt=(FITTER,),
    ),
    Row(
        "large_dataset.LargeDatasetFitter._fit_with_streaming[engine-raises]",
        "LargeDatasetFitter._fit_with_streaming",
        HYBRID,
        _drive_fitter_streaming,
        _check_fitter_streaming_failure,
        {HYBRID: _raise_kaboom},
        prebuilt=(FITTER,),
    ),
    # --- minpack._fit_with_config (via nlsq.fit workflow=) ------------------
    Row(
        "minpack._fit_with_config[HybridStreamingConfig,gtol]",
        "_fit_with_config",
        HYBRID,
        _drive_workflow_hybrid_full,
        _check_workflow_hybrid_config,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_config[HybridStreamingConfig,goal]",
        "_fit_with_config",
        HYBRID,
        lambda ctx: _drive_workflow_hybrid_config(ctx, goal="fast"),
        _check_workflow_hybrid_goal,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_config[HybridStreamingConfig,no-gtol,no-pcov]",
        "_fit_with_config",
        HYBRID,
        _drive_workflow_hybrid_config,
        _check_workflow_hybrid_no_gtol,
        {HYBRID: hybrid_no_pcov},
    ),
    Row(
        "minpack._fit_with_config[HybridStreamingConfig,real]",
        "_fit_with_config",
        HYBRID,
        lambda ctx: _drive_workflow_hybrid_config(ctx, p0=P0, absolute_sigma=True),
        _check_workflow_hybrid_real,
        marks=SLOW,
    ),
    Row(
        "minpack._fit_with_config[LDMemoryConfig,args]",
        "_fit_with_config",
        FITTER,
        _drive_workflow_ld_full,
        _check_workflow_ld_config,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_with_config[LDMemoryConfig,curvefitresult-passthrough]",
        "_fit_with_config",
        FITTER,
        _drive_workflow_ld_config,
        _check_workflow_ld_passthrough,
        {FITTER: fitter_curve_fit_result},
    ),
    Row(
        "minpack._fit_with_config[LDMemoryConfig,real]",
        "_fit_with_config",
        FITTER,
        _drive_workflow_ld_config,
        _check_workflow_ld_config_real,
    ),
    Row(
        "minpack._fit_with_config[workflow=auto->chunked]",
        "_fit_with_config",
        FITTER,
        lambda ctx: nlsq.fit(
            MODEL,
            X,
            Y,
            p0=P0,
            workflow="auto",
            memory_limit_gb=CHUNKED_LIMIT_GB,
            gtol=1e-3,
        ),
        _check_auto_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_with_config[workflow=auto->streaming]",
        "_fit_with_config",
        HYBRID,
        _drive_auto_streaming,
        _check_auto_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_config[workflow=auto->standard,no-engine]",
        "_fit_with_config (fallthrough)",
        None,
        lambda ctx: nlsq.fit(MODEL, X, Y, p0=P0, workflow="auto"),
        _check_auto_standard,
    ),
    # --- minpack._fit_with_preset (direct helper only) ----------------------
    Row(
        "minpack._fit_with_preset[large_robust,n_points=1M,direct]",
        "_fit_with_preset (chunked tier)",
        FITTER,
        _drive_preset_chunked,
        _check_preset_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_with_preset[large_robust,curvefitresult-passthrough]",
        "_fit_with_preset (chunked tier)",
        FITTER,
        _drive_preset_chunked_passthrough,
        _check_workflow_ld_passthrough,
        {FITTER: fitter_curve_fit_result},
    ),
    Row(
        "minpack._fit_with_preset[streaming,direct]",
        "_fit_with_preset (streaming tier)",
        HYBRID,
        _drive_preset_streaming,
        _check_preset_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_preset[hpc_distributed,direct]",
        "_fit_with_preset (streaming tier)",
        HYBRID,
        _drive_preset_hpc,
        _check_preset_hpc,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_preset[streaming,goal-overrides-preset-gtol]",
        "_fit_with_preset (streaming tier)",
        HYBRID,
        lambda ctx: _call_preset("streaming", goal=OptimizationGoal.QUALITY),
        _check_preset_goal_tolerance,
        {HYBRID: hybrid_ok},
    ),
    # --- minpack._fit_global_multistart -------------------------------------
    Row(
        "minpack._fit_global_multistart[auto_global->chunked]",
        "_fit_global_multistart (chunked)",
        FITTER,
        _drive_global_chunked,
        _check_global_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_global_multistart[auto_global->streaming]",
        "_fit_global_multistart (streaming)",
        HYBRID,
        _drive_global_streaming,
        _check_global_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_global_multistart[chunked,no-memory_config,direct]",
        "_fit_global_multistart (chunked)",
        FITTER,
        _drive_multistart_default_chunked,
        _check_multistart_default_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_global_multistart[chunked,given-memory_config,direct]",
        "_fit_global_multistart (chunked)",
        FITTER,
        _drive_multistart_given_chunked,
        _check_multistart_given_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_global_multistart[streaming,no-memory_config,direct]",
        "_fit_global_multistart (streaming)",
        HYBRID,
        _drive_multistart_default_streaming,
        _check_multistart_default_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_global_multistart[streaming,given-memory_config,direct]",
        "_fit_global_multistart (streaming)",
        HYBRID,
        _drive_multistart_given_streaming,
        _check_multistart_given_streaming,
        {HYBRID: hybrid_ok},
    ),
    # --- minpack.CurveFit ----------------------------------------------------
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[real]",
        "_curve_fit_hybrid_streaming",
        HYBRID,
        _drive_cf_hybrid,
        _check_cf_hybrid_real,
        marks=SLOW,
    ),
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[args]",
        "_curve_fit_hybrid_streaming",
        HYBRID,
        _drive_cf_hybrid_full,
        _check_cf_hybrid_full,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[config-object]",
        "_curve_fit_hybrid_streaming",
        HYBRID,
        _drive_cf_hybrid_config_object,
        _check_cf_hybrid_config_object,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[return_eval]",
        "_curve_fit_hybrid_streaming",
        HYBRID,
        lambda ctx: _drive_cf_hybrid(ctx, return_eval=True),
        _check_cf_hybrid_eval,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[full_output]",
        "_curve_fit_hybrid_streaming",
        HYBRID,
        lambda ctx: _drive_cf_hybrid(ctx, full_output=True),
        _check_cf_hybrid_full_output,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[full_output,failed]",
        "_curve_fit_hybrid_streaming",
        HYBRID,
        lambda ctx: _drive_cf_hybrid(ctx, full_output=True),
        _check_cf_hybrid_full_output_failed,
        {HYBRID: hybrid_failed},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[streaming]",
        "_curve_fit_hybrid_streaming via _curve_fit_auto_memory",
        HYBRID,
        _drive_cf_auto_streaming,
        _check_cf_auto_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked]",
        "_curve_fit_auto_memory (chunked)",
        FITTER,
        _drive_cf_auto_chunked_full,
        _check_cf_auto_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked,defaults]",
        "_curve_fit_auto_memory (chunked)",
        FITTER,
        _drive_cf_auto_chunked,
        _check_cf_auto_chunked_defaults,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked,return_eval]",
        "_curve_fit_auto_memory (chunked)",
        FITTER,
        lambda ctx: _drive_cf_auto_chunked(ctx, return_eval=True),
        _check_cf_auto_chunked_eval,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked,timeit]",
        "_curve_fit_auto_memory (chunked)",
        FITTER,
        lambda ctx: _drive_cf_auto_chunked(ctx, timeit=True),
        _check_cf_auto_chunked_timeit,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked,real]",
        "_curve_fit_auto_memory (chunked)",
        FITTER,
        _drive_cf_auto_chunked,
        _check_cf_auto_chunked_real,
    ),
]


class _Ctx(SimpleNamespace):
    """Per-row scratch: configs the caller builds, plus their pre-call snapshot."""

    def before_config(self) -> Any:
        """A fresh config equal to the caller's config as it was before the call."""
        return type(self.config)(**self.before)


@pytest.mark.parametrize(
    "row",
    [pytest.param(r, id=r.id, marks=r.marks) for r in ROWS],
)
def test_routing_table(row: Row, rec: Recorder) -> None:
    rec.stubs.update(row.stubs)
    ambient_before = dataclasses.asdict(get_memory_config())
    ctx = _Ctx()
    with np.errstate(all="ignore"):
        result = row.drive(ctx)
    # Exactly one construction, of exactly the expected class (none for bypass rows);
    # this is what catches a class swap or a class built twice.
    built = [c.engine for c in rec.ctors()]
    expected = [*row.prebuilt, *([row.engine] if row.engine else [])]
    assert built == expected, f"constructed {built}, expected {expected}"
    row.check(rec, result, ctx)
    # No global memory config leaked out of the entry point.
    expect(dataclasses.asdict(get_memory_config()), ambient_before, "ambient config")


# --------------------------------------------------------------------------
# Behaviour around the rows: rejections, failures, negative routing
# --------------------------------------------------------------------------


def test_legacy_preset_keys_rejected_publicly(rec: Recorder) -> None:
    """Every WORKFLOW_PRESETS key is rejected before _fit_with_preset can run.

    So the two constructors inside _fit_with_preset have no public path; the
    rows above drive them through the helper directly. (If
    _fit_with_preset is restructured, this behavioural gate must keep holding.)
    """
    assert set(minpack.WORKFLOW_PRESETS) == set(minpack.REMOVED_PRESETS)
    for key in minpack.WORKFLOW_PRESETS:
        for spelling in (key, key.upper()):
            with pytest.raises(ValueError, match=r"removed in v0\.6\.3"):
                nlsq.fit(MODEL, X, Y, p0=P0, workflow=spelling, bounds=FINITE_BOUNDS)
            with pytest.raises(ValueError, match=r"removed in v0\.6\.3"):
                minpack.fit(MODEL, X, Y, p0=P0, workflow=spelling)
    assert rec.calls == []


def test_fitter_fit_never_builds_hybrid(rec: Recorder) -> None:
    """LargeDatasetFitter.fit (use_streaming defaults to True) never routes to
    the hybrid optimizer, so _fit_with_streaming stays unreachable."""
    assert LDMemoryConfig().use_streaming is True
    fitter = _REAL[FITTER](config=LDMemoryConfig(memory_limit_gb=2.0))
    result = fitter.fit(MODEL, X, Y, p0=P0)
    assert result.success
    assert [c.engine for c in rec.ctors()] == [FITTER]
    close_to_truth(result.x)


def test_hybrid_routes_reject_unsupported_kwargs(rec: Recorder) -> None:
    """curve_fit_large / curve_fit hybrid routes accept only config fields:
    tolerances and callback (on curve_fit_large) are rejected, never forwarded."""
    with pytest.raises(TypeError, match="Unrecognized keyword"):
        nlsq.curve_fit_large(MODEL, X, Y, p0=P0, method="hybrid_streaming", gtol=1e-3)
    with pytest.raises(TypeError, match="Unrecognized keyword"):
        nlsq.curve_fit_large(
            MODEL, X, Y, p0=P0, method="hybrid_streaming", callback=_CALLBACK
        )
    with pytest.raises(TypeError, match="Unrecognized keyword"):
        nlsq.curve_fit(MODEL, X, Y, p0=P0, method="hybrid_streaming", gtol=1e-3)
    assert rec.ctors() == []


def test_hybrid_routes_reject_nonfinite_data_before_construction(
    rec: Recorder,
) -> None:
    bad = Y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        nlsq.fit(
            MODEL,
            X,
            bad,
            p0=P0,
            workflow=HybridStreamingConfig(**FAST_HYBRID),
            verbose=0,
        )
    assert rec.ctors() == []


def test_auto_global_chunked_failure_and_sigma_guard(rec: Recorder) -> None:
    rec.stubs[FITTER] = fitter_failed
    with pytest.raises(RuntimeError, match="chunked multi-start fit failed: boom"):
        _drive_auto_global(CHUNKED_LIMIT_GB)
    with pytest.raises(RuntimeError, match=r"workflow='hpc' chunked multi-start"):
        nlsq.fit(
            MODEL,
            X,
            Y,
            p0=P0,
            workflow="hpc",
            bounds=FINITE_BOUNDS,
            method="multi-start",
            memory_limit_gb=CHUNKED_LIMIT_GB,
        )
    assert [c.engine for c in rec.ctors()] == [FITTER, FITTER]
    rec.calls.clear()
    with pytest.raises(NotImplementedError, match="sigma"):
        _drive_auto_global(CHUNKED_LIMIT_GB, sigma=SIG)
    assert rec.ctors() == []


def test_curve_fit_auto_chunked_failure_and_guards(rec: Recorder) -> None:
    rec.stubs[FITTER] = fitter_failed
    with pytest.raises(RuntimeError, match="method='auto' chunked fit failed: boom"):
        nlsq.curve_fit(
            MODEL, X, Y, p0=P0, method="auto", memory_limit_gb=CHUNKED_LIMIT_GB
        )
    rec.calls.clear()
    with pytest.raises(NotImplementedError, match="sigma"):
        nlsq.curve_fit(
            MODEL,
            X,
            Y,
            p0=P0,
            method="auto",
            memory_limit_gb=CHUNKED_LIMIT_GB,
            sigma=SIG,
        )
    with pytest.raises(NotImplementedError, match="data_mask"):
        nlsq.curve_fit(
            MODEL,
            X,
            Y,
            p0=P0,
            method="hybrid_streaming",
            data_mask=np.ones(N, bool),
        )
    assert rec.ctors() == []


def test_preset_chunked_below_1m_points_uses_curve_fit(
    rec: Recorder, monkeypatch
) -> None:
    """Below 1M points the 'chunked' preset tier never builds the chunked engine."""
    seen: list[dict[str, Any]] = []

    def fake_curve_fit(**kwargs: Any) -> str:
        seen.append(kwargs)
        return "curve_fit-result"

    monkeypatch.setattr(minpack, "curve_fit", fake_curve_fit)
    assert _call_preset("large_robust", n_points=999_999) == "curve_fit-result"
    assert rec.ctors() == []
    (kwargs,) = seen
    assert kwargs["multistart"] is True and kwargs["n_starts"] == 10
    assert kwargs["gtol"] == kwargs["ftol"] == kwargs["xtol"] == 1e-8


def test_multistart_standard_strategy_uses_curve_fit(
    rec: Recorder, monkeypatch
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        minpack, "curve_fit", lambda **kw: seen.append(kw) or "curve_fit-result"
    )
    assert _call_multistart("standard") == "curve_fit-result"
    assert rec.ctors() == []
    assert seen[0]["multistart"] is True and seen[0]["n_starts"] == 3


def test_multistart_wrong_typed_memory_config_is_replaced(rec: Recorder) -> None:
    """A memory_config of the other engine's type is ignored, not reused or mutated."""
    rec.stubs[FITTER] = fitter_ok
    rec.stubs[HYBRID] = hybrid_ok
    wrong_for_chunked = HybridStreamingConfig(chunk_size=17)
    wrong_for_streaming = LDMemoryConfig(memory_limit_gb=1.0, min_chunk_size=5)
    before = (snap(wrong_for_chunked), snap(wrong_for_streaming))
    _call_multistart("chunked", memory_config=wrong_for_chunked)
    expect_fitter_ctor(
        rec,
        memory_limit_gb=8.0,
        config=LDMemoryConfig(
            memory_limit_gb=8.0, min_chunk_size=1000, max_chunk_size=1_000_000
        ),
    )
    _call_multistart("streaming", memory_config=wrong_for_streaming)
    assert rec.ctors(HYBRID)[0].configs["config"].chunk_size == (
        HybridStreamingConfig().chunk_size
    )
    expect_unchanged(wrong_for_chunked, before[0], "wrong-typed chunked memory_config")
    expect_unchanged(
        wrong_for_streaming, before[1], "wrong-typed streaming memory_config"
    )
