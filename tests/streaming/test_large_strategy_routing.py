"""Routing table for the two large-dataset engines (Part D, task D1).

Pins, for each of the 13 places that construct ``AdaptiveHybridStreamingOptimizer``
(the "hybrid" engine) or ``LargeDatasetFitter`` (the "chunked" engine), what a
given (entry point, kwargs) builds: which class, the constructor arguments that
matter (config values, forwarded or dropped tolerances), how the engine's fit
method is called, and what the entry point returns. It is the safety net for
moving those construction sites behind ``build_large_strategy``: the rows must
keep passing unchanged.

Recording patches ``__init__``/``fit``/``fit_with_progress`` on the two real
classes (not module attributes), so it sees construction wherever the classes
are imported from. Arguments are normalised through the real signatures, so a
positional-vs-keyword change at a call site does not matter, but an added,
dropped or changed value does. Rows that would run a long fit stub the engine's
fit with a recognisable sentinel; the ``*_real`` rows run the real fit.

Public vs direct: every ``WORKFLOW_PRESETS`` key is rejected before
``_fit_with_preset`` (see ``test_legacy_preset_keys_rejected_publicly``), so its
two constructors are driven through the module-level helper directly.
``LargeDatasetFitter._fit_with_streaming`` has no caller anywhere; it is driven
directly and ``test_fitter_fit_never_builds_hybrid`` pins that it stays unwired.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
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

SENT_X = np.array([7.0, 8.0])
SENT_PCOV = 3.0 * np.eye(2)


def MODEL(x, a, b):
    return a * jnp.exp(-b * x)


# --------------------------------------------------------------------------
# Recording harness
# --------------------------------------------------------------------------


@dataclass
class Call:
    engine: str
    method: str
    named: dict[str, Any]
    extra: dict[str, Any]  # the **kwargs the call site forwarded


@dataclass
class Recorder:
    calls: list[Call] = field(default_factory=list)
    stubs: dict[str, Callable[[], Any]] = field(default_factory=dict)

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
        rec.calls.append(Call(engine, method, named, extra))
        if method != "__init__" and engine in rec.stubs:
            return rec.stubs[engine]()
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
    return {"x": SENT_X.copy(), "success": True, "message": "stub", "pcov": SENT_PCOV}


def hybrid_no_pcov() -> dict[str, Any]:
    return {"x": SENT_X.copy(), "success": True, "message": "stub"}


def fitter_ok() -> OptimizeResult:
    return OptimizeResult(
        x=SENT_X.copy(),
        popt=SENT_X.copy(),
        pcov=SENT_PCOV,
        success=True,
        message="stub",
    )


def fitter_x_only() -> OptimizeResult:
    return OptimizeResult(x=SENT_X.copy(), success=True, message="stub")


def fitter_failed() -> OptimizeResult:
    return OptimizeResult(x=SENT_X.copy(), success=False, message="boom")


def same(a: Any, b: Any) -> bool:
    """Equality that understands arrays, tuples and dicts (and identity of None)."""
    if a is None or b is None:
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(i, j) for i, j in zip(a, b, strict=True))
    if hasattr(a, "shape") or hasattr(b, "shape"):
        return np.shape(a) == np.shape(b) and bool(
            np.array_equal(np.asarray(a), np.asarray(b))
        )
    return bool(a == b)


def _short(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_short(v)}" for k, v in value.items()) + "}"
    if hasattr(value, "shape") and np.size(value) > 6:
        return f"<array shape={np.shape(value)} first={np.ravel(value)[:3]}>"
    return repr(value)


def expect(actual: Any, expected: Any, what: str) -> None:
    assert same(actual, expected), (
        f"{what}: got {_short(actual)}, expected {_short(expected)}"
    )


def expect_config(actual: Any, expected: Any, what: str) -> None:
    """Whole-config equality: any value changed, added or dropped fails."""
    assert type(actual) is type(expected), f"{what}: {type(actual)} vs {type(expected)}"
    expect(dataclasses.asdict(actual), dataclasses.asdict(expected), what)


def expect_hybrid_fit(
    call: Call,
    *,
    p0: Any,
    bounds: Any,
    sigma: Any,
    absolute_sigma: bool = False,
    callback: Any = None,
    verbose: int = 1,
) -> None:
    n = call.named
    assert isinstance(n["data_source"], tuple) and len(n["data_source"]) == 2
    expect(n["data_source"][0], X, "hybrid data_source x")
    expect(n["data_source"][1], Y, "hybrid data_source y")
    assert n["func"] is MODEL
    expect(n["p0"], p0, "hybrid p0")
    expect(n["bounds"], bounds, "hybrid bounds")
    expect(n["sigma"], sigma, "hybrid sigma")
    assert n["absolute_sigma"] is absolute_sigma
    assert n["callback"] is callback
    assert n["verbose"] == verbose
    assert call.extra == {}


def expect_curve_fit_result(result: Any, *, pcov: Any) -> None:
    """The hand-off shape of the _fit_with_* helpers: result + model/xdata/ydata."""
    assert isinstance(result, CurveFitResult)
    assert result["model"] is MODEL
    expect(result["xdata"], X, "result xdata")
    expect(result["ydata"], Y, "result ydata")
    expect(result["x"], SENT_X, "result x")
    expect(result["pcov"], pcov, "result pcov")


def close_to_truth(popt: Any) -> None:
    np.testing.assert_allclose(np.asarray(popt), TRUE, atol=0.05)


# --------------------------------------------------------------------------
# The routing table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    id: str
    site: str  # construction site this row exercises (function; see report)
    engine: str | None  # class that must be built exactly once; None = neither
    drive: Callable[[], Any]
    check: Callable[[Recorder, Any], None]
    stubs: dict[str, Callable[[], Any]] = field(default_factory=dict)
    # Constructions the drive function itself performs before the routed one.
    prebuilt: tuple[str, ...] = ()


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


# ---- nlsq/__init__.py::fit (preset="streaming") --------------------------


def _drive_fit_preset_streaming(**kw: Any) -> Any:
    with pytest.warns(FutureWarning, match="preset"):
        return nlsq.fit(MODEL, X, Y, p0=P0, preset="streaming", **kw)


def _check_fit_preset_streaming(rec: Recorder, result: Any) -> None:
    # gtol/ftol are dropped on this route (no config field is set from them);
    # only the multistart settings reach the config.
    expect_config(
        rec.ctors(HYBRID)[0].named["config"],
        HybridStreamingConfig(
            enable_multistart=True, n_starts=4, multistart_sampler="sobol"
        ),
        "fit(preset=streaming) config",
    )
    expect_hybrid_fit(rec.only_fit(HYBRID), p0=np.array(P0), bounds=None, sigma=SIG)
    assert isinstance(result, CurveFitResult)
    expect(result["x"], SENT_X, "x")
    expect(result["pcov"], SENT_PCOV, "pcov")
    assert "model" not in result  # this route does not attach model/xdata/ydata
    expect(
        result["multistart_diagnostics"],
        {"n_starts_configured": 4, "bypassed": False, "preset": "streaming"},
        "multistart_diagnostics",
    )


def _check_fit_preset_streaming_off(rec: Recorder, result: Any) -> None:
    expect_config(
        rec.ctors(HYBRID)[0].named["config"],
        HybridStreamingConfig(enable_multistart=False, n_starts=10),
        "multistart=False config",
    )
    expect(
        result["multistart_diagnostics"],
        {"n_starts_configured": 10, "bypassed": True, "preset": "streaming"},
        "multistart_diagnostics",
    )


# ---- nlsq/__init__.py::curve_fit_large ------------------------------------


def _drive_clf_hybrid(**kw: Any) -> Any:
    return nlsq.curve_fit_large(
        MODEL, X, Y, p0=P0, method="hybrid_streaming", verbose=0, **FAST_HYBRID, **kw
    )


def _check_clf_hybrid_real(rec: Recorder, result: Any) -> None:
    expect_config(
        rec.ctors(HYBRID)[0].named["config"],
        HybridStreamingConfig(**FAST_HYBRID),
        "curve_fit_large(hybrid_streaming) config",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID), p0=np.array(P0), bounds=None, sigma=None, verbose=0
    )
    # A (popt, pcov) tuple of arrays, not a result object.
    assert type(result) is tuple and len(result) == 2
    close_to_truth(result[0])
    assert np.asarray(result[1]).shape == (2, 2) and np.all(np.isfinite(result[1]))


def _check_clf_hybrid_stub(rec: Recorder, result: Any) -> None:
    assert type(result) is tuple and len(result) == 2
    expect(result[0], SENT_X, "popt")
    expect(result[1], SENT_PCOV, "pcov")


def _check_clf_hybrid_no_pcov(rec: Recorder, result: Any) -> None:
    expect(result[0], SENT_X, "popt")
    expect(result[1], np.full((2, 2), np.inf), "pcov fallback is all-inf")


def _drive_clf_chunked(**kw: Any) -> Any:
    return nlsq.curve_fit_large(MODEL, X, Y, p0=P0, size_threshold=100, **kw)


def _check_clf_chunked_real(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    # The constructor gets the limit and a config only: multistart & co. go to fit().
    assert ctor.named["memory_limit_gb"] == 2.0
    expect_config(
        ctor.named["config"],
        LDMemoryConfig(memory_limit_gb=2.0, min_chunk_size=600, max_chunk_size=600),
        "curve_fit_large chunked config",
    )
    assert ctor.named["multistart"] is False and ctor.named["n_starts"] == 10
    assert ctor.named["logger"] is None
    call = rec.only_fit(FITTER)
    expect(call.named["p0"], np.array(P0), "p0")
    expect(call.named["bounds"], INF_BOUNDS, "bounds")
    assert call.named["method"] == "trf"
    expect(
        call.extra,
        {
            "absolute_sigma": False,
            "check_finite": True,
            "multistart": False,
            "n_starts": 10,
            "sampler": "lhs",
            "center_on_p0": True,
            "scale_factor": 1.0,
        },
        "forwarded fit kwargs",
    )
    assert type(result) is tuple and len(result) == 2
    close_to_truth(result[0])
    assert np.asarray(result[1]).shape == (2, 2)


def _check_clf_chunked_args(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    expect_config(
        ctor.named["config"],
        LDMemoryConfig(memory_limit_gb=3.0, min_chunk_size=300, max_chunk_size=300),
        "chunk_size=300 config",
    )
    call = rec.only_fit(FITTER)
    expect(call.named["p0"], np.array(P0), "p0")
    expect(call.named["bounds"], FINITE_BOUNDS, "bounds")
    assert call.named["method"] == "lm"
    expect(call.extra["sigma"], SIG, "sigma forwarded")
    assert call.extra["absolute_sigma"] is True
    assert call.extra["multistart"] is True and call.extra["n_starts"] == 5
    assert call.extra["sampler"] == "sobol"
    assert call.extra["gtol"] == 1e-3  # solver kwargs pass through untouched
    expect(result[0], SENT_X, "popt from result.popt")
    expect(result[1], SENT_PCOV, "pcov from result.pcov")


def _check_clf_chunked_x_only(rec: Recorder, result: Any) -> None:
    expect(result[0], SENT_X, "popt from result.x")
    expect(result[1], np.eye(2), "identity pcov when the result has none")


def _check_clf_chunked_progress(rec: Recorder, result: Any) -> None:
    assert rec.fits(FITTER, "fit") == []
    call = rec.only_fit(FITTER, "fit_with_progress")
    assert call.named["method"] == "trf"
    expect(result[0], SENT_X, "popt")


def _check_clf_small(rec: Recorder, result: Any) -> None:
    # Below size_threshold the engine is bypassed entirely.
    assert rec.calls == []
    assert isinstance(result, CurveFitResult)
    close_to_truth(result.x)


# ---- nlsq/streaming/large_dataset.py --------------------------------------


def _drive_fit_large_dataset(**kw: Any) -> Any:
    return large_dataset.fit_large_dataset(
        MODEL, X, Y, p0=P0, memory_limit_gb=2.0, multistart=False, n_starts=3, **kw
    )


def _check_fit_large_dataset_real(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    # Constructed from loose limits, no config object; multistart is a ctor arg here.
    expect(
        ctor.named,
        {
            "memory_limit_gb": 2.0,
            "config": None,
            "curve_fit_class": None,
            "logger": None,
            "multistart": False,
            "n_starts": 3,
            "sampler": "lhs",
        },
        "fit_large_dataset ctor args",
    )
    call = rec.only_fit(FITTER)
    expect(call.named["p0"], P0, "p0")
    assert call.extra == {}
    assert isinstance(result, OptimizeResult) and result.success
    close_to_truth(result.x)


def _check_fit_large_dataset_progress(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    assert ctor.named["multistart"] is True and ctor.named["n_starts"] == 3
    assert ctor.named["sampler"] == "halton"
    assert ctor.named["logger"] is _LOGGER
    assert rec.fits(FITTER, "fit") == []
    call = rec.only_fit(FITTER, "fit_with_progress")
    expect(call.named["bounds"], FINITE_BOUNDS, "bounds passthrough")
    assert isinstance(result, OptimizeResult)
    expect(result.x, SENT_X, "result is the engine's, unchanged")


def _drive_fitter_streaming(**kw: Any) -> Any:
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
        kw.get("p0", [2.0, 1.0]),
        kw.get("bounds", FINITE_BOUNDS),
        "trf",
        "auto",
        kw.get("show_progress", False),
        sigma=SIG,  # accepted via **kwargs and silently unused
    )


def _check_fitter_streaming(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(HYBRID)
    expect_config(
        ctor.named["config"],
        HybridStreamingConfig(chunk_size=123, gauss_newton_max_iterations=7),
        "_fit_with_streaming config",
    )
    # sigma is not forwarded; bounds are passed through unprepared; verbose is 1.
    expect_hybrid_fit(
        rec.only_fit(HYBRID), p0=np.array(P0), bounds=FINITE_BOUNDS, sigma=None
    )
    assert isinstance(result, OptimizeResult)
    expect(result.x, SENT_X, "x")
    expect(result.popt, SENT_X, "popt")
    expect(result.pcov, SENT_PCOV, "pcov")
    assert result.success is True and result.nfev == N


def _check_fitter_streaming_failure(rec: Recorder, result: Any) -> None:
    assert isinstance(result, OptimizeResult) and result.success is False
    assert "Streaming fit failed: kaboom" in result.message
    expect(result.x, np.array(P0), "falls back to p0")
    expect(result.pcov, np.full((2, 2), np.inf), "pcov is all-inf, not identity")


def _raise_kaboom() -> Any:
    raise RuntimeError("kaboom")


# ---- nlsq/core/minpack.py::_fit_with_config -------------------------------


def _drive_workflow_hybrid_config(**kw: Any) -> Any:
    config = HybridStreamingConfig(**FAST_HYBRID)
    return nlsq.fit(MODEL, X, Y, workflow=config, verbose=0, **kw)


def _check_workflow_hybrid_config(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(HYBRID)
    # gtol is mapped into gauss_newton_tol on a *copy*; ftol/xtol have no target.
    expect_config(
        ctor.named["config"],
        HybridStreamingConfig(gauss_newton_tol=1e-3, **FAST_HYBRID),
        "workflow=HybridStreamingConfig config",
    )
    call = rec.only_fit(HYBRID)
    expect_hybrid_fit(call, p0=np.ones(2), bounds=None, sigma=None, verbose=0)
    expect_curve_fit_result(result, pcov=SENT_PCOV)


def _check_workflow_hybrid_goal(rec: Recorder, result: Any) -> None:
    gtol = calculate_adaptive_tolerances(N, OptimizationGoal.FAST)["gtol"]
    expect_config(
        rec.ctors(HYBRID)[0].named["config"],
        HybridStreamingConfig(gauss_newton_tol=gtol, **FAST_HYBRID),
        "goal-derived gtol reaches gauss_newton_tol",
    )


def _check_workflow_hybrid_no_gtol(rec: Recorder, result: Any) -> None:
    expect_config(
        rec.ctors(HYBRID)[0].named["config"],
        HybridStreamingConfig(**FAST_HYBRID),
        "no gtol -> config untouched",
    )
    expect_curve_fit_result(result, pcov=np.full((2, 2), np.inf))


LD_WORKFLOW_CONFIG = LDMemoryConfig(
    memory_limit_gb=2.0, min_chunk_size=100, max_chunk_size=300
)


def _drive_workflow_ld_config(**kw: Any) -> Any:
    return nlsq.fit(MODEL, X, Y, p0=P0, workflow=LD_WORKFLOW_CONFIG, **kw)


def _check_workflow_ld_config(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    assert ctor.named["config"] is LD_WORKFLOW_CONFIG  # used as given, not copied
    assert ctor.named["memory_limit_gb"] == 2.0
    assert ctor.named["multistart"] is False and ctor.named["logger"] is None
    call = rec.only_fit(FITTER)
    expect(call.named["p0"], np.array(P0), "p0")
    expect(call.named["bounds"], FINITE_BOUNDS, "bounds")
    assert call.named["method"] == "trf"  # method or "trf"
    # sigma/absolute_sigma/check_finite and all three tolerances are forwarded
    # (unlike the hybrid route). n_starts=None / memory_limit_gb=None are
    # nlsq.fit's own passthrough of its named parameters into minpack.fit's
    # **kwargs, which _fit_with_config then forwards to the engine as-is.
    expect(
        call.extra,
        {
            "sigma": SIG,
            "absolute_sigma": True,
            "check_finite": True,
            "n_starts": None,
            "memory_limit_gb": None,
            "gtol": 1e-3,
            "ftol": 1e-4,
            "xtol": 1e-5,
        },
        "forwarded fit kwargs",
    )
    # Plain OptimizeResult is wrapped and gets model/xdata/ydata attached.
    expect_curve_fit_result(result, pcov=SENT_PCOV)


def _check_workflow_ld_config_real(rec: Recorder, result: Any) -> None:
    assert len(rec.ctors()) == 1 and len(rec.ctors(FITTER)) == 1
    assert isinstance(result, CurveFitResult)
    close_to_truth(result["x"])
    assert result["model"] is MODEL


def _check_auto_chunked(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=CHUNKED_LIMIT_GB
    )
    assert isinstance(expected, LDMemoryConfig)
    expect_config(ctor.named["config"], expected, "selector's chunked config")
    assert ctor.named["memory_limit_gb"] == expected.memory_limit_gb
    call = rec.only_fit(FITTER)
    assert call.named["method"] == "trf"
    assert call.extra["gtol"] == 1e-3 and call.extra["n_starts"] is None
    assert "memory_limit_gb" not in call.extra  # consumed by the selector
    expect_curve_fit_result(result, pcov=SENT_PCOV)


def _check_auto_streaming(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(HYBRID)
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=STREAMING_LIMIT_GB
    )
    assert isinstance(expected, HybridStreamingConfig)
    expect_config(
        ctor.named["config"],
        dataclasses.replace(expected, gauss_newton_tol=1e-3),
        "selector's streaming config + gtol",
    )
    expect_hybrid_fit(rec.only_fit(HYBRID), p0=np.array(P0), bounds=None, sigma=None)
    expect_curve_fit_result(result, pcov=SENT_PCOV)


def _check_auto_standard(rec: Recorder, result: Any) -> None:
    assert rec.calls == []  # data fits: neither engine is built
    assert isinstance(result, CurveFitResult)
    close_to_truth(result["popt"] if "popt" in result else result["x"])


# ---- nlsq/core/minpack.py::_fit_with_preset (direct helper) ---------------


def _check_preset_chunked(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    expect_config(
        ctor.named["config"],
        LDMemoryConfig(
            memory_limit_gb=8.0, min_chunk_size=1000, max_chunk_size=100_000
        ),
        "large_robust config at n_points=1M",
    )
    assert ctor.named["memory_limit_gb"] == 8.0
    call = rec.only_fit(FITTER)
    expect(call.named["p0"], np.array(P0), "p0")
    assert call.named["method"] == "trf"
    # sigma, absolute_sigma and check_finite are dropped on this route; the preset's
    # multistart settings and tolerances are forwarded.
    expect(
        call.extra,
        {
            "multistart": True,
            "n_starts": 10,
            "gtol": 1e-8,
            "ftol": 1e-8,
            "xtol": 1e-8,
        },
        "forwarded fit kwargs",
    )
    expect_curve_fit_result(result, pcov=SENT_PCOV)


def _check_preset_streaming(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(HYBRID)
    expect_config(
        ctor.named["config"],
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
        bounds=None,
        sigma=None,
        callback=_CALLBACK,
        verbose=0,
    )
    expect_curve_fit_result(result, pcov=SENT_PCOV)


def _check_preset_hpc(rec: Recorder, result: Any) -> None:
    expect_config(
        rec.ctors(HYBRID)[0].named["config"],
        HybridStreamingConfig(
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=1e-6,
            enable_checkpoints=True,
            enable_multistart=True,
            n_starts=10,
        ),
        "hpc_distributed preset config",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID), p0=np.array(P0), bounds=FINITE_BOUNDS_ARR, sigma=SIG
    )


def _check_preset_goal_tolerance(rec: Recorder, result: Any) -> None:
    gtol = calculate_adaptive_tolerances(N, OptimizationGoal.QUALITY)["gtol"]
    assert gtol != 1e-7
    assert rec.ctors(HYBRID)[0].named["config"].gauss_newton_tol == gtol


# ---- nlsq/core/minpack.py::_fit_global_multistart -------------------------


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


def _check_global_chunked(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=CHUNKED_LIMIT_GB
    )
    expect_config(ctor.named["config"], expected, "selector's config used as-is")
    assert ctor.named["memory_limit_gb"] == expected.memory_limit_gb
    call = rec.only_fit(FITTER)
    expect(call.named["p0"], np.array(P0), "p0")
    expect(call.named["bounds"], FINITE_BOUNDS_ARR, "bounds")
    assert call.named["method"] == "trf"
    # multistart/n_starts go to fit(), not to the constructor.
    assert ctor.named["multistart"] is False
    expect(
        call.extra,
        {
            "multistart": True,
            "n_starts": 6,
            "check_finite": True,
            "absolute_sigma": False,
        },
        "forwarded fit kwargs",
    )
    expect_curve_fit_result(result, pcov=SENT_PCOV)


def _check_global_streaming(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(HYBRID)
    _, base = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=STREAMING_LIMIT_GB
    )
    assert isinstance(base, HybridStreamingConfig)
    expect_config(
        ctor.named["config"],
        dataclasses.replace(
            base,
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=1e-8,
            enable_multistart=True,
            n_starts=6,
        ),
        "multistart streaming config",
    )
    # Unlike the other hybrid sites, bounds are passed on unconditionally.
    expect_hybrid_fit(
        rec.only_fit(HYBRID), p0=np.array(P0), bounds=FINITE_BOUNDS_ARR, sigma=None
    )
    expect_curve_fit_result(result, pcov=SENT_PCOV)


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


def _check_multistart_default_chunked(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    expect_config(
        ctor.named["config"],
        LDMemoryConfig(
            memory_limit_gb=8.0, min_chunk_size=1000, max_chunk_size=1_000_000
        ),
        "fallback chunked config without memory_config",
    )
    assert ctor.named["memory_limit_gb"] == 8.0


def _check_multistart_default_streaming(rec: Recorder, result: Any) -> None:
    expect_config(
        rec.ctors(HYBRID)[0].named["config"],
        HybridStreamingConfig(
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=1e-8,
            enable_multistart=True,
            n_starts=3,
        ),
        "fallback streaming config without memory_config",
    )


# ---- nlsq/core/minpack.py::CurveFit ---------------------------------------


def _drive_cf_hybrid(**kw: Any) -> Any:
    return nlsq.curve_fit(
        MODEL, X, Y, p0=P0, method="hybrid_streaming", verbose=0, **FAST_HYBRID, **kw
    )


def _check_cf_hybrid_real(rec: Recorder, result: Any) -> None:
    expect_config(
        rec.ctors(HYBRID)[0].named["config"],
        HybridStreamingConfig(**FAST_HYBRID),
        "curve_fit(hybrid_streaming) config from kwargs",
    )
    expect_hybrid_fit(
        rec.only_fit(HYBRID), p0=np.array(P0), bounds=None, sigma=None, verbose=0
    )
    assert isinstance(result, CurveFitResult)
    close_to_truth(result.x)
    assert result["model"] is MODEL and np.asarray(result["pcov"]).shape == (2, 2)


def _check_cf_hybrid_eval(rec: Recorder, result: Any) -> None:
    assert type(result) is tuple and len(result) == 3
    expect(result[0], SENT_X, "popt")
    expect(result[1], SENT_PCOV, "pcov")
    expect(result[2], np.asarray(MODEL(X, *SENT_X)), "f(xdata, *popt)")


def _check_cf_hybrid_full_output(rec: Recorder, result: Any) -> None:
    assert type(result) is tuple and len(result) == 5
    expect(result[0], SENT_X, "popt")
    expect(result[1], SENT_PCOV, "pcov")
    assert result[4] == 1  # success -> ier 1


def _check_cf_auto_streaming(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(HYBRID)
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=STREAMING_LIMIT_GB
    )
    # The selector's config is handed through (not a fresh default).
    expect_config(ctor.named["config"], expected, "method=auto streaming config")
    assert expected.chunk_size != HybridStreamingConfig().chunk_size
    expect_hybrid_fit(
        rec.only_fit(HYBRID), p0=np.array(P0), bounds=None, sigma=None, verbose=1
    )
    assert isinstance(result, CurveFitResult)
    expect(result["x"], SENT_X, "x")


def _check_cf_auto_chunked(rec: Recorder, result: Any) -> None:
    (ctor,) = rec.ctors(FITTER)
    _, expected = MemoryBudgetSelector().select(
        n_points=N, n_params=2, memory_limit_gb=CHUNKED_LIMIT_GB
    )
    expect_config(ctor.named["config"], expected, "method=auto chunked config")
    # The raw memory_limit_gb kwarg (not the config's) is the constructor limit.
    assert ctor.named["memory_limit_gb"] == CHUNKED_LIMIT_GB
    call = rec.only_fit(FITTER)
    assert call.named["f"] is MODEL
    expect(call.named["xdata"], X, "xdata")
    expect(call.named["ydata"], Y, "ydata")
    expect(call.named["p0"], P0, "p0")
    expect(call.named["bounds"], INF_BOUNDS, "bounds")
    assert call.named["method"] == "trf"  # engine default, no method forwarded
    expect(call.extra, {"check_finite": True, "absolute_sigma": False}, "fit kwargs")
    assert isinstance(result, CurveFitResult)
    expect(result["x"], SENT_X, "x")
    expect(result["pcov"], SENT_PCOV, "pcov")
    assert result["model"] is MODEL
    expect(result["xdata"], X, "xdata")


def _check_cf_auto_chunked_eval(rec: Recorder, result: Any) -> None:
    assert type(result) is tuple and len(result) == 3
    expect(result[0], SENT_X, "popt")
    expect(result[1], SENT_PCOV, "pcov")


def _check_cf_auto_chunked_timeit(rec: Recorder, result: Any) -> None:
    assert type(result) is tuple and len(result) == 5
    expect(result[0], SENT_X, "popt")
    assert isinstance(result[2], OptimizeResult)
    assert result[3] == 0


def _check_cf_auto_chunked_real(rec: Recorder, result: Any) -> None:
    assert len(rec.ctors()) == 1 and len(rec.ctors(FITTER)) == 1
    assert isinstance(result, CurveFitResult)
    close_to_truth(result.x)


ROWS: list[Row] = [
    # --- nlsq/__init__.py -------------------------------------------------
    Row(
        "init.fit[preset=streaming]",
        "nlsq.fit :651",
        HYBRID,
        lambda: _drive_fit_preset_streaming(
            sigma=SIG,
            multistart=True,
            n_starts=4,
            sampler="sobol",
            gtol=1e-3,
            ftol=1e-3,
        ),
        _check_fit_preset_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "init.fit[preset=streaming,multistart=False]",
        "nlsq.fit :651",
        HYBRID,
        lambda: _drive_fit_preset_streaming(multistart=False),
        _check_fit_preset_streaming_off,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "init.curve_fit_large[hybrid_streaming,real]",
        "nlsq.curve_fit_large :1146",
        HYBRID,
        _drive_clf_hybrid,
        _check_clf_hybrid_real,
    ),
    Row(
        "init.curve_fit_large[hybrid_streaming,stub]",
        "nlsq.curve_fit_large :1146",
        HYBRID,
        _drive_clf_hybrid,
        _check_clf_hybrid_stub,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "init.curve_fit_large[hybrid_streaming,no-pcov]",
        "nlsq.curve_fit_large :1146",
        HYBRID,
        _drive_clf_hybrid,
        _check_clf_hybrid_no_pcov,
        {HYBRID: hybrid_no_pcov},
    ),
    Row(
        "init.curve_fit_large[chunked,real]",
        "nlsq.curve_fit_large :1253",
        FITTER,
        lambda: _drive_clf_chunked(memory_limit_gb=2.0),
        _check_clf_chunked_real,
    ),
    Row(
        "init.curve_fit_large[chunked,args]",
        "nlsq.curve_fit_large :1253",
        FITTER,
        lambda: _drive_clf_chunked(
            memory_limit_gb=3.0,
            chunk_size=300,
            method="lm",
            sigma=SIG,
            absolute_sigma=True,
            bounds=FINITE_BOUNDS,
            multistart=True,
            n_starts=5,
            sampler="sobol",
            gtol=1e-3,
        ),
        _check_clf_chunked_args,
        {FITTER: fitter_ok},
    ),
    Row(
        "init.curve_fit_large[chunked,x-only-result]",
        "nlsq.curve_fit_large :1253",
        FITTER,
        lambda: _drive_clf_chunked(memory_limit_gb=2.0),
        _check_clf_chunked_x_only,
        {FITTER: fitter_x_only},
    ),
    Row(
        "init.curve_fit_large[chunked,show_progress]",
        "nlsq.curve_fit_large :1253",
        FITTER,
        lambda: _drive_clf_chunked(memory_limit_gb=2.0, show_progress=True),
        _check_clf_chunked_progress,
        {FITTER: fitter_ok},
    ),
    Row(
        "init.curve_fit_large[small-data,no-engine]",
        "nlsq.curve_fit_large (bypass)",
        None,
        lambda: nlsq.curve_fit_large(MODEL, X, Y, p0=P0),
        _check_clf_small,
    ),
    # --- nlsq/streaming/large_dataset.py ----------------------------------
    Row(
        "large_dataset.fit_large_dataset[real]",
        "fit_large_dataset :3053",
        FITTER,
        _drive_fit_large_dataset,
        _check_fit_large_dataset_real,
    ),
    Row(
        "large_dataset.fit_large_dataset[progress,stub]",
        "fit_large_dataset :3053",
        FITTER,
        lambda: large_dataset.fit_large_dataset(
            MODEL,
            X,
            Y,
            p0=P0,
            memory_limit_gb=2.0,
            show_progress=True,
            logger=_LOGGER,
            multistart=True,
            n_starts=3,
            sampler="halton",
            bounds=FINITE_BOUNDS,
        ),
        _check_fit_large_dataset_progress,
        {FITTER: fitter_ok},
    ),
    Row(
        "large_dataset.LargeDatasetFitter._fit_with_streaming[unwired,direct]",
        "LargeDatasetFitter._fit_with_streaming :1900",
        HYBRID,
        _drive_fitter_streaming,
        _check_fitter_streaming,
        {HYBRID: hybrid_ok},
        prebuilt=(FITTER,),
    ),
    Row(
        "large_dataset.LargeDatasetFitter._fit_with_streaming[engine-raises]",
        "LargeDatasetFitter._fit_with_streaming :1900",
        HYBRID,
        _drive_fitter_streaming,
        _check_fitter_streaming_failure,
        {HYBRID: _raise_kaboom},
        prebuilt=(FITTER,),
    ),
    # --- nlsq/core/minpack.py::_fit_with_config (via nlsq.fit workflow=) ---
    Row(
        "minpack._fit_with_config[HybridStreamingConfig,gtol]",
        "_fit_with_config :778",
        HYBRID,
        lambda: _drive_workflow_hybrid_config(gtol=1e-3, ftol=1e-4, xtol=1e-5),
        _check_workflow_hybrid_config,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_config[HybridStreamingConfig,goal]",
        "_fit_with_config :778",
        HYBRID,
        lambda: _drive_workflow_hybrid_config(goal="fast"),
        _check_workflow_hybrid_goal,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_config[HybridStreamingConfig,no-gtol,no-pcov]",
        "_fit_with_config :778",
        HYBRID,
        _drive_workflow_hybrid_config,
        _check_workflow_hybrid_no_gtol,
        {HYBRID: hybrid_no_pcov},
    ),
    Row(
        "minpack._fit_with_config[LDMemoryConfig,args]",
        "_fit_with_config :808",
        FITTER,
        lambda: _drive_workflow_ld_config(
            gtol=1e-3,
            ftol=1e-4,
            xtol=1e-5,
            sigma=SIG,
            absolute_sigma=True,
            bounds=FINITE_BOUNDS,
        ),
        _check_workflow_ld_config,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_with_config[LDMemoryConfig,real]",
        "_fit_with_config :808",
        FITTER,
        _drive_workflow_ld_config,
        _check_workflow_ld_config_real,
    ),
    Row(
        "minpack._fit_with_config[workflow=auto->chunked]",
        "_fit_with_config :808",
        FITTER,
        lambda: nlsq.fit(
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
        "_fit_with_config :778",
        HYBRID,
        lambda: nlsq.fit(
            MODEL,
            X,
            Y,
            p0=P0,
            workflow="auto",
            memory_limit_gb=STREAMING_LIMIT_GB,
            gtol=1e-3,
            verbose=1,
        ),
        _check_auto_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_config[workflow=auto->standard,no-engine]",
        "_fit_with_config (fallthrough)",
        None,
        lambda: nlsq.fit(MODEL, X, Y, p0=P0, workflow="auto"),
        _check_auto_standard,
    ),
    # --- nlsq/core/minpack.py::_fit_with_preset (direct helper only) -------
    Row(
        "minpack._fit_with_preset[large_robust,n_points=1M,direct]",
        "_fit_with_preset :1026",
        FITTER,
        lambda: _call_preset(
            "large_robust",
            n_points=1_000_000,
            sigma=SIG,
            absolute_sigma=True,
            check_finite=False,
        ),
        _check_preset_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_with_preset[streaming,direct]",
        "_fit_with_preset :1087",
        HYBRID,
        lambda: _call_preset("streaming", callback=_CALLBACK, verbose=0),
        _check_preset_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_preset[hpc_distributed,direct]",
        "_fit_with_preset :1087",
        HYBRID,
        lambda: _call_preset("hpc_distributed", sigma=SIG, bounds=FINITE_BOUNDS),
        _check_preset_hpc,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_with_preset[streaming,goal-overrides-preset-gtol]",
        "_fit_with_preset :1087",
        HYBRID,
        lambda: _call_preset("streaming", goal=OptimizationGoal.QUALITY),
        _check_preset_goal_tolerance,
        {HYBRID: hybrid_ok},
    ),
    # --- nlsq/core/minpack.py::_fit_global_multistart ---------------------
    Row(
        "minpack._fit_global_multistart[auto_global->chunked]",
        "_fit_global_multistart :1760",
        FITTER,
        lambda: _drive_auto_global(CHUNKED_LIMIT_GB),
        _check_global_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_global_multistart[auto_global->streaming]",
        "_fit_global_multistart :1825",
        HYBRID,
        lambda: _drive_auto_global(STREAMING_LIMIT_GB),
        _check_global_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack._fit_global_multistart[chunked,no-memory_config,direct]",
        "_fit_global_multistart :1760",
        FITTER,
        lambda: _call_multistart("chunked", memory_config=None),
        _check_multistart_default_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack._fit_global_multistart[streaming,no-memory_config,direct]",
        "_fit_global_multistart :1825",
        HYBRID,
        lambda: _call_multistart("streaming", memory_config=None),
        _check_multistart_default_streaming,
        {HYBRID: hybrid_ok},
    ),
    # --- nlsq/core/minpack.py::CurveFit -----------------------------------
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[real]",
        "_curve_fit_hybrid_streaming :3564",
        HYBRID,
        _drive_cf_hybrid,
        _check_cf_hybrid_real,
    ),
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[return_eval]",
        "_curve_fit_hybrid_streaming :3564",
        HYBRID,
        lambda: _drive_cf_hybrid(return_eval=True),
        _check_cf_hybrid_eval,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_hybrid_streaming[full_output]",
        "_curve_fit_hybrid_streaming :3564",
        HYBRID,
        lambda: _drive_cf_hybrid(full_output=True),
        _check_cf_hybrid_full_output,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[streaming]",
        "_curve_fit_hybrid_streaming :3564 via _curve_fit_auto_memory",
        HYBRID,
        lambda: nlsq.curve_fit(
            MODEL,
            X,
            Y,
            p0=P0,
            method="auto",
            memory_limit_gb=STREAMING_LIMIT_GB,
        ),
        _check_cf_auto_streaming,
        {HYBRID: hybrid_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked]",
        "_curve_fit_auto_memory :3776",
        FITTER,
        lambda: nlsq.curve_fit(
            MODEL,
            X,
            Y,
            p0=P0,
            method="auto",
            memory_limit_gb=CHUNKED_LIMIT_GB,
        ),
        _check_cf_auto_chunked,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked,return_eval]",
        "_curve_fit_auto_memory :3776",
        FITTER,
        lambda: nlsq.curve_fit(
            MODEL,
            X,
            Y,
            p0=P0,
            method="auto",
            memory_limit_gb=CHUNKED_LIMIT_GB,
            return_eval=True,
        ),
        _check_cf_auto_chunked_eval,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked,timeit]",
        "_curve_fit_auto_memory :3776",
        FITTER,
        lambda: nlsq.curve_fit(
            MODEL,
            X,
            Y,
            p0=P0,
            method="auto",
            memory_limit_gb=CHUNKED_LIMIT_GB,
            timeit=True,
        ),
        _check_cf_auto_chunked_timeit,
        {FITTER: fitter_ok},
    ),
    Row(
        "minpack.CurveFit._curve_fit_auto_memory[chunked,real]",
        "_curve_fit_auto_memory :3776",
        FITTER,
        lambda: nlsq.curve_fit(
            MODEL, X, Y, p0=P0, method="auto", memory_limit_gb=CHUNKED_LIMIT_GB
        ),
        _check_cf_auto_chunked_real,
    ),
]


@pytest.mark.parametrize("row", ROWS, ids=[r.id for r in ROWS])
def test_routing_table(row: Row, rec: Recorder) -> None:
    rec.stubs.update(row.stubs)
    with np.errstate(all="ignore"):
        result = row.drive()
    # Exactly one construction, of exactly the expected class (none for bypass rows);
    # this is what catches a class swap or a class built twice.
    built = [c.engine for c in rec.ctors()]
    expected = [*row.prebuilt, *([row.engine] if row.engine else [])]
    assert built == expected, f"constructed {built}, expected {expected}"
    row.check(rec, result)


# --------------------------------------------------------------------------
# Behaviour around the rows: rejections, failures, negative routing
# --------------------------------------------------------------------------


def test_legacy_preset_keys_rejected_publicly(rec: Recorder) -> None:
    """Every WORKFLOW_PRESETS key is rejected before _fit_with_preset can run.

    So the two constructors inside _fit_with_preset have no public path; the
    rows above drive them through the helper directly.
    """
    assert set(minpack.WORKFLOW_PRESETS) == set(minpack.REMOVED_PRESETS)
    for key in minpack.WORKFLOW_PRESETS:
        for spelling in (key, key.upper()):
            with pytest.raises(ValueError, match=r"removed in v0\.6\.3"):
                nlsq.fit(MODEL, X, Y, p0=P0, workflow=spelling, bounds=FINITE_BOUNDS)
            with pytest.raises(ValueError, match=r"removed in v0\.6\.3"):
                minpack.fit(MODEL, X, Y, p0=P0, workflow=spelling)
    assert rec.calls == []


def test_fit_preset_helper_has_a_single_gated_caller() -> None:
    """_fit_with_preset is only called from fit(), after the REMOVED_PRESETS gate."""
    source = inspect.getsource(minpack)
    assert source.count("_fit_with_preset(") == 2  # the def + the one call
    fit_source = inspect.getsource(minpack.fit)
    gate = fit_source.index("REMOVED_PRESETS")
    assert gate < fit_source.index("_fit_with_preset(")


def test_fitter_fit_never_builds_hybrid(rec: Recorder) -> None:
    """LargeDatasetFitter.fit (use_streaming defaults to True) never routes to
    the hybrid optimizer, so _fit_with_streaming stays unreachable."""
    assert LDMemoryConfig().use_streaming is True
    fitter = _REAL[FITTER](config=LDMemoryConfig(memory_limit_gb=2.0))
    result = fitter.fit(MODEL, X, Y, p0=P0)
    assert result.success
    assert [c.engine for c in rec.ctors()] == [FITTER]
    close_to_truth(result.x)


def test_hybrid_workflow_routes_reject_tolerance_kwargs(rec: Recorder) -> None:
    """curve_fit_large / curve_fit hybrid routes accept only config fields."""
    with pytest.raises(TypeError, match="Unrecognized keyword"):
        nlsq.curve_fit_large(MODEL, X, Y, p0=P0, method="hybrid_streaming", gtol=1e-3)
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
    """A memory_config of the other engine's type is ignored, not reused."""
    rec.stubs[FITTER] = fitter_ok
    rec.stubs[HYBRID] = hybrid_ok
    _call_multistart("chunked", memory_config=HybridStreamingConfig(chunk_size=17))
    expect_config(
        rec.ctors(FITTER)[0].named["config"],
        LDMemoryConfig(
            memory_limit_gb=8.0, min_chunk_size=1000, max_chunk_size=1_000_000
        ),
        "chunked fallback config",
    )
    _call_multistart("streaming", memory_config=LDMemoryConfig(memory_limit_gb=1.0))
    assert rec.ctors(HYBRID)[0].named["config"].chunk_size == (
        HybridStreamingConfig().chunk_size
    )
