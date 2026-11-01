"""curve_fit_large(method="exact_streaming") dispatch, argument policy and the fit() gate."""

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
import nlsq.streaming.exact_lm as lm
from benchmarks.compare.problems import PROBLEMS
from nlsq import curve_fit_large
from nlsq.streaming import ExactSolverError
from nlsq.streaming.large_dataset import LargeDatasetFitter

EXP = PROBLEMS["exp_decay"]
N = 3000
M = "exact_streaming"


def exp_model(x, a, b, c):
    return EXP.model(jnp, x, a, b, c)


@pytest.fixture(scope="module")
def data():
    return EXP.data(N)


@pytest.fixture
def spy(monkeypatch):
    """Record every exact_lm_fit call made through the dispatch."""
    calls = []
    real = lm.exact_lm_fit

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(lm, "exact_lm_fit", wrapper)
    return calls


def test_dispatch_returns_popt_pcov_and_matches_module(data):
    direct = lm.exact_lm_fit(exp_model, *data, EXP.p0, chunk_size=1024)
    assert direct["success"]
    out = curve_fit_large(exp_model, *data, p0=EXP.p0, method=M, chunk_size=1024)
    assert isinstance(out, tuple) and len(out) == 2
    popt, pcov = out
    np.testing.assert_array_equal(popt, direct["x"])
    np.testing.assert_array_equal(pcov, direct["pcov"])
    np.testing.assert_allclose(popt, EXP.p_true, rtol=1e-2)


def test_honoured_arguments_are_forwarded(data, spy):
    curve_fit_large(
        exp_model,
        *data,
        p0=EXP.p0,
        method=M,
        chunk_size=1024,
        check_finite=False,
        absolute_sigma=True,
        memory_limit_gb=2.0,
        show_progress=True,
        sigma=np.full(N, 0.05),
    )
    kw = spy[0][1]
    np.testing.assert_array_equal(kw["sigma"], np.full(N, 0.05))
    assert kw["chunk_size"] == 1024 and kw["check_finite"] is False
    assert kw["absolute_sigma"] is True and kw["memory_limit_gb"] == 2.0
    assert kw["show_progress"] is True


def test_weighted_dispatch_matches_module(data):
    x, y = data
    s = 0.02 + 0.1 * x
    direct = lm.exact_lm_fit(exp_model, x, y, EXP.p0, sigma=s, chunk_size=1024)
    assert direct["success"]
    for a in (False, True):
        ref = lm.exact_lm_fit(
            exp_model, x, y, EXP.p0, sigma=s, absolute_sigma=a, chunk_size=1024
        )
        popt, pcov = curve_fit_large(
            exp_model,
            x,
            y,
            p0=EXP.p0,
            sigma=s,
            absolute_sigma=a,
            method=M,
            chunk_size=1024,
        )
        np.testing.assert_array_equal(popt, direct["x"])
        np.testing.assert_array_equal(pcov, ref["pcov"])
    unweighted = curve_fit_large(exp_model, x, y, p0=EXP.p0, method=M, chunk_size=1024)
    assert not np.array_equal(unweighted[0], popt)


def test_failure_raises_with_result(data):
    a0 = EXP.p0[0]

    def stalling(x, a, b, c):
        return jnp.where(a == a0, exp_model(x, a, b, c), jnp.nan)

    with pytest.raises(ExactSolverError, match=r"\(stalled\)") as exc:
        curve_fit_large(stalling, *data, p0=EXP.p0, method=M, chunk_size=1024)
    assert isinstance(exc.value, RuntimeError)
    assert exc.value.result["termination"] == "stalled"
    assert exc.value.result["success"] is False

    with pytest.raises(ExactSolverError, match=r"\(max_nfev\)") as exc:
        curve_fit_large(exp_model, *data, p0=EXP.p0, method=M, max_nfev=2)
    assert exc.value.result["termination"] == "max_nfev"
    assert exc.value.result["nfev"] == 2


def test_exact_solver_error_pickles():
    err = ExactSolverError(
        "fit failed (stalled): m", nlsq.OptimizeResult(termination="stalled")
    )
    back = pickle.loads(pickle.dumps(err))
    assert isinstance(back, ExactSolverError)
    assert str(back) == "fit failed (stalled): m"
    assert back.result["termination"] == "stalled"


def test_solver_kwargs_are_forwarded(data, spy):
    curve_fit_large(
        exp_model, *data, p0=EXP.p0, method=M, ftol=1e-10, xtol=1e-11, gtol=1e-12
    )
    kw = spy[0][1]
    assert (kw["ftol"], kw["xtol"], kw["gtol"]) == (1e-10, 1e-11, 1e-12)


@pytest.mark.parametrize("bad", ["foo", "verbose", "loss", "callback"])
def test_unknown_kwarg_typeerror(data, spy, bad):
    with pytest.raises(TypeError, match=rf"exact_streaming.*\['{bad}'\]"):
        curve_fit_large(exp_model, *data, p0=EXP.p0, method=M, **{bad: 1})
    assert not spy


NON_DEFAULT = [
    ("stability", "auto"),
    ("rescale_data", False),
    ("max_jacobian_elements_for_svd", 5),
    ("auto_size_detection", False),
    ("size_threshold", 10),
]


class Lazy1D:
    """Array-like that must never be read whole: only shape/len are allowed."""

    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def __len__(self):
        return self.shape[0]

    def __array__(self, *a, **k):
        raise AssertionError("data was materialised")


@pytest.mark.parametrize(("name", "value"), NON_DEFAULT)
def test_unsupported_arguments_valueerror(data, spy, name, value):
    with pytest.raises(ValueError, match=rf"^{name}={value!r} is not supported by "):
        curve_fit_large(exp_model, *data, p0=EXP.p0, method=M, **{name: value})
    assert not spy


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"multistart": True}, r"^multistart=True is not supported by "),
        ({"global_search": True}, r"^multistart=True is not supported by "),
        ({"sigma": Lazy1D((N, N))}, r"^sigma must be 1-D for method='exact_streaming'"),
        ({"bounds": (1.0, 0.0)}, r"^each lower bound must be strictly less"),
        ({"bounds": (0.0, 0.0)}, r"^each lower bound must be strictly less"),
        ({"bounds": (np.inf, -np.inf)}, r"^each lower bound must be strictly less"),
        ({"bounds": ([0.0] * 3, [1.0, 0.0, 1.0])}, r"^each lower bound must be"),
        (
            {"bounds": ([0.0] * 2, [1.0] * 3)},
            r"^lower bound has shape \(2,\), expected",
        ),
        ({"bounds": (0.0, [1.0] * 4)}, r"^upper bound has shape \(4,\), expected"),
    ],
)
def test_stage_scoped_arguments_valueerror(data, spy, kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        curve_fit_large(exp_model, *data, p0=EXP.p0, method=M, **kwargs)
    assert not spy


def test_bounded_dispatch_matches_module(spy):
    prob = PROBLEMS["bounded_exp"]
    x, y = prob.data(N)

    def model(x, a, b, c):
        return prob.model(jnp, x, a, b, c)

    popt, pcov = curve_fit_large(
        model, x, y, p0=prob.p0, bounds=prob.bounds, method=M, chunk_size=1024
    )
    lo, hi = spy[0][1]["bounds"]
    np.testing.assert_array_equal(lo, prob.bounds[0])
    np.testing.assert_array_equal(hi, prob.bounds[1])
    direct = lm.exact_lm_fit(model, x, y, prob.p0, bounds=prob.bounds, chunk_size=1024)
    np.testing.assert_array_equal(popt, direct["x"])
    np.testing.assert_array_equal(pcov, direct["pcov"])
    assert popt[0] == 2.0  # the upper bound on `a` is active


def test_exact_defaults_match_curve_fit_large_signature():
    import inspect

    params = inspect.signature(curve_fit_large).parameters
    assert {k: params[k].default for k in nlsq._EXACT_DEFAULTS} == nlsq._EXACT_DEFAULTS


def test_multistart_suboptions_ignored_when_off(data, spy):
    curve_fit_large(
        exp_model,
        *data,
        p0=EXP.p0,
        method=M,
        n_starts=3,
        sampler="sobol",
        center_on_p0=False,
        scale_factor=2.0,
    )
    assert len(spy) == 1


def test_shapes_validated_without_reading_data(spy):
    y = np.ones(10)
    with pytest.raises(ValueError, match=r"^xdata must be 1-D for method="):
        curve_fit_large(exp_model, Lazy1D((10, 2)), y, p0=EXP.p0, method=M)
    with pytest.raises(ValueError, match=r"^ydata must be 1-D for method="):
        curve_fit_large(exp_model, Lazy1D((10,)), Lazy1D((10, 1)), p0=EXP.p0, method=M)
    with pytest.raises(ValueError, match=r"same length: 10 vs 9"):
        curve_fit_large(exp_model, Lazy1D((10,)), Lazy1D((9,)), p0=EXP.p0, method=M)
    with pytest.raises(
        ValueError, match=r"at least 2 data points for fitting, got 1\."
    ):
        curve_fit_large(exp_model, Lazy1D((1,)), Lazy1D((1,)), p0=EXP.p0, method=M)
    assert not spy


@pytest.mark.serial
def test_runs_regardless_of_size_threshold(spy):
    n = 100_000  # below the default 1M threshold, which would route to curve_fit
    x, y = EXP.data(n)
    popt, _ = curve_fit_large(exp_model, x, y, p0=EXP.p0, method=M)
    assert len(spy) == 1
    np.testing.assert_allclose(popt, EXP.p_true, rtol=1e-2)


def test_p0_none_infers_parameter_count(data, spy):
    popt, pcov = curve_fit_large(exp_model, *data, method=M, chunk_size=1024)
    np.testing.assert_array_equal(spy[0][0][3], np.ones(3))
    assert popt.shape == (3,) and pcov.shape == (3, 3)
    with pytest.raises(
        ValueError, match="Unable to determine number of fit parameters"
    ):
        curve_fit_large(lambda x: x, *data, method=M)


def test_hybrid_streaming_default_p0(data, monkeypatch):
    from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer

    seen = []

    def fake_fit(self, *args, **kwargs):
        seen.append(kwargs["p0"])
        return {"x": kwargs["p0"], "pcov": np.eye(len(kwargs["p0"]))}

    monkeypatch.setattr(AdaptiveHybridStreamingOptimizer, "fit", fake_fit)
    curve_fit_large(exp_model, *data, method="hybrid_streaming")
    np.testing.assert_array_equal(seen[0], np.ones(3))
    with pytest.raises(
        ValueError, match="Unable to determine number of fit parameters"
    ):
        curve_fit_large(lambda x: x, *data, method="hybrid_streaming")
    assert len(seen) == 1  # the second call failed before reaching the optimizer


@pytest.mark.parametrize("n", [100, 5000])
def test_fit_rejects_exact_streaming(spy, n):
    x, y = EXP.data(n)
    # size_threshold=1000 sends n=5000 down fit()'s curve_fit_large branch
    with pytest.raises(
        ValueError, match=r"exact_streaming.*only through curve_fit_large"
    ):
        nlsq.fit(exp_model, x, y, EXP.p0, method=M, size_threshold=1000)
    assert not spy


def test_default_chunked_path_unchanged(monkeypatch, spy):
    results = []
    real = LargeDatasetFitter.fit

    def capture(self, *args, **kwargs):
        results.append(real(self, *args, **kwargs))
        return results[-1]

    monkeypatch.setattr(LargeDatasetFitter, "fit", capture)
    x = np.linspace(0.0, 4.0, 5000)
    y = 2.0 * x + 1.0 + np.random.default_rng(0).normal(0, 0.01, 5000)
    popt, _ = curve_fit_large(
        lambda x, a, b: a * x + b, x, y, p0=[1.0, 0.0], size_threshold=1000
    )
    assert not spy
    (res,) = results
    assert res["equivalent_to_full_batch"] is (res["n_chunks"] == 1)
    np.testing.assert_allclose(popt, [2.0, 1.0], rtol=1e-2)


# ---- chunk factories (callable xdata) ---------------------------------------------------


def _factory(x, y):
    def factory(chunk):
        for s in range(0, len(x), chunk):
            yield x[s : s + chunk], y[s : s + chunk]

    return factory


@pytest.mark.parametrize("n_points", [None, N])
def test_factory_through_curve_fit_large(data, spy, n_points):
    kw = {} if n_points is None else {"n_points": n_points}
    popt, pcov = curve_fit_large(
        exp_model, _factory(*data), None, p0=EXP.p0, method=M, chunk_size=1024, **kw
    )
    assert callable(spy[0][0][1]) and spy[0][0][2] is None
    assert spy[0][1].get("n_points") == n_points
    direct = lm.exact_lm_fit(exp_model, *data, EXP.p0, chunk_size=1024)
    np.testing.assert_allclose(popt, direct["x"], rtol=1e-9, atol=0)
    np.testing.assert_allclose(pcov, direct["pcov"], rtol=1e-9, atol=0)


def test_callable_xdata_with_ydata_raises(data, spy):
    with pytest.raises(
        ValueError, match=r"^ydata must be None when xdata is a chunk factory"
    ):
        curve_fit_large(exp_model, _factory(*data), data[1], p0=EXP.p0, method=M)
    with pytest.raises(ValueError, match=r"^sigma must be None with a chunk factory"):
        curve_fit_large(
            exp_model, _factory(*data), None, p0=EXP.p0, sigma=np.ones(N), method=M
        )
    assert not spy


@pytest.mark.parametrize("n_points", [N, 0])
def test_n_points_rejected_without_factory(data, spy, n_points):
    with pytest.raises(
        ValueError, match=r"^n_points is accepted only with a chunk factory"
    ):
        curve_fit_large(exp_model, *data, p0=EXP.p0, method=M, n_points=n_points)
    assert not spy


def test_n_points_none_allowed_without_factory(data, spy):
    curve_fit_large(
        exp_model, *data, p0=EXP.p0, method=M, chunk_size=1024, n_points=None
    )
    assert len(spy) == 1 and "n_points" not in spy[0][1]
