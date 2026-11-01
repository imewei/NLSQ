"""curve_fit_large(method="exact_streaming") dispatch, argument policy and the fit() gate."""

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
        popt, _ = curve_fit_large(
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
        ({"bounds": (0.0, np.inf)}, r"^finite bounds are not supported by "),
        ({"bounds": ([-np.inf] * 3, [np.inf, 5.0, np.inf])}, r"^finite bounds are "),
    ],
)
def test_stage_scoped_arguments_valueerror(data, spy, kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        curve_fit_large(exp_model, *data, p0=EXP.p0, method=M, **kwargs)
    assert not spy


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
