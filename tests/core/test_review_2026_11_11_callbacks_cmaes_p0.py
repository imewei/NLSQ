"""Regression tests: positional callbacks, CMA-ES scalar sigma streaming,
IterationLogger under multistart deepcopy, exponential_decay p0 x-origin."""

import copy
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.callbacks import IterationLogger, StopOptimization
from nlsq.diagnostics.gradient_health import GradientMonitor
from nlsq.diagnostics.types import DiagnosticsConfig
from nlsq.global_optimization.cmaes_config import CMAESConfig
from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer
from nlsq.precision.parameter_estimation import estimate_p0_for_pattern


def _f(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=60):
    x = np.linspace(0, 4, n)
    rng = np.random.default_rng(0)
    return x, 2.0 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(n)


def test_callback_with_arbitrary_param_names_is_called():
    x, y = _data()
    seen = []

    def cb(it, c, p, inf):
        seen.append(it)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        curve_fit(_f, x, y, p0=[1.0, 1.0], callback=cb)
    assert len(seen) > 0


def test_callback_stop_optimization_fires_with_arbitrary_names():
    x, y = _data()
    seen = []

    def cb(it, c, p, inf):
        seen.append(it)
        raise StopOptimization

    curve_fit(_f, x, y, p0=[1.0, 1.0], callback=cb)
    assert len(seen) == 1


def test_gradient_monitor_forwards_positionally():
    seen = []

    def user(it, c, p, inf):
        seen.append(it)

    cb = GradientMonitor(DiagnosticsConfig()).create_callback(user_callback=user)
    cb(3, 1.0, np.array([1.0, 2.0]), {"gradient_norm": 1.0})
    assert seen == [3]


@pytest.mark.slow
def test_cmaes_scalar_sigma_streaming_matches_nonstreaming():
    n = 3000
    x = np.linspace(0, 4, n)
    y = 2.0 * np.exp(-0.7 * x) + 0.01 * np.random.default_rng(1).standard_normal(n)
    bounds = ([0.0, 0.0], [10.0, 5.0])
    res = []
    for chunk in (1024, None):
        cfg = CMAESConfig(
            data_chunk_size=chunk, restart_strategy="none", seed=0, max_generations=30
        )
        res.append(CMAESOptimizer(cfg).fit(_f, x, y, bounds=bounds, sigma=0.1))
    np.testing.assert_allclose(res[0]["popt"], res[1]["popt"], rtol=1e-6)


@pytest.mark.serial
def test_iteration_logger_survives_multistart_deepcopy(tmp_path):
    x, y = _data()
    path = tmp_path / "log.txt"
    logger = IterationLogger(str(path))
    assert copy.deepcopy(logger) is logger
    curve_fit(_f, x, y, p0=[1.0, 1.0], multistart=True, n_starts=4, callback=logger)
    text = path.read_text()
    assert text.count("NLSQ Optimization Log") >= 1
    n_iter_lines = sum(1 for ln in text.splitlines() if ln.startswith("Iter "))
    assert n_iter_lines > 4
    assert logger._start_time is not None  # the user's instance saw iterations


def test_p0_exponential_decay_shifted_x():
    x = np.linspace(10, 15, 50)
    y = 3 * np.exp(-0.5 * (x - 10)) + 1
    p0 = estimate_p0_for_pattern("exponential_decay", x, y, 3)
    assert abs(p0[1] - 0.5) / 0.5 < 0.2


# --- callback signature styles (keyword vs positional dispatch) ---------------


def _make_style(style):  # noqa: C901 - flat table of signature styles
    """Return (callback, calls); each call records (iteration, cost, params, info)."""
    import functools

    from nlsq.callbacks import CallbackBase

    calls = []

    def rec(iteration, cost, params, info):
        calls.append((iteration, cost, params, info))

    if style == "canonical":

        def cb(iteration, cost, params, info):
            rec(iteration, cost, params, info)

    elif style == "renamed":

        def cb(it, c, p, inf):
            rec(it, c, p, inf)

    elif style == "kwonly":

        def cb(*, iteration, cost, params, info):
            rec(iteration, cost, params, info)

    elif style == "varkw":

        def cb(**kw):
            rec(kw["iteration"], kw["cost"], kw["params"], kw["info"])

    elif style == "partial_names_kw":  # names only some of the four + **kw

        def cb(iteration, cost, **kw):
            rec(iteration, cost, kw["params"], kw["info"])

    elif style == "three_names_kw":

        def cb(iteration, cost, params, **kw):
            rec(iteration, cost, params, kw["info"])

    elif style == "one_name_kw":

        def cb(iteration, **kw):
            rec(iteration, kw["cost"], kw["params"], kw["info"])

    elif style == "varargs":

        def cb(*args):
            rec(*args)

    elif style == "shuffled":  # keyword names must win over positions

        def cb(params, cost, iteration, info):
            rec(iteration, cost, params, info)

    elif style == "partial":

        def base(tag, a, b, c, d):
            rec(a, b, c, d)

        cb = functools.partial(base, "t")
    elif style == "bound_method":

        class H:
            def method(self, iteration, cost, params, info):
                rec(iteration, cost, params, info)

        cb = H().method
    elif style == "bound_method_renamed":

        class H2:
            def method(self, a, b, c, d):
                rec(a, b, c, d)

        cb = H2().method
    else:  # callbackbase

        class Sub(CallbackBase):
            def __call__(self, iteration, cost, params, info):
                rec(iteration, cost, params, info)

        cb = Sub()
    return cb, calls


@pytest.mark.parametrize(
    "style",
    [
        "canonical",
        "renamed",
        "kwonly",
        "varkw",
        "partial_names_kw",
        "three_names_kw",
        "one_name_kw",
        "varargs",
        "shuffled",
        "partial",
        "bound_method",
        "bound_method_renamed",
        "callbackbase",
    ],
)
def test_callback_signature_styles_called_once_per_iteration(style):
    x, y = _data()
    cb, calls = _make_style(style)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        curve_fit(_f, x, y, p0=[1.0, 1.0], callback=cb)
    assert len(calls) > 0
    its = [c[0] for c in calls]
    assert its == sorted(set(its))  # each iteration exactly once, in order
    assert isinstance(calls[0][1], float)
    assert np.asarray(calls[0][2]).shape == (2,)
    assert "gradient_norm" in calls[0][3]


def test_callback_kwonly_stop_optimization_aborts_fit():
    x, y = _data()
    seen = []

    def cb(*, iteration, cost, params, info):
        seen.append(iteration)
        raise StopOptimization

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        curve_fit(_f, x, y, p0=[1.0, 1.0], callback=cb)
    assert len(seen) == 1


def test_callback_chain_forwards_kwonly_children():
    from nlsq.callbacks import CallbackChain

    seen = []

    def kw(*, iteration, cost, params, info):
        seen.append(iteration)

    CallbackChain(kw)(2, 1.0, np.zeros(2), {})
    assert seen == [2]


def test_gradient_monitor_forwards_kwonly_user_callback():
    seen = []

    def user(*, iteration, cost, params, info):
        seen.append(iteration)

    cb = GradientMonitor(DiagnosticsConfig()).create_callback(user_callback=user)
    cb(iteration=3, cost=1.0, params=np.array([1.0, 2.0]), info={"gradient_norm": 1.0})
    assert seen == [3]


def test_iteration_logger_pickle_roundtrip_and_deepcopy_identity(tmp_path):
    import pickle

    p = tmp_path / "log.txt"
    lg = IterationLogger(filename=str(p), log_params=True)
    lg2 = pickle.loads(pickle.dumps(lg))
    assert lg2.filename == str(p) and lg2.log_params is True
    assert lg2._lock is not lg._lock
    assert copy.deepcopy(lg) is lg
    # unpickled copy must not truncate an existing log
    p.write_text("keep\n")
    lg2(0, 1.0, np.zeros(2), {"gradient_norm": 1.0, "nfev": 1})
    lg2.close()
    assert p.read_text().startswith("keep\n")


def test_iteration_logger_pickle_after_use_drops_open_handle(tmp_path):
    import pickle

    p = tmp_path / "log2.txt"
    lg = IterationLogger(filename=str(p))
    lg(0, 1.0, np.zeros(2), {"gradient_norm": 1.0, "nfev": 1})  # opens file
    lg2 = pickle.loads(pickle.dumps(lg))
    assert lg2._file is None
    lg.close()
