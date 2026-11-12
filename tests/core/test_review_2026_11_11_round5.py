"""Round-5 hygiene regressions: band Jacobian fallback step, callback dispatch,
CMA-ES seed draw, exact-route diagnostics, hybrid default verbosity."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit, curve_fit_large
from nlsq.callbacks import CallbackBase, CallbackChain
from nlsq.streaming.specs import Exact


def _exp(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# 1. central-difference fallback keeps an absolute floor on the step
def test_band_fallback_jacobian_small_parameter_exact():
    def np_model(x, a, b, c):  # numpy-only: jacfwd tracing fails
        return a * np.exp(-b * np.asarray(x)) + c

    x = np.linspace(0, 3, 40)
    res = curve_fit(_exp, x, _exp(x, 2.0, 1.0, 1e-12), p0=[1.5, 0.8, 1e-12])
    res.model = np_model
    _, jac = res._finite_diff_jacobian(x, np.array([2.0, 1.0, 1e-12]))
    np.testing.assert_allclose(jac[:, 2], 1.0, atol=1e-6)


# 2. functools.wraps around *args child inside a chain
def test_callback_chain_wrapped_varargs_child():
    seen = []

    def w(*args):
        seen.append(args)

    @functools.wraps(lambda iteration, cost, params, info: None)
    def wrapped(*args):
        return w(*args)

    CallbackChain(wrapped)(1, 2.0, np.zeros(2), {})
    assert len(seen) == 1 and seen[0][0] == 1


def test_callback_chain_keyword_styles_still_dispatch():
    got = []

    def kw_only(*, iteration, cost, params, info):
        got.append(("kw", iteration))

    def kwargs_only(**kw):
        got.append(("kwargs", kw["iteration"]))

    CallbackChain(kw_only, kwargs_only)(3, 1.0, np.zeros(1), {})
    assert got == [("kw", 3), ("kwargs", 3)]


# 3. CMA-ES seed=None must not consume the global numpy stream
def test_cmaes_seed_none_leaves_global_rng_untouched():
    pytest.importorskip("evosax")
    from nlsq.global_optimization import CMAESConfig
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

    np.random.seed(123)
    before = np.random.get_state()[1].copy(), np.random.get_state()[2]
    seed = CMAESOptimizer(CMAESConfig(seed=None))._resolve_seed()
    after = np.random.get_state()[1], np.random.get_state()[2]
    assert isinstance(seed, int)
    assert np.array_equal(before[0], after[0]) and before[1] == after[1]


# 5. per-point captured array on the default/Exact route
def _per_point_problem(n=10_000):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, n)
    w = jnp.asarray(rng.uniform(0.5, 1.5, n))

    def model(x, a, b):
        return a * w * jnp.exp(-b * x)

    return model, x, 2.0 * np.asarray(w) * np.exp(-1.0 * x)


def test_exact_route_per_point_capture_raises_value_error_with_hint():
    model, x, y = _per_point_problem()
    with pytest.raises(ValueError, match=r"not pointwise.*Chunked\(\)"):
        curve_fit_large(model, x, y, p0=[1.5, 0.8], size_threshold=1000)
    with pytest.raises(ValueError, match=r"not pointwise.*Chunked\(\)"):
        curve_fit_large(model, x, y, p0=[1.5, 0.8], strategy=Exact())


# 6. non-finite data under Exact + check_finite=False blames the data
def test_exact_nonfinite_y_blames_data():
    x = np.linspace(0, 3, 2000)
    y = 2.0 * np.exp(-x)
    y[5] = np.nan
    with pytest.raises(ValueError, match=r"non-finite.*(check_finite|Chunked)"):
        curve_fit_large(
            lambda x, a, b: a * jnp.exp(-b * x),
            x,
            y,
            p0=[1.5, 0.8],
            strategy=Exact(),
            check_finite=False,
        )


# 4. hybrid streaming is silent by default; explicit verbose still prints
def _hybrid_data():
    x = np.linspace(0, 3, 3000)
    return x, 2 * np.exp(-x) + 0.5


@pytest.mark.slow
@pytest.mark.parametrize("fitter", [curve_fit, curve_fit_large])
def test_hybrid_streaming_silent_by_default(fitter, capfd):
    x, y = _hybrid_data()
    fitter(_exp, x, y, p0=[1, 1, 0.1], method="hybrid_streaming")
    out = capfd.readouterr().out
    assert out == ""
    fitter(_exp, x, y, p0=[1, 1, 0.1], method="hybrid_streaming", verbose=1)
    assert "Adaptive Hybrid Streaming Optimizer" in capfd.readouterr().out


def test_hybrid_config_verbose_zero_silences_fit(capfd):
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    x, y = _hybrid_data()
    curve_fit(
        _exp,
        x,
        y,
        p0=[1, 1, 0.1],
        method="hybrid_streaming",
        config=HybridStreamingConfig(verbose=0),
        verbose=1,
    )
    assert capfd.readouterr().out == ""
