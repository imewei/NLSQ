import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, 100)
    return x, 2.0 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(100)


def test_fit_with_workflow_forwards_multistart():
    x, y = _data()
    res = nlsq.fit(
        _model,
        x,
        y,
        bounds=([0, 0], [10, 10]),
        workflow="auto",
        multistart=True,
        n_starts=4,
    )
    diag = getattr(res, "multistart_diagnostics", None)
    assert diag is not None and not diag.get("bypassed", True)


def test_fit_with_workflow_warns_on_preset():
    x, y = _data()
    with pytest.warns(UserWarning, match="preset"):
        nlsq.fit(_model, x, y, p0=[1.0, 1.0], workflow="auto", preset="robust")


def test_telemetry_recent_events_nonpositive_n():
    from nlsq.streaming.telemetry import DefenseLayerTelemetry

    t = DefenseLayerTelemetry()
    assert t.get_recent_events(0) == []
    assert t.get_recent_events(-3) == []


def test_algorithm_selector_scalar_p0():
    from nlsq.precision.algorithm_selector import auto_select_algorithm

    x = np.linspace(0, 1, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = auto_select_algorithm(lambda x, a: a * x, x, 2.0 * x, p0=1.0)
    assert out is not None


@pytest.mark.parametrize("workflow", ["auto_global", "hpc"])
def test_fit_global_workflows_accept_multistart_and_lm(workflow):
    x, y = _data()
    kw = {"bounds": ([0, 0], [10, 10]), "workflow": workflow}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = nlsq.fit(_model, x, y, p0=[1.0, 1.0], multistart=True, n_starts=2, **kw)
        r2 = nlsq.fit(_model, x, y, p0=[1.0, 1.0], method="lm", **kw)
    # multistart is not used under an explicit workflow (it is ignored with a
    # warning); both calls must still return the correct fit.
    for r in (r1, r2):
        np.testing.assert_allclose(np.asarray(r.popt), [2.0, 0.7], atol=0.05)


def test_fit_workflow_show_progress_warns_not_raises():
    from nlsq.streaming import LDMemoryConfig

    x, y = _data()
    with pytest.warns(UserWarning, match="show_progress"):
        nlsq.fit(
            _model,
            x,
            y,
            p0=[1.0, 1.0],
            workflow=LDMemoryConfig(memory_limit_gb=1.0),
            show_progress=True,
        )
