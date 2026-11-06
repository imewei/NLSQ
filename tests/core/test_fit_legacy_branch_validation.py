"""R8-3 / R8-5: nlsq.fit() legacy (workflow=None) goal and preset handling."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq import fit


def _f(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=400):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, n)
    y = 3.0 * np.exp(-0.7 * x) + 0.05 * rng.standard_normal(n)
    return x, y


def test_unknown_goal_raises_without_workflow():
    x, y = _data()
    with pytest.raises(ValueError, match="Unknown goal"):
        fit(_f, x, y, p0=[1, 1], goal="bogus")


def test_goal_applies_tolerances_without_workflow(monkeypatch):
    """goal reaches curve_fit as ftol/xtol/gtol (and no 'goal' kwarg leaks)."""
    seen = {}
    import nlsq as pkg

    real = pkg.curve_fit

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(pkg, "curve_fit", spy)
    x, y = _data()
    fit(_f, x, y, p0=[1, 1], goal="quality")
    assert "goal" not in seen
    assert {"ftol", "xtol", "gtol"} <= set(seen)
    seen_fast = {}
    seen.clear()
    fit(_f, x, y, p0=[1, 1], goal="fast")
    assert seen["gtol"] > 0
    q = {k: v for k, v in seen.items() if k in ("ftol", "xtol", "gtol")}
    seen.clear()
    fit(_f, x, y, p0=[1, 1], goal="quality")
    assert all(seen[k] < q[k] for k in q)
    del seen_fast


def test_explicit_tolerance_wins_over_goal(monkeypatch):
    seen = {}
    import nlsq as pkg

    real = pkg.curve_fit
    monkeypatch.setattr(
        pkg, "curve_fit", lambda *a, **k: (seen.update(k), real(*a, **k))[1]
    )
    x, y = _data()
    fit(_f, x, y, p0=[1, 1], goal="fast", gtol=1e-3)
    assert seen["gtol"] == 1e-3


def test_unknown_preset_raises():
    x, y = _data()
    with (
        pytest.raises(ValueError, match="Unknown preset"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        fit(_f, x, y, p0=[1, 1], preset="bogus")


def test_preset_name_is_case_insensitive():
    x, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fit(_f, x, y, p0=[1, 1], preset="FAST")
    assert np.allclose(np.asarray(r[0]), [3.0, 0.7], rtol=0.1)


@pytest.mark.filterwarnings("ignore:The .preset. parameter:FutureWarning")
class TestStreamingPresetOptions:
    def test_full_output_and_return_eval_rejected(self):
        x, y = _data()
        for kw in ({"full_output": True}, {"return_eval": True}):
            with pytest.raises(ValueError, match="streaming"):
                fit(_f, x, y, p0=[1, 1], preset="streaming", verbose=0, **kw)

    def test_ftol_xtol_not_silently_dropped(self):
        x, y = _data()
        with pytest.warns(UserWarning, match="ftol"):
            fit(_f, x, y, p0=[1, 1], preset="streaming", verbose=0, ftol=1e-3)

    def test_max_nfev_and_gtol_reach_config(self, monkeypatch):
        from nlsq.streaming import strategy

        got = {}
        real = strategy.build_large_strategy

        def spy(kind, config, **kw):
            got["config"] = config
            return real(kind, config, **kw)

        monkeypatch.setattr(strategy, "build_large_strategy", spy)
        x, y = _data()
        fit(_f, x, y, p0=[1, 1], preset="streaming", verbose=0, max_nfev=3, gtol=1e-5)
        assert got["config"].gauss_newton_max_iterations == 3
        assert got["config"].gauss_newton_tol == 1e-5


@pytest.mark.filterwarnings("ignore:The .preset. parameter:FutureWarning")
def test_goal_with_streaming_preset_does_not_warn_about_ftol_xtol():
    x, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit(_f, x, y, p0=[1, 1], preset="streaming", verbose=0, goal="fast")
