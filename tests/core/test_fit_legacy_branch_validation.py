"""R8-3: nlsq.fit() (workflow=None) goal handling."""

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


def test_large_dataset_branch_defaults(monkeypatch):
    """At or above size_threshold: curve_fit_large, multistart off unless asked, 5 starts."""
    import nlsq as pkg

    seen = {}

    def spy(*a, **k):
        seen.update(k)
        return "large"

    monkeypatch.setattr(pkg, "curve_fit_large", spy)
    x, y = _data(400)
    assert fit(_f, x, y, p0=[1, 1], size_threshold=100) == "large"
    assert seen["multistart"] is False and seen["n_starts"] == 5
    seen.clear()
    fit(_f, x, y, p0=[1, 1], size_threshold=100, multistart=True, n_starts=7)
    assert seen["multistart"] is True and seen["n_starts"] == 7


@pytest.mark.parametrize(
    ("n_points", "threshold", "backend"),
    [
        (99, 100, "curve_fit"),
        (100, 100, "curve_fit_large"),
        (101, 100, "curve_fit_large"),
    ],
)
def test_size_threshold_boundary_is_inclusive(
    monkeypatch, n_points, threshold, backend
):
    """n_points >= size_threshold goes to curve_fit_large; one below stays on curve_fit."""
    import nlsq as pkg

    monkeypatch.setattr(pkg, "curve_fit", lambda *a, **k: "curve_fit")
    monkeypatch.setattr(pkg, "curve_fit_large", lambda *a, **k: "curve_fit_large")
    x, y = _data(n_points)
    assert fit(_f, x, y, p0=[1, 1], size_threshold=threshold) == backend


def test_options_after_workflow_are_keyword_only():
    """Removing `preset` must not shift later positionals: an old positional
    preset (11th argument) now fails loudly instead of landing in `multistart`."""
    x, y = _data(50)
    with pytest.raises(TypeError, match="positional"):
        fit(_f, x, y, [1, 1], None, False, True, (-np.inf, np.inf), None, None, "fast")
