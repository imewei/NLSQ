"""R2 streaming validation regressions (S2-1..S2-3)."""

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.streaming.exact_lm as lm
from nlsq.streaming.hybrid_config import HybridStreamingConfig


def _model(x, a, b):
    return a * jnp.exp(-b * x)


@pytest.fixture(scope="module")
def data():
    x = np.linspace(0, 3, 200)
    return x, 2.0 * np.exp(-1.3 * x)


def _fit(data, **kw):
    return lm.exact_lm_fit(_model, *data, [1.0, 1.0], chunk_size=64, **kw)


# ---- S2-1: tolerances ----------------------------------------------------------------
@pytest.mark.parametrize("name", ["ftol", "xtol", "gtol"])
@pytest.mark.parametrize("bad", [float("nan"), -1.0, float("inf")])
def test_exact_rejects_bad_tolerance(data, name, bad):
    with pytest.raises(ValueError, match=name):
        _fit(data, **{name: bad})


def test_exact_rejects_all_tolerances_below_eps(data):
    with pytest.raises(ValueError, match="machine epsilon"):
        _fit(data, ftol=0.0, xtol=0.0, gtol=0.0)


def test_exact_accepts_tight_tolerances(data):
    assert _fit(data, ftol=1e-12, xtol=1e-12, gtol=1e-12).x.shape == (2,)
    assert _fit(data, ftol=0.0, xtol=0.0).x.shape == (2,)  # gtol keeps it valid


# ---- S2-2: memory_limit_gb -----------------------------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), -1.0, 0.0, float("inf")])
def test_exact_rejects_bad_memory_limit(data, bad):
    with pytest.raises(ValueError, match="memory_limit_gb must be positive"):
        _fit(data, memory_limit_gb=bad)
    with pytest.raises(ValueError, match="memory_limit_gb must be positive"):
        lm.exact_lm_fit(_model, *data, [1.0, 1.0], memory_limit_gb=bad)


# ---- S2-3: NaN/inf counts in HybridStreamingConfig -----------------------------------
@pytest.mark.parametrize(
    "kw",
    [
        {"max_retries_per_batch": float("inf")},
        {"max_retries_per_batch": float("nan")},
        {"use_learning_rate_schedule": True, "lr_schedule_warmup_steps": float("inf")},
        {"use_learning_rate_schedule": True, "lr_schedule_warmup_steps": float("nan")},
        {"gc_chunk_interval": float("nan")},
        {"gc_chunk_interval": float("inf")},
        {"enable_multistart": True, "n_starts": float("nan")},
        {"enable_multistart": True, "n_starts": float("inf")},
        {"enable_multistart": True, "elimination_rounds": float("inf")},
        {"enable_multistart": True, "elimination_rounds": float("nan")},
    ],
)
def test_config_rejects_nonfinite_counts(kw):
    with pytest.raises(ValueError):
        HybridStreamingConfig(**kw)


def test_config_still_accepts_valid_counts():
    HybridStreamingConfig(
        max_retries_per_batch=0,
        lr_schedule_warmup_steps=0,
        gc_chunk_interval=1,
        enable_multistart=True,
        n_starts=3,
        elimination_rounds=0,
    )
