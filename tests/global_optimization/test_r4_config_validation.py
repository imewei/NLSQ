"""R4 CC-02 / CC-09: CMAESConfig and GlobalOptimizationConfig reject hostile values."""

import numpy as np
import pytest

from nlsq.global_optimization.cmaes_config import CMAESConfig
from nlsq.global_optimization.config import GlobalOptimizationConfig


@pytest.mark.parametrize("name", ["sigma", "tol_fun", "tol_x"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_cmaes_non_finite_floats_rejected(name, bad):
    with pytest.raises(ValueError, match=name):
        CMAESConfig(**{name: bad})


@pytest.mark.parametrize(
    "name,bad",
    [
        ("popsize", 5.5),
        ("popsize", float("nan")),
        ("max_generations", 2.9),
        ("max_generations", True),
        ("max_restarts", 2.5),
        ("population_batch_size", 2.5),
        ("data_chunk_size", 2000.5),
        ("checkpoint_interval", 2.9),
        ("checkpoint_interval", True),
    ],
)
def test_cmaes_non_integral_counts_rejected(name, bad):
    with pytest.raises(ValueError, match=name):
        CMAESConfig(**{name: bad})


def test_cmaes_integral_values_still_accepted():
    cfg = CMAESConfig(
        popsize=np.int64(8),
        max_generations=100.0,
        max_restarts=np.int32(0),
        data_chunk_size=2048,
    )
    assert cfg.max_generations == 100


def test_with_overrides_unknown_field_raises():
    with pytest.raises(TypeError, match="n_start"):
        GlobalOptimizationConfig().with_overrides(n_start=50)


def test_roundtrip_and_valid_overrides():
    cfg = GlobalOptimizationConfig.from_preset("robust")
    assert GlobalOptimizationConfig.from_dict(cfg.to_dict()).n_starts == cfg.n_starts
    assert cfg.with_overrides(n_starts=7).n_starts == 7


def _is_int(v):
    return type(v) is int


def test_cmaes_config_integral_floats_become_int():
    cfg = CMAESConfig(max_generations=20.0, popsize=8.0, seed=3.0)
    assert _is_int(cfg.max_generations)
    assert _is_int(cfg.popsize)
    assert _is_int(cfg.seed)
    assert _is_int(CMAESConfig(seed=np.int64(3)).seed)


def test_cmaes_config_rejects_bad_seed():
    with pytest.raises(ValueError):
        CMAESConfig(seed=1.5)
    with pytest.raises(ValueError):
        CMAESConfig(seed=True)
