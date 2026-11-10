"""R4 CC-08 / CC-13 / CC-15: streaming config and strategy-spec validation."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.specs import Auto, Chunked, Exact, InMemory, Multistart


def test_mutated_hybrid_config_is_revalidated_at_fit():
    cfg = HybridStreamingConfig(max_warmup_iterations=5, warmup_iterations=5)
    cfg.chunk_size = -1
    opt = AdaptiveHybridStreamingOptimizer(cfg)
    x = np.linspace(0, 1, 50)
    y = 2.0 * x + 1.0
    with pytest.raises(ValueError, match="chunk_size"):
        opt.fit((x, y), lambda x, a, b: a * x + b, jnp.array([1.0, 1.0]), verbose=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regularization_factor": float("inf")},
        {"cg_param_threshold": 2.9},
        {"cg_param_threshold": True},
        {"verbose": -1},
        {"verbose": 2.9},
        {"verbose": float("nan")},
        {"verbose": True},
    ],
)
def test_hybrid_config_minor_leftovers_rejected(kwargs):
    with pytest.raises(ValueError):
        HybridStreamingConfig(**kwargs)


@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_hybrid_config_valid_verbose(verbose):
    assert HybridStreamingConfig(verbose=verbose).verbose == verbose


@pytest.mark.parametrize("cls", [Chunked, Exact])
@pytest.mark.parametrize("bad", [0, -10, 2.5, True, float("nan")])
def test_chunk_size_validated_at_construction(cls, bad):
    with pytest.raises(ValueError, match="chunk_size"):
        cls(chunk_size=bad)


@pytest.mark.parametrize("cls", [Chunked, Exact])
def test_chunk_size_none_and_positive_accepted(cls):
    cls()
    cls(chunk_size=1000)


@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_auto_size_threshold_validated(bad):
    with pytest.raises(ValueError, match="size_threshold"):
        Auto(size_threshold=bad)


def test_auto_size_threshold_zero_accepted():
    Auto(size_threshold=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_starts": -1},
        {"n_starts": 2.5},
        {"sampler": "sobal"},
        {"scale_factor": 0.0},
        {"scale_factor": float("nan")},
    ],
)
def test_multistart_validated(kwargs):
    with pytest.raises(ValueError):
        Multistart(**kwargs)


def test_multistart_defaults_and_zero_accepted():
    Multistart()
    Multistart(n_starts=0)


@pytest.mark.parametrize("bad", [True, "Auto", "bogus"])
def test_in_memory_stability_validated(bad):
    with pytest.raises(ValueError, match="stability"):
        InMemory(stability=bad)
