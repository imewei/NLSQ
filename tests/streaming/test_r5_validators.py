"""R5-03/05: integer coercion and exact_streaming bool max_nfev."""

import numpy as np
import pytest

from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.large_dataset import LDMemoryConfig
from nlsq.streaming.specs import Auto, Chunked, Exact, Multistart


def _is_int(v):
    return type(v) is int


def test_hybrid_config_integral_floats_become_int():
    cfg = HybridStreamingConfig(
        chunk_size=5000.0, warmup_iterations=20.0, n_starts=4.0, gc_chunk_interval=3.0
    )
    assert all(
        _is_int(getattr(cfg, n))
        for n in ("chunk_size", "warmup_iterations", "n_starts", "gc_chunk_interval")
    )
    assert cfg.chunk_size == 5000


def test_hybrid_config_fractional_still_rejected():
    with pytest.raises(ValueError):
        HybridStreamingConfig(chunk_size=5000.5)


def test_ld_memory_config_integral_floats_become_int():
    cfg = LDMemoryConfig(min_chunk_size=10.0, max_chunk_size=100.0)
    assert _is_int(cfg.min_chunk_size)
    assert _is_int(cfg.max_chunk_size)


def test_specs_integral_floats_become_int():
    assert _is_int(Multistart(n_starts=3.0).n_starts)
    assert _is_int(Chunked(chunk_size=100.0).chunk_size)
    assert _is_int(Exact(chunk_size=100.0).chunk_size)
    assert _is_int(Auto(size_threshold=1000.0).size_threshold)


@pytest.mark.parametrize("bad", [True, np.bool_(True)])
def test_exact_rejects_bool_max_nfev(bad):
    with pytest.raises(ValueError, match="max_nfev"):
        Exact(max_nfev=bad)


def test_curve_fit_large_exact_streaming_rejects_bool_max_nfev():
    import jax.numpy as jnp

    from nlsq import curve_fit_large

    x = np.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError, match="max_nfev"):
        curve_fit_large(
            lambda x, a, b: a * jnp.exp(-b * x),
            x,
            2.0 * np.exp(-x),
            p0=[1.0, 1.0],
            method="exact_streaming",
            max_nfev=True,
        )
