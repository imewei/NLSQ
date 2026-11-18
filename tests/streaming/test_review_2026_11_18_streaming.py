"""Regressions from the 2026-11-18 three-brain review of nlsq/streaming."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large
from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig
from nlsq.streaming.specs import Auto, Chunked, Exact


def _const(x, a):
    return a + 0 * x


def test_exact_factory_reusing_its_buffer_keeps_the_first_block():
    """The retained probe block must be a copy: the factory overwrites its buffer."""
    bx, by = np.ones(1), np.zeros(1)

    def factory(chunk):
        for r in (0.0, 3.0, 3.0):
            by[0] = r
            yield bx, by

    popt, pcov = curve_fit_large(_const, factory, None, p0=[3.0], strategy=Exact(1))[:2]
    np.testing.assert_allclose(popt, [2.0])
    np.testing.assert_allclose(pcov, [[1.0]])


def test_direct_fitter_rejects_extra_ydata():
    fitter = LargeDatasetFitter(config=LDMemoryConfig(max_chunk_size=3))
    with pytest.raises(ValueError, match="same length"):
        fitter.fit(
            _const, np.arange(1.0, 7.0), np.array([1, 1, 1, 1, 1, 1, 99.0]), p0=[1.0]
        )


def test_auto_probe_projects_p0_into_bounds():
    """A model pointwise on the feasible box must not be rejected at an infeasible p0."""

    def model(x, a):
        return jnp.where(a < 0, jnp.mean(x), a * x)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    popt = curve_fit_large(
        model, x, 2 * x, p0=[-1.0], bounds=(0, 4), strategy=Auto(size_threshold=0)
    )[0]
    np.testing.assert_allclose(popt, [2.0], rtol=1e-8)


def test_direct_fitter_accepts_scalar_p0():
    fitter = LargeDatasetFitter(config=LDMemoryConfig(max_chunk_size=2))
    result = fitter.fit(_const, np.arange(1.0, 5.0), np.ones(4), p0=1.0)
    np.testing.assert_allclose(result.popt, [1.0])
    popt = curve_fit_large(
        _const, np.arange(1.0, 5.0), np.ones(4), p0=1.0, strategy=Chunked(chunk_size=2)
    )[0]
    np.testing.assert_allclose(popt, [1.0])


def test_failed_chunked_result_keeps_chunk_records():
    fitter = LargeDatasetFitter(config=LDMemoryConfig(max_chunk_size=3))
    x = np.arange(1.0, 7.0)
    result = fitter.fit(lambda x, a: a * x, x, 2 * x, p0=[1.0], max_nfev=1)
    assert not result.success
    assert len(result["chunk_results"]) == 2
    assert result["failure_summary"]["failed_chunk_indices"] == [0, 1]


def test_memory_limit_none_autodetects():
    fitter = LargeDatasetFitter(memory_limit_gb=None)
    assert fitter.config.memory_limit_gb > 0


@pytest.mark.parametrize("flag", ["timeit", "full_output", "return_eval"])
def test_chunked_route_rejects_output_altering_options_up_front(flag):
    """timeit made curve_fit return a 3-tuple: every chunk failed to unpack."""
    x = np.arange(1.0, 7.0)
    with pytest.raises(TypeError, match=flag):
        curve_fit_large(
            lambda x, a: a * x,
            x,
            2 * x,
            p0=[1.0],
            strategy=Chunked(chunk_size=3, options={flag: True}),
        )
    fitter = LargeDatasetFitter(config=LDMemoryConfig(max_chunk_size=3))
    with pytest.raises(TypeError, match=flag):
        fitter.fit(lambda x, a: a * x, x, 2 * x, p0=[1.0], **{flag: True})


def test_lbfgs_step_sizes_must_follow_documented_hierarchy():
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    HybridStreamingConfig()  # defaults: 0.1 <= 0.5 <= 1.0
    with pytest.raises(ValueError, match=r"lbfgs_\w+_step_size"):
        HybridStreamingConfig(
            lbfgs_exploration_step_size=2.0, lbfgs_refinement_step_size=0.01
        )
