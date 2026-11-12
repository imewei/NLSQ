"""Round-6 review fixes for the chunked large-dataset path (2026-11-11)."""

import contextlib
import logging
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large
from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig


def _gauss(x, a, mu, s, c):
    return a * jnp.exp(-((x - mu) ** 2) / (2 * s**2)) + c


def _gauss_data(n=100_000):
    x = np.linspace(-5, 5, n)
    rng = np.random.default_rng(0)
    y = np.asarray(_gauss(x, 3, 0.4, 0.8, 0.2)) + 0.05 * rng.standard_normal(n)
    return x, y


@pytest.mark.serial
class TestChunkConvergenceAccounting:
    """A chunk whose inner fit did not converge is a failed chunk."""

    def _fit(self, chunk_size, **cfg):
        x, y = _gauss_data()
        config = LDMemoryConfig(memory_limit_gb=1.0, max_chunk_size=chunk_size, **cfg)
        return LargeDatasetFitter(config=config).fit(
            _gauss, x, y, [1, -1, 2, 0], max_nfev=2
        )

    def test_single_chunk_reference_is_unsuccessful(self):
        assert not self._fit(200_000).success

    def test_all_chunks_nonconverged_is_unsuccessful(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._fit(10_000)
        assert not res.success

    def test_nonconverged_chunks_counted_in_success_rate(self):
        # min_success_rate=0 lets the combination finish, so the accounting
        # is visible: every chunk is recorded as failed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._fit(10_000, min_success_rate=0.0)
        assert res["success_rate"] == 0.0
        assert not res.success
        assert all(not r["success"] for r in res["chunk_results"])


@pytest.mark.serial
class TestFullOutputRejectedOnLargePath:
    @pytest.mark.parametrize("opt", ["full_output", "return_eval"])
    def test_rejected_up_front(self, opt):
        n = 200_001
        x = np.linspace(0, 3, n)
        y = 2.0 * np.exp(-1.3 * x) + 0.1

        def e(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        with pytest.raises((TypeError, ValueError), match=opt):
            curve_fit_large(e, x, y, p0=[1, 1, 0], size_threshold=1000, **{opt: True})


class TestTinyChunkWarning:
    def test_raised_chunk_size_warns(self, caplog):
        n = 10_000
        x = np.linspace(0, 3, n)
        y = 2.0 * np.exp(-1.3 * x) + 0.1

        def e(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        config = LDMemoryConfig(memory_limit_gb=1.0, min_chunk_size=1, max_chunk_size=2)
        with caplog.at_level(logging.WARNING), contextlib.suppress(Exception):
            LargeDatasetFitter(config=config).fit(e, x, y, [1, 1, 0])
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "requested chunk size 2" in msgs
        assert "effective" in msgs
