"""LargeDatasetFitter flags results that are not full-batch-equivalent."""

import logging

import jax.numpy as jnp
import numpy as np

from nlsq.streaming.large_dataset import LargeDatasetFitter, fit_large_dataset

FLAG = "equivalent_to_full_batch"
WARNING = "not equivalent to a full-batch fit"


def _model(x, a, b):
    return a * x + b


def _data(n=5000):
    x = np.linspace(0.0, 4.0, n)
    return x, 2.0 * x + 1.0 + np.random.default_rng(0).normal(0, 0.01, n)


def _fitter(memory_limit_gb, caplog):
    logger = logging.getLogger("test_equivalence_flag")
    caplog.set_level(logging.WARNING, logger=logger.name)
    return LargeDatasetFitter(memory_limit_gb=memory_limit_gb, logger=logger)


def _flag_records(caplog):
    return [r for r in caplog.records if WARNING in r.getMessage()]


def test_single_chunk_fit_is_equivalent(caplog):
    x, y = _data()
    result = _fitter(8.0, caplog).fit(_model, x, y, p0=[1.0, 0.0])

    assert result[FLAG] is True
    assert not _flag_records(caplog)


def test_multi_chunk_fit_is_flagged(caplog):
    x, y = _data()
    result = _fitter(0.0001, caplog).fit(_model, x, y, p0=[1.0, 0.0])

    assert result.success
    assert result[FLAG] is False
    assert len(_flag_records(caplog)) == 1


def test_failed_multi_chunk_fit_is_flagged(caplog):
    x, y = _data()
    y[::2] = np.nan  # every chunk sees non-finite data and fails
    result = _fitter(0.0001, caplog).fit(_model, x, y, p0=[1.0, 0.0])

    assert not result.success
    assert result[FLAG] is False
    assert len(_flag_records(caplog)) == 1


def test_public_entry_point_keeps_flag(caplog):
    x, y = _data()
    caplog.set_level(logging.WARNING)
    result = fit_large_dataset(
        lambda x, a, b: a * jnp.asarray(x) + b,
        x,
        y,
        p0=[1.0, 0.0],
        memory_limit_gb=0.0001,
        logger=logging.getLogger("test_equivalence_flag_public"),
    )

    assert result[FLAG] is False
