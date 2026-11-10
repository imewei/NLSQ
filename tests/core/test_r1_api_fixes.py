"""Regression tests for round-1 review findings in callbacks/result/curve_fit."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit, curve_fit
from nlsq.callbacks import IterationLogger, ProgressBar


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data():
    x = np.linspace(0, 2, 30)
    return x, 2.0 * np.exp(-1.3 * x)


def test_iteration_logger_reusable_after_close(tmp_path):
    path = tmp_path / "log.txt"
    cb = IterationLogger(filename=str(path), mode="a")
    info = {"nfev": 1, "gradient_norm": 0.1}
    cb(0, 1.0, np.array([1.0]), info)
    cb.close()
    assert cb._start_time is None
    cb(0, 1.0, np.array([1.0]), info)  # must reopen the file, not print
    cb.close()
    assert path.read_text().count("Iter") >= 2


def test_progress_bar_close_resets_last_nfev():
    pytest.importorskip("tqdm")
    cb = ProgressBar(max_nfev=100)
    cb(0, 1.0, np.array([1.0]), {"nfev": 8, "gradient_norm": 0.1})
    cb.close()
    assert cb._last_nfev == 0


def test_result_getitem_accepts_numpy_integer():
    x, y = _data()
    r = curve_fit(_model, x, y, p0=[1.0, 1.0])
    np.testing.assert_array_equal(r[np.int64(0)], r[0])
    np.testing.assert_array_equal(r[np.int32(1)], r[1])


def test_tuple_return_modes_give_numpy_popt():
    x, y = _data()
    out = CurveFit().curve_fit(_model, x, y, p0=[1.0, 1.0], full_output=True)
    assert isinstance(out[0], np.ndarray)
    out = CurveFit().curve_fit(_model, x, y, p0=[1.0, 1.0], return_eval=True)
    assert isinstance(out[0], np.ndarray)
    out = CurveFit().curve_fit(_model, x, y, p0=[1.0, 1.0], timeit=True)
    assert isinstance(out[0], np.ndarray)
