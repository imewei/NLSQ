"""Chunk pools must retain both independent and dependent data precision."""

import numpy as np
import pytest

from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig


@pytest.mark.parametrize("x_dtype", [np.int64, np.float32])
def test_integer_x_and_fractional_y_fit_through_chunk_pool(x_dtype):
    x = np.arange(40, dtype=x_dtype)
    y = 1.5 * np.asarray(x, dtype=np.float64) + 0.25

    def model(x, slope, intercept):
        return slope * x + intercept

    fitter = LargeDatasetFitter(
        config=LDMemoryConfig(min_chunk_size=10, max_chunk_size=10)
    )
    result = fitter.fit(model, x, y, p0=[1.0, 0.0])
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [1.5, 0.25], atol=1e-7)
    assert fitter._buffer_pool.dtype == np.dtype(np.float64)
