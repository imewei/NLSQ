"""np.ma.MaskedArray cannot round-trip; it must be rejected, not half-saved."""

import numpy as np
import pytest

from nlsq.utils.safe_serialize import SafeSerializationError, safe_dumps, safe_loads


@pytest.mark.parametrize("dtype", [float, int])
def test_masked_array_rejected(dtype):
    arr = np.ma.masked_array(np.arange(4, dtype=dtype), mask=[0, 1, 0, 0])
    with pytest.raises(SafeSerializationError, match="masked"):
        safe_dumps({"a": arr})


def test_plain_ndarray_round_trips():
    arr = np.arange(4.0)
    np.testing.assert_array_equal(safe_loads(safe_dumps({"a": arr}))["a"], arr)
