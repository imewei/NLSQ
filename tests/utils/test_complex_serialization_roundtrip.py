"""All numeric array dtypes admitted by the serializer must round-trip."""

import numpy as np
import pytest

from nlsq.utils.safe_serialize import safe_dumps, safe_loads


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_array_roundtrip(dtype):
    original = np.array([[1 + 2j, 3 - 4j]], dtype=dtype)
    restored = safe_loads(safe_dumps(original))
    assert restored.dtype == original.dtype
    np.testing.assert_array_equal(restored, original)
