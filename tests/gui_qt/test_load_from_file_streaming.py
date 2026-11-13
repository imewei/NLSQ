"""load_from_file streams file-like input instead of read()-ing it whole."""

import io

import numpy as np

from nlsq.gui_qt.adapters.data_adapter import load_from_file

CONFIG = {"format": "csv", "columns": {"x": 0, "y": 1}}


class _NoReadAll(io.BytesIO):
    def read(self, size=-1):
        assert size is not None and size > 0, "whole-input read() is not allowed"
        return super().read(size)


def test_file_like_input_is_streamed_and_rewound():
    buf = _NoReadAll(b"1,2\n3,4\n5,6\n")
    x, y, sigma = load_from_file(buf, CONFIG)
    np.testing.assert_array_equal(x, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(y, [2.0, 4.0, 6.0])
    assert sigma is None
    assert buf.tell() == 0
