"""Public workflow=HybridStreamingConfig() streaming route: finite-check behaviour."""

from unittest.mock import patch

import numpy as np
import pytest

import nlsq
from nlsq.streaming.hybrid_config import HybridStreamingConfig


@pytest.mark.parametrize("invalid_data", ["x", "y"])
def test_public_streaming_checks_finite_before_engine(invalid_data):
    x = np.arange(4.0)
    y = 2 * x
    (x if invalid_data == "x" else y)[1] = np.nan
    with patch("nlsq.streaming.strategy.build_large_strategy") as build:
        build.return_value.fit.return_value = {
            "x": np.array([2.0]),
            "pcov": np.eye(1),
            "success": True,
        }
        with pytest.raises(ValueError, match=r"NaN|Inf|finite"):
            nlsq.fit(
                lambda x, a: a * x,
                x,
                y,
                p0=[1.0],
                workflow=HybridStreamingConfig(),
                check_finite=True,
            )
    build.assert_not_called()


def test_public_streaming_can_disable_finite_check():
    x = np.arange(4.0)
    y = 2 * x
    y[1] = np.nan
    with patch("nlsq.streaming.strategy.build_large_strategy") as build:
        build.return_value.fit.return_value = {
            "x": np.array([2.0]),
            "pcov": np.eye(1),
            "success": True,
        }
        nlsq.fit(
            lambda x, a: a * x,
            x,
            y,
            p0=[1.0],
            workflow=HybridStreamingConfig(),
            check_finite=False,
        )
    build.return_value.fit.assert_called_once()
