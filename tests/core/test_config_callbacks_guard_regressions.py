"""Live x64 flag, provided-file logger and non-finite Hessian regressions."""

import io

import jax
import numpy as np
import pytest

from nlsq.callbacks import IterationLogger
from nlsq.config import JAXConfig


@pytest.mark.serial
def test_x64_controls_follow_live_jax_flag():
    original = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", False)  # outside change
        assert JAXConfig.is_x64_enabled() is False
        JAXConfig.enable_x64(True)
        assert bool(jax.config.jax_enable_x64) is True
        jax.config.update("jax_enable_x64", False)
        with JAXConfig.precision_context(True):
            assert bool(jax.config.jax_enable_x64) is True
        assert bool(jax.config.jax_enable_x64) is False
    finally:
        jax.config.update("jax_enable_x64", original)
        JAXConfig.is_x64_enabled()


def test_iteration_logger_keeps_provided_file_after_close():
    buf = io.StringIO()
    cb = IterationLogger(file=buf)
    cb.close()
    cb._write("second fit line")
    assert "second fit line" in buf.getvalue()


def test_regularize_hessian_warns_on_nonfinite():
    import jax.numpy as jnp

    from nlsq.stability.guard import NumericalStabilityGuard

    with pytest.warns(UserWarning, match="regularize_hessian"):
        NumericalStabilityGuard().regularize_hessian(
            jnp.array([[np.nan, 0.0], [0.0, 1.0]])
        )
