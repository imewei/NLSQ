"""Cache isolation must survive NLSQ imports during pytest collection."""

import os

import jax
import pytest

from nlsq.config import JAXConfig

pytestmark = pytest.mark.serial

# Real test modules initialize NLSQ configuration before session fixtures run.
JAXConfig()


def test_worker_cache_configuration_reaches_jax(configure_jax_for_tests):
    assert (
        jax.config.jax_compilation_cache_dir == os.environ["JAX_COMPILATION_CACHE_DIR"]
    )
