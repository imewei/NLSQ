"""Cache isolation must survive NLSQ imports during pytest collection."""

import os
from pathlib import Path

import jax
import pytest

from nlsq.config import JAXConfig

pytestmark = pytest.mark.serial

# Real test modules initialize NLSQ configuration before session fixtures run.
JAXConfig()

# Captured at import (collection) time, before any fixture can touch the config.
# Deliberately do not request configure_jax_for_tests: it would set both values
# itself and mask a missing module-level assignment in tests/conftest.py.
_ENV_AT_IMPORT = os.environ.get("JAX_COMPILATION_CACHE_DIR")
_JAX_DIR_AT_IMPORT = jax.config.jax_compilation_cache_dir


def test_collection_time_cache_dir_is_isolated():
    assert _ENV_AT_IMPORT is not None
    assert _JAX_DIR_AT_IMPORT == _ENV_AT_IMPORT
    assert not Path(_JAX_DIR_AT_IMPORT).is_relative_to(Path.home() / ".cache" / "nlsq")
