"""Round-5 security regression: private jax-profiling trace dir."""

from __future__ import annotations

import os
import stat

import pytest


def test_default_trace_dir_is_private_and_unpredictable(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    import tempfile

    from nlsq.utils.profiling import profile_transfers_runtime

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _, stats = profile_transfers_runtime(lambda: 1)
    d = stats.get("trace_dir")
    assert d is not None
    assert os.path.basename(d) != "jax-profiling"
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
