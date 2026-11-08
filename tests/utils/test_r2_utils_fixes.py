"""Regression tests for R2 utils findings (history filename / env flag, tuple marker)."""

import numpy as np
import pytest

from nlsq.utils.logging import NLSQLogger
from nlsq.utils.safe_serialize import SafeSerializationError, safe_loads


def _logger_with_history(name, costs):
    lg = NLSQLogger(name)
    lg.optimization_history = [{"iter": i, "cost": c} for i, c in enumerate(costs)]
    return lg


def test_same_second_saves_do_not_overwrite(tmp_path):
    _logger_with_history("r2a", [1.0]).save_iteration_data(str(tmp_path))
    _logger_with_history("r2a", [2.0, 3.0]).save_iteration_data(str(tmp_path))
    assert len(list(tmp_path.glob("optimization_history_*r2a_*.npz"))) == 2


@pytest.mark.parametrize("val", ["0", "false", "No", "OFF", ""])
def test_save_iterations_disabled_values(tmp_path, monkeypatch, val):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", val)
    _logger_with_history("r2b", [1.0]).save_iteration_data()
    assert not list(tmp_path.rglob("*.npz"))


def test_save_iterations_directory_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path / "h"))
    _logger_with_history("r2c", [1.0]).save_iteration_data()
    assert list((tmp_path / "h").glob("*.npz"))


@pytest.mark.parametrize("bad", ['"abc"', "1", "null", '{"a": 1}'])
def test_malformed_tuple_marker_rejected(bad):
    with pytest.raises(SafeSerializationError):
        safe_loads(('{"__nlsq_type__":"tuple","value":%s}' % bad).encode())


def test_valid_tuple_roundtrip():
    from nlsq.utils.safe_serialize import safe_dumps

    assert safe_loads(safe_dumps({"t": (1, 2.5)})) == {"t": (1, 2.5)}
