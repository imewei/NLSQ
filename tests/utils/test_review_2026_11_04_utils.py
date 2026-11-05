"""Regression tests for the 2026-11-04 review of nlsq/utils and diagnostics."""

import json
import subprocess
import sys

import numpy as np
import pytest

from nlsq import CurveFit
from nlsq.diagnostics.parameter_sensitivity import ParameterSensitivityAnalyzer
from nlsq.utils.diagnostics import OptimizationDiagnostics
from nlsq.utils.safe_serialize import SafeSerializationError, safe_dumps, safe_loads
from nlsq.utils.validators import InputValidator


def _model_with_default(x, a, b, c=0.0):
    return a * x + b + c


def test_defaulted_param_not_counted_with_stability():
    x = np.linspace(0, 1, 10)
    y = 2 * x + 1
    popt, _ = CurveFit(enable_stability=True).curve_fit(
        _model_with_default, x, y, p0=[1.0, 1.0]
    )
    np.testing.assert_allclose(popt, [2.0, 1.0], atol=1e-6)


def test_estimate_n_params_without_p0_skips_defaults_and_kwargs():
    def f(x, a, b, c=0.0, **kw):
        return a * x + b

    assert InputValidator()._estimate_n_params(f, None) == 2


def test_m_equals_n_accepted_by_validator():
    errors, _ = InputValidator()._validate_data_shapes(2, np.zeros(2), 2)
    assert errors == []


def test_zero_jacobian_column_not_healthy():
    rng = np.random.default_rng(0)
    J = np.column_stack([rng.normal(size=20), np.zeros(20)])
    report = ParameterSensitivityAnalyzer().analyze(J)
    assert report.health_status.name != "HEALTHY"
    assert report.sloppy_indices == [1]


def test_wide_jacobian_not_healthy():
    J = np.random.default_rng(1).normal(size=(2, 4))
    report = ParameterSensitivityAnalyzer().analyze(J)
    assert report.health_status.name != "HEALTHY"


@pytest.mark.parametrize("verbosity", [1, 2])
def test_rank_deficient_and_nonsquare_jacobian_flagged(verbosity):
    mon = OptimizationDiagnostics(verbosity=verbosity)
    J = np.column_stack([np.ones(10), np.ones(10)])  # rank 1, non-square
    mon.record_iteration(
        0, np.array([1.0, 1.0]), 1.0, gradient=np.array([0.1, 0.1]), jacobian=J
    )
    assert any("Ill-conditioned" in s for s in mon.numerical_issues)


@pytest.mark.parametrize(
    "payload",
    [
        {"__nlsq_type__": "tuple"},
        {"__nlsq_type__": "float"},
        {"__nlsq_type__": "ndarray", "dtype": "foo", "shape": [1], "data": [1]},
        {"__nlsq_type__": "ndarray", "dtype": "float64", "shape": 3, "data": [1]},
        {"__nlsq_type__": "tuple", "value": 5},
    ],
)
def test_safe_loads_malformed_raises_safe_error(payload):
    with pytest.raises(SafeSerializationError):
        safe_loads(json.dumps(payload).encode())


def test_safe_dumps_rejects_string_array():
    with pytest.raises(SafeSerializationError):
        safe_dumps(np.array(["a", "b"]))


def test_bool_array_requires_bool_data():
    payload = {
        "__nlsq_type__": "ndarray",
        "dtype": "bool",
        "shape": [1],
        "data": ["False"],
    }
    with pytest.raises(SafeSerializationError):
        safe_loads(json.dumps(payload).encode())


def test_noncanonical_and_duplicate_int_keys_rejected():
    with pytest.raises(SafeSerializationError):
        safe_loads(b'{"__int_key__1": "a", "__int_key__01": "b"}')
    with pytest.raises(SafeSerializationError):
        safe_loads(b'{"__int_key__01": "b"}')
    assert safe_loads(safe_dumps({1: "a", -2: "b"})) == {1: "a", -2: "b"}


def test_nested_log_dir_created(tmp_path):
    target = tmp_path / "a" / "b"
    code = "import nlsq"
    env = {"NLSQ_DEBUG": "1", "NLSQ_LOG_DIR": str(target), "PATH": ""}
    import os

    res = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert target.is_dir()
