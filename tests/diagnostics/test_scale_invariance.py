"""Unit-invariance and JSON-safety regressions for the diagnostics (GD-1/2/3)."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.diagnostics import DiagnosticsConfig, GradientMonitor
from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer
from nlsq.diagnostics.parameter_sensitivity import ParameterSensitivityAnalyzer
from nlsq.diagnostics.types import HealthStatus

pytestmark = pytest.mark.diagnostics


@pytest.fixture
def jac() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((50, 3))


@pytest.mark.parametrize("scale", [[1e7, 1, 1e-7], [1e3, 1, 1e-3], [1e-9, 1e4, 1]])
def test_identifiability_invariant_to_column_units(jac, scale) -> None:
    a = IdentifiabilityAnalyzer(DiagnosticsConfig())
    base, scaled = a.analyze(jac), a.analyze(jac * scale)
    assert scaled.health_status == HealthStatus.HEALTHY
    assert scaled.numerical_rank == 3
    assert scaled.issues == []
    np.testing.assert_allclose(
        scaled.condition_number, base.condition_number, rtol=1e-6
    )
    np.testing.assert_allclose(
        scaled.correlation_matrix, base.correlation_matrix, atol=1e-8
    )


def test_identifiability_fim_invariant_to_units(jac) -> None:
    a = IdentifiabilityAnalyzer(DiagnosticsConfig())
    d = np.array([1e7, 1.0, 1e-7])
    base = a.analyze_from_fim(jac.T @ jac)
    scaled = a.analyze_from_fim((jac * d).T @ (jac * d))
    assert scaled.numerical_rank == 3
    np.testing.assert_allclose(
        scaled.condition_number, base.condition_number, rtol=1e-6
    )


def test_zero_column_stays_structurally_unidentifiable(jac) -> None:
    jac[:, 1] = 0.0
    report = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze(jac * [1e7, 1, 1e-7])
    assert report.numerical_rank == 2
    assert any(i.code == "IDENT-001" for i in report.issues)


def test_sensitivity_invariant_to_column_units(jac) -> None:
    a = ParameterSensitivityAnalyzer(DiagnosticsConfig())
    base, scaled = a.analyze(jac), a.analyze(jac * [1e3, 1, 1e-3])
    assert scaled.is_sloppy is base.is_sloppy is False
    np.testing.assert_allclose(
        scaled.eigenvalue_range, base.eigenvalue_range, atol=1e-6
    )
    assert scaled.issues == []


def test_sensitivity_fim_invariant_to_units(jac) -> None:
    a = ParameterSensitivityAnalyzer(DiagnosticsConfig())
    d = np.array([1e3, 1.0, 1e-3])
    base = a.analyze_from_fim(jac.T @ jac)
    scaled = a.analyze_from_fim((jac * d).T @ (jac * d))
    np.testing.assert_allclose(
        scaled.eigenvalue_range, base.eigenvalue_range, atol=1e-6
    )
    assert scaled.is_sloppy is base.is_sloppy


def test_exploding_is_relative_not_absolute() -> None:
    m = GradientMonitor(DiagnosticsConfig(exploding_threshold=1e8))
    for _ in range(15):
        m.record_gradient(np.array([1e12, 1e12]), cost=1.0)
    assert m.get_report().exploding_detected is False


def test_imbalance_is_invariant_to_parameter_units() -> None:
    cfg = DiagnosticsConfig(imbalance_threshold=1e6)
    g, p = np.array([1e9, 1.0]), np.array([1e-9, 1.0])  # g_j * p_j balanced
    scaled = GradientMonitor(cfg)
    scaled.record_gradient(g, cost=1.0, params=p)
    assert scaled.get_report().imbalance_detected is False
    raw = GradientMonitor(cfg)
    raw.record_gradient(g, cost=1.0)  # no params: raw ratio 1e9
    assert raw.get_report().imbalance_detected is True


def test_detectors_return_python_bool() -> None:
    m = GradientMonitor(DiagnosticsConfig())
    norms = [float(np.float64(1.0))] * 20
    for r in (
        m._detect_vanishing_gradients(norms, norms),
        m._detect_gradient_imbalance(),
        m._detect_gradient_stagnation(norms),
        m._detect_exploding_gradients(norms),
    ):
        assert type(r) is bool


@pytest.mark.parametrize("tau_scale", [1.0, 1e-6])
def test_rescaled_wellposed_fit_is_healthy_and_json_safe(tau_scale) -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(0, 5, 200) * tau_scale
    y = 1e3 * np.exp(-x / tau_scale) + rng.normal(0, 1.0, x.size)

    def model(x, a, tau):
        return a * jnp.exp(-x / tau)

    r = curve_fit(model, x, y, p0=[900.0, 1.2 * tau_scale], compute_diagnostics=True)
    assert r.diagnostics.status == HealthStatus.HEALTHY
    assert r.diagnostics.identifiability.issues == []
    json.dumps(r.diagnostics.to_dict())
