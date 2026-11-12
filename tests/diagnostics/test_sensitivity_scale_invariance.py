"""Sensitivity dimensionality must not depend on an overall unit scale."""

import numpy as np
import pytest

from nlsq.diagnostics import HealthStatus
from nlsq.diagnostics.parameter_sensitivity import ParameterSensitivityAnalyzer

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("from_fim", [True, False])
def test_equal_sensitivity_directions_remain_two_dimensional(scale, from_fim):
    """Squaring unnormalized eigenvalues must not erase two equal directions."""
    analyzer = ParameterSensitivityAnalyzer()
    if from_fim:
        report = analyzer.analyze_from_fim(np.eye(2) * scale)
    else:
        report = analyzer.analyze(np.eye(2) * np.sqrt(scale))

    assert report.available
    assert report.effective_dimensionality == pytest.approx(2.0)
    assert report.health_status is HealthStatus.HEALTHY


@pytest.mark.parametrize("scale", [1e-200, 1e-170, 1e160, 1e200])
def test_jacobian_metrics_invariant_to_extreme_scale(scale):
    """s**2 underflow/overflow must not zero out range / effective dimension."""
    from nlsq.diagnostics import DiagnosticsConfig

    jac = np.c_[np.linspace(0, 1, 100), np.ones(100)]
    analyzer = ParameterSensitivityAnalyzer(DiagnosticsConfig())
    ref = analyzer.analyze(jac)
    got = analyzer.analyze(scale * jac)

    assert got.available
    assert got.eigenvalue_range == pytest.approx(ref.eigenvalue_range, rel=1e-6)
    assert got.effective_dimensionality == pytest.approx(
        ref.effective_dimensionality, rel=1e-6
    )
    assert got.health_status is ref.health_status
    assert got.stiff_indices == ref.stiff_indices
    assert got.sloppy_indices == ref.sloppy_indices
    assert [i.code for i in got.issues] == [i.code for i in ref.issues]
