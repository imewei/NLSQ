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
