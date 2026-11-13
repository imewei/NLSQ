import numpy as np
import pytest

from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer
from nlsq.diagnostics.parameter_sensitivity import ParameterSensitivityAnalyzer
from nlsq.diagnostics.types import DiagnosticsConfig


@pytest.mark.parametrize(
    "analyzer", [IdentifiabilityAnalyzer, ParameterSensitivityAnalyzer]
)
@pytest.mark.parametrize(
    "fim",
    [
        np.diag([1e-20, -1e-20]),
        np.array([[1e-20, 1e-20], [0.0, 1e-20]]),
        np.full((2, 2), -(2**63), dtype=np.int64),
    ],
)
def test_small_invalid_information_matrix_is_rejected(analyzer, fim):
    report = analyzer(DiagnosticsConfig()).analyze_from_fim(fim)
    assert not report.available
