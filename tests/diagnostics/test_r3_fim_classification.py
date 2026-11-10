"""R3-1: FIM and Jacobian paths must classify well-resolved directions alike."""

import numpy as np
import pytest

from nlsq.diagnostics.parameter_sensitivity import ParameterSensitivityAnalyzer


# Exact-zero singular value: before the fix the null direction inflated the
# range and mis-classified index 1 as sloppy too ([1, 2]).
# Not parametrized with sv=[1, 1e-4, 1e-12]: the FIM path squares the spectrum
# and cannot resolve eigenvalues below n*eps*max, so FIM and Jacobian paths
# legitimately differ there (inherent resolution difference).
@pytest.mark.parametrize("sv", [[1.0, 1e-3, 0.0], [1e3, 1.0, 0.0]])
def test_fim_and_jacobian_paths_agree_with_null_direction(sv):
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(200, 3)))
    jac = q @ np.diag(sv)
    a = ParameterSensitivityAnalyzer()
    rj = a.analyze(jac)
    rf = a.analyze_from_fim(jac.T @ jac)
    assert rj.sloppy_indices == [2]
    assert rf.sloppy_indices == rj.sloppy_indices
    assert rf.stiff_indices == rj.stiff_indices
