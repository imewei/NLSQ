"""Problem analysis agrees with the actual model parameter count."""

import numpy as np

from nlsq.precision.algorithm_selector import AlgorithmSelector


def test_algorithm_analysis_ignores_model_variadics():
    def slope(x, a, **options):
        return a * x

    x = np.linspace(0, 1, 12)
    analysis = AlgorithmSelector().analyze_problem(slope, x, 2 * x)
    assert analysis["n_params"] == 1
    assert analysis["overdetermination_ratio"] == 12
