"""Output formatting must preserve unsuccessful optimization status."""

from unittest.mock import patch

import numpy as np
import pytest

from nlsq.core.minpack import _apply_output_flags, curve_fit, fit
from nlsq.result import CurveFitResult


def test_cmaes_full_output_preserves_failure():
    result = CurveFitResult(
        x=np.array([1.0]),
        pcov=np.eye(1),
        success=False,
        status=0,
        message="budget exhausted",
    )
    output = _apply_output_flags(
        result, lambda x, a: a * x, np.arange(3.0), {"full_output": True}
    )
    assert output[4] == 5


@pytest.mark.parametrize("workflow", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_public_cmaes_wrappers_preserve_optimizer_failure(workflow, nested):
    failure = {"success": False, "status": 0, "message": "budget exhausted"}
    payload = {"popt": np.array([1.0]), "pcov": np.eye(1)}
    payload.update({"nlsq_result": failure} if nested else failure)
    x = np.arange(3.0)
    with (
        patch("nlsq.global_optimization.cmaes_optimizer.CMAESOptimizer") as optimizer,
        patch(
            "nlsq.global_optimization.method_selector.MethodSelector.select",
            return_value="cmaes",
        ),
    ):
        optimizer.return_value.fit.return_value = payload
        fn = fit if workflow else curve_fit
        extra = {"workflow": "auto_global"} if workflow else {}
        result = fn(
            lambda x, a: a * x,
            x,
            x,
            p0=[1.0],
            bounds=([0.0], [2.0]),
            method="cmaes",
            **extra,
        )
    assert result.success is False
    assert result.status == 0
    assert result.message == "budget exhausted"
