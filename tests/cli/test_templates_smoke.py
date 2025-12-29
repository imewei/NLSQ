"""Smoke tests for CLI templates to cover fast branches."""

from __future__ import annotations

import numpy as np

from nlsq.cli.templates import custom_model_template as template


def test_template_estimate_p0_and_bounds() -> None:
    """Template helpers should return sensible shapes with tiny data."""
    x = np.linspace(0.0, 1.0, 10)
    y = np.cos(2 * np.pi * x)

    p0 = template.estimate_p0(x, y)
    lower, upper = template.bounds()

    assert len(p0) == 4
    assert len(lower) == 4
    assert len(upper) == 4
    assert lower[0] == 0.0
    assert upper[0] == np.inf
