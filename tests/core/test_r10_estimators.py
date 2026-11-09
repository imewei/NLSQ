"""Round-10 regressions: scale-free p0 estimators, template phase sign, GUI max_nfev."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core import functions as F


def test_gaussian_auto_p0_on_tiny_x_range():
    x = np.linspace(0, 1e-5, 200)
    y = 3 * np.exp(-((x - 5e-6) ** 2) / (2 * (1e-6) ** 2))
    popt = np.asarray(curve_fit(F.gaussian, x, y, p0="auto")[0])
    np.testing.assert_allclose(popt, [3, 5e-6, 1e-6], rtol=1e-3)


def test_lorentzian_auto_p0_on_tiny_x_range():
    x = np.linspace(0, 1e-5, 400)
    y = 3 / (1 + ((x - 5e-6) / 5e-7) ** 2)
    p0 = F.estimate_p0_lorentzian(x, y)
    assert 1e-8 < p0[2] < 1e-5  # was clamped to 0.01 (1e4x too wide)


def test_exp_decay_rate_floor_scales_with_x_range():
    x = np.linspace(0, 1e-5, 100)
    p0 = F.estimate_p0_exponential_decay(x, 2 * np.exp(-x / 2e-6))
    assert p0[1] > 1e3  # 1/x units: true rate 5e5; floor was an absolute 0.01


def test_exp_growth_p0_respects_its_own_bounds():
    x = np.linspace(0, 10, 100)
    y = 2 * np.exp(0.3 * x) - 10
    p0 = F.estimate_p0_exponential_growth(x, y)
    lo, _ = F.exponential_growth.bounds()
    assert p0[0] >= lo[0]
    popt = np.asarray(curve_fit(F.exponential_growth, x, y, p0=p0)[0])
    np.testing.assert_allclose(popt, [2, 0.3, -10], rtol=1e-3)


def _template():
    path = Path(__file__).parents[2] / "nlsq/cli/templates/custom_model_template.py"
    spec = importlib.util.spec_from_file_location("_nlsq_tmpl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("phi", [1.0, -1.0, np.pi / 2])
def test_oscillator_template_phase_has_correct_sign(phi):
    mod = _template()
    x = np.linspace(0, 10, 400)
    y = np.asarray(mod.damped_oscillator(x, 2.0, 0.1, 1.5, phi))
    est = mod.estimate_p0(x, y)
    assert np.sign(est[3]) == np.sign(phi) or abs(est[3]) < 1e-6
