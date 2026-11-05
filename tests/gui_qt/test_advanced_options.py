"""Tolerance inputs of AdvancedOptionsWidget keep full precision (SWM4)."""

import pytest

pytestmark = pytest.mark.serial


def test_tolerances_round_trip_and_clamp(qtbot):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    qtbot.addWidget(w)
    w.set_options({"gtol": 1e-13, "ftol": 3e-15, "xtol": 5.0})
    opts = w.get_options()
    assert opts["gtol"] == 1e-13
    assert opts["ftol"] == 3e-15
    assert opts["xtol"] == 1.0  # clamped to the documented maximum


def test_tolerance_has_no_step_buttons(qtbot):
    """A wheel tick/arrow press must not turn gtol 1e-8 into 1.0."""
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    qtbot.addWidget(w)
    assert not hasattr(w._gtol_spin, "stepBy")
    assert w.get_options()["gtol"] == 1e-8
