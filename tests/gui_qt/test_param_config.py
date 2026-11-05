"""Tests for ParamConfigWidget exact value round-trip (GUI7)."""

import math

import pytest

pytestmark = pytest.mark.serial


def test_tiny_and_large_values_round_trip_exactly(qtbot):
    from nlsq.gui_qt.widgets.param_config import ParamConfigWidget

    w = ParamConfigWidget()
    qtbot.addWidget(w)
    w.set_param_names(["a", "b"])
    w._auto_p0_check.setChecked(False)
    w.set_values([1.23456789e-9, 3.0], ([-5e-8, -1e10], [2e12, 1e10]))
    p0, bounds = w.get_values()
    assert p0 == [1.23456789e-9, 3.0]
    assert bounds == ([-5e-8, -1e10], [2e12, 1e10])  # +-1e10 are real bounds


def test_default_is_unbounded_and_typed_text_parses(qtbot):
    from nlsq.gui_qt.widgets.param_config import ParamConfigWidget

    w = ParamConfigWidget()
    qtbot.addWidget(w)
    w.set_param_names(["a"])
    w._auto_p0_check.setChecked(False)
    assert w.get_values()[1] == ([None], [None])
    assert w._table.cellWidget(0, 2).text() == "-inf"
    assert w._table.cellWidget(0, 3).text() == "inf"

    edit = w._table.cellWidget(0, 1)
    edit.setText("2.5e-9")
    edit.editingFinished.emit()
    assert w.get_values()[0] == [2.5e-9]
    edit.setText("1e")  # unparsable: keeps last good value
    assert math.isclose(w.get_values()[0][0], 2.5e-9)


def test_half_typed_text_is_normalised_on_focus_out(qtbot):
    """GUI-2: display and value must never diverge after leaving the field."""
    from nlsq.gui_qt.widgets.param_config import ParamConfigWidget

    w = ParamConfigWidget()
    qtbot.addWidget(w)
    w.show()
    w.set_param_names(["a", "b"])
    w._auto_p0_check.setChecked(False)
    edit = w._table.cellWidget(0, 1)
    other = w._table.cellWidget(1, 1)
    edit.setFocus()
    qtbot.waitUntil(edit.hasFocus)
    edit.setText("1e")
    other.setFocus()
    qtbot.waitUntil(lambda: not edit.hasFocus())
    assert edit.text() == repr(edit.value()) == "1.0"
