"""FitStatisticsWidget keeps semantic colors across theme switches."""

import pytest

pytestmark = pytest.mark.serial


def test_semantic_colors_survive_theme_switch(qtbot):
    from nlsq.gui_qt.theme import DARK_THEME, LIGHT_THEME
    from nlsq.gui_qt.widgets.fit_statistics import FitStatisticsWidget

    w = FitStatisticsWidget()
    qtbot.addWidget(w)
    w.set_statistics(0.5, 1, 1, 1, 1, 3, False)
    w.set_theme(LIGHT_THEME)
    assert LIGHT_THEME.stat_bad in w._r2_card._value_label.styleSheet()
    assert LIGHT_THEME.stat_bad in w._status_card._value_label.styleSheet()
    w.set_statistics(0.999, 1, 1, 1, 1, 3, True)
    w.set_theme(DARK_THEME)
    assert DARK_THEME.stat_good in w._r2_card._value_label.styleSheet()
    assert DARK_THEME.stat_good in w._status_card._value_label.styleSheet()
