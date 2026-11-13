"""Custom-code editor regressions: theme change, function choice, recovery."""

import pytest

from nlsq.gui_qt.pages.model_selection import ModelSelectionPage
from nlsq.gui_qt.theme import LIGHT_THEME

CODE = (
    "import jax.numpy as jnp\n"
    "def helper(x, a):\n    return a * x\n"
    "def model(x, a, b):\n    return a * x + b\n"
)

pytestmark = pytest.mark.serial


def _page(app_state, qtbot):
    page = ModelSelectionPage(app_state)
    qtbot.addWidget(page)
    return page


def test_theme_change_does_not_switch_model_type(app_state, qtbot):
    page = _page(app_state, qtbot)
    page._custom_radio.click()
    page._code_editor.set_code(CODE)
    page._builtin_radio.click()
    page._builtin_combo.setCurrentIndex(3)
    before = (page._current_model_type, dict(page._current_config))
    page.set_theme(LIGHT_THEME)
    assert (page._current_model_type, page._current_config) == before
    assert page._builtin_radio.isChecked()


def test_editing_code_keeps_selected_function(app_state, qtbot):
    page = _page(app_state, qtbot)
    page._custom_radio.click()
    page._code_editor.set_code(CODE)
    page._function_combo.setCurrentText("model")
    page._code_editor.set_code(CODE + "\n")
    assert page._function_combo.currentText() == "model"
    assert page._current_config["function"] == "model"


def test_function_combo_enabled_after_recovery(app_state, qtbot):
    app_state.set_model("custom", {"code": CODE, "function": "model"})
    page = _page(app_state, qtbot)
    page._restore_session_controls()
    assert page._function_combo.isEnabled()
    assert page._function_combo.currentText() == "model"
