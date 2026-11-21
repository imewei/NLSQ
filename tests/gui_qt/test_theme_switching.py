"""Theme switching: one signal per switch, status labels follow the theme."""

from __future__ import annotations

import pytest

from nlsq.gui_qt.theme import DARK_THEME, LIGHT_THEME


@pytest.mark.serial
def test_set_theme_emits_theme_changed_once(main_window):
    main_window.set_theme("dark")  # start from a known state
    emitted = []
    main_window._theme_manager.theme_changed.connect(lambda t: emitted.append(t.name))
    state_emitted = []
    main_window._app_state.theme_changed.connect(state_emitted.append)

    main_window.set_theme("light")
    assert emitted == ["light"]
    assert state_emitted == ["light"]
    assert main_window._theme_toggle.isChecked() is False

    emitted.clear()
    state_emitted.clear()
    main_window._theme_toggle.setChecked(True)
    assert emitted == ["dark"]
    assert state_emitted == ["dark"]


@pytest.mark.serial
def test_status_labels_follow_theme(main_window):
    data_page = main_window._pages["data_loading"]
    export_page = main_window._pages["export"]
    for page, label in (
        (data_page, data_page._validation_label),
        (export_page, export_page._status_label),
    ):
        page._status.paint(label, "ok")
        main_window.set_theme("dark")
        assert DARK_THEME.success in label.styleSheet()
        main_window.set_theme("light")
        assert LIGHT_THEME.success in label.styleSheet()
        page._status.paint(label, "bad")
        assert LIGHT_THEME.error in label.styleSheet()
        main_window.set_theme("dark")
        assert DARK_THEME.error in label.styleSheet()


@pytest.mark.serial
def test_cleared_status_label_is_not_repainted(main_window):
    page = main_window._pages["data_loading"]
    page._status.paint(page._file_path_label, "ok")
    page._status.clear(page._file_path_label, "color: gray;")
    main_window.set_theme("light")
    assert page._file_path_label.styleSheet() == "color: gray;"


@pytest.mark.serial
@pytest.mark.parametrize("name, theme", [("light", LIGHT_THEME), ("dark", DARK_THEME)])
def test_every_widget_palette_follows_theme(qtbot, main_window, name, theme):
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QLabel

    other = "dark" if name == "light" else "light"
    main_window.set_theme(other)
    main_window.set_theme(name)
    qtbot.wait(50)  # palette reaches the window, then the stylesheet re-polishes
    role = QPalette.ColorRole
    # the window background and its text must switch together, or labels turn
    # unreadable (light-on-light) as in the half-switched Fusion palette
    for widget in (main_window, main_window.findChild(QLabel, "sidebarTitle")):
        pal = widget.palette()
        assert pal.color(role.Window).name() == theme.background
        assert pal.color(role.WindowText).name() == theme.text_primary
        assert pal.color(role.Base).name() == theme.surface
        assert pal.color(role.Text).name() == theme.text_primary


@pytest.mark.serial
def test_dark_mode_toggle_is_visible_and_text_is_clickable(qtbot, main_window):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    main_window.show()
    toggle = main_window._theme_toggle
    qtbot.wait(50)
    toggle.setChecked(True)
    checked = toggle.grab().toImage()
    toggle.setChecked(False)
    unchecked = toggle.grab().toImage()
    # a stylesheet rule on QCheckBox::indicator used to erase the box entirely
    assert checked != unchecked
    assert "QCheckBox::indicator" not in main_window.styleSheet()

    # the "Dark Mode" text is part of the checkbox: clicking it flips the theme
    before = toggle.isChecked()
    QTest.mouseClick(
        toggle,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(40, toggle.height() // 2),
    )
    assert toggle.isChecked() is (not before)
    assert main_window._theme_manager.current_theme.is_dark is (not before)
