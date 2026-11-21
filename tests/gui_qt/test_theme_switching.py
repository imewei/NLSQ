"""Theme switching: one signal per switch, status labels follow the theme."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from nlsq.gui_qt.theme import DARK_THEME, LIGHT_THEME, StatusPainter, ThemeManager


@pytest.fixture(autouse=True)
def _restore_app_palette(qtbot):
    """ThemeManager sets the process-wide palette; do not leak it to other tests."""
    app = QApplication.instance()
    saved = app.palette()
    yield
    app.setPalette(saved)


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


def _self_coloured(widget, root):
    while widget is not None and widget is not root:
        if "color" in widget.styleSheet():
            return True
        widget = widget.parentWidget()
    return False


@pytest.mark.serial
@pytest.mark.parametrize("name, theme", [("light", LIGHT_THEME), ("dark", DARK_THEME)])
def test_every_widget_palette_follows_theme(qtbot, main_window, name, theme):
    from PySide6.QtWidgets import QLabel, QWidget

    other = "dark" if name == "light" else "light"
    main_window.set_theme(other)
    main_window.set_theme(name)
    role = QPalette.ColorRole
    page = main_window._pages["data_loading"]
    page._status.paint(page._validation_label, "ok")  # label with its own stylesheet
    probe = main_window.findChild(QLabel, "sidebarTitle")

    # the stylesheet re-polish is deferred: wait for it instead of sleeping
    qtbot.waitUntil(
        lambda: probe.palette().color(role.WindowText).name() == theme.text_primary,
        timeout=2000,
    )
    # the window background and its text must switch together, or labels turn
    # unreadable (light-on-light) as in the half-switched Fusion palette
    checked = 0
    for widget in [main_window, *main_window.findChildren(QWidget)]:
        # widgets that colour themselves through their own stylesheet (or sit
        # inside one, like a text editor's viewport) keep that colour on purpose;
        # everything else must follow the application palette
        if _self_coloured(widget, main_window):
            continue
        pal = widget.palette()
        assert pal.color(role.Window).name() == theme.background, type(widget)
        assert pal.color(role.Base).name() == theme.surface, type(widget)
        checked += 1
    assert checked > 50


@pytest.mark.serial
def test_dark_mode_toggle_is_visible_and_text_is_clickable(qtbot, main_window):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    main_window.show()
    toggle = main_window._theme_toggle
    qtbot.wait(50)

    def indicator(checked):
        # same theme for both grabs: only the indicator may differ
        toggle.blockSignals(True)
        toggle.setChecked(checked)
        toggle.blockSignals(False)
        qtbot.wait(20)
        return toggle.grab().toImage().copy(0, 0, 14, toggle.height())

    checked, unchecked = indicator(True), indicator(False)
    # a stylesheet rule on QCheckBox::indicator used to erase the box entirely:
    # the unchecked box must draw something, and the checkmark must differ
    colours = {
        unchecked.pixelColor(x, y).name()
        for x in range(unchecked.width())
        for y in range(unchecked.height())
    }
    assert len(colours) >= 2
    assert checked != unchecked

    # the "Dark Mode" text is part of the checkbox: clicking it flips the theme
    before = toggle.isChecked()
    QTest.mouseClick(
        toggle,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(toggle.sizeHint().width() - 8, toggle.height() // 2),
    )
    assert toggle.isChecked() is (not before)
    assert main_window._theme_manager.current_theme.is_dark is (not before)


def test_status_painter_rejects_unknown_role_and_drops_cleared_labels(qtbot):
    from PySide6.QtWidgets import QLabel

    painter, label = StatusPainter(), QLabel()
    qtbot.addWidget(label)
    with pytest.raises(ValueError, match="unknown status role"):
        painter.paint(label, "OK")
    for role, attr in (
        ("ok", "success"),
        ("bad", "error"),
        ("warn", "warning"),
        ("info", "info"),
    ):
        painter.paint(label, role)
        assert getattr(DARK_THEME, attr) in label.styleSheet()
    painter.set_theme(LIGHT_THEME)
    assert LIGHT_THEME.info in label.styleSheet()
    painter.clear(label, "color: gray;")
    assert label not in painter._roles


def test_theme_manager_rejects_unknown_theme_name():
    from PySide6.QtCore import Qt

    from tests.gui_qt.test_review_2026_11_16_gui_r1 import _FakeApp

    tm = ThemeManager(_FakeApp(Qt.ColorScheme.Light))
    with pytest.raises(ValueError, match="unknown theme"):
        tm.set_theme("Dark")
    assert not tm.explicit


def _window_color(palette):
    return palette.color(QPalette.ColorRole.Window).name()


def test_theme_manager_reasserts_palette_after_scheme_signal():
    from PySide6.QtCore import Qt

    from tests.gui_qt.test_review_2026_11_16_gui_r1 import _FakeApp

    app = _FakeApp(Qt.ColorScheme.Light)
    tm = ThemeManager(app)
    assert _window_color(app.palettes[-1]) == LIGHT_THEME.background  # at construction

    seen = []
    tm.theme_changed.connect(lambda t: seen.append(t.name))
    tm.set_theme("light")
    n = len(app.palettes)
    # the platform rebuilds its palette when the scheme signal arrives; an
    # explicit choice must put its own palette back and must not re-emit
    for slot in app.hints.colorSchemeChanged.slots:
        slot(Qt.ColorScheme.Dark)
    assert len(app.palettes) == n + 1
    assert _window_color(app.palettes[-1]) == LIGHT_THEME.background
    assert seen == ["light"]  # only the set_theme emission


def test_theme_manager_follows_os_scheme_with_matching_palette():
    from PySide6.QtCore import Qt

    from tests.gui_qt.test_review_2026_11_16_gui_r1 import _FakeApp

    app = _FakeApp(Qt.ColorScheme.Light)
    tm = ThemeManager(app)
    for slot in app.hints.colorSchemeChanged.slots:
        slot(Qt.ColorScheme.Dark)
    assert tm.current_theme.name == "dark"
    assert _window_color(app.palettes[-1]) == DARK_THEME.background
