"""
pytest-qt fixtures for NLSQ Qt GUI testing.

This module provides common fixtures for testing Qt widgets using pytest-qt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from nlsq.gui_qt.app_state import AppState


@pytest.fixture
def app_state() -> AppState:
    """Create a fresh AppState instance for testing."""
    from nlsq.gui_qt.app_state import AppState

    return AppState()


@pytest.fixture
def theme_manager(qtbot: QtBot) -> None:
    """Create a ThemeManager instance for testing.

    Note: ThemeManager requires a QApplication instance, which pytest-qt provides.
    """
    from PySide6.QtWidgets import QApplication

    from nlsq.gui_qt.theme import ThemeManager

    app = QApplication.instance()
    if app is None:
        pytest.skip("QApplication not available")

    return ThemeManager(app)


@pytest.fixture
def main_window(qtbot: QtBot, app_state: AppState) -> None:
    """Create a MainWindow instance for testing."""
    from nlsq.gui_qt.main_window import MainWindow

    window = MainWindow(app_state)
    qtbot.addWidget(window)
    return window
