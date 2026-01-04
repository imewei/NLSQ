"""
NLSQ Qt GUI Main Window

This module provides the main application window with sidebar navigation
and stacked page layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from nlsq.gui_qt.autosave import AutosaveManager
from nlsq.gui_qt.pages import PageState
from nlsq.gui_qt.theme import ThemeManager

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState

__all__ = ["MainWindow"]

# Page configuration: (name, display_label, icon_placeholder)
PAGE_CONFIG = [
    ("data_loading", "Data Loading", "📁"),
    ("model_selection", "Model Selection", "📐"),
    ("fitting_options", "Fitting Options", "⚙️"),
    ("results", "Results", "📊"),
    ("export", "Export", "💾"),
]


class MainWindow(QMainWindow):
    """Main application window with sidebar navigation.

    This window provides:
    - Sidebar navigation with page list
    - Theme toggle switch
    - Stacked widget for page content
    - Status bar for application state
    - Navigation guards based on workflow state
    """

    # Minimum window size (EC-004)
    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    def __init__(self, app_state: AppState) -> None:
        """Initialize the main window.

        Args:
            app_state: Application state manager
        """
        super().__init__()
        self._app_state = app_state
        self._pages: dict[str, QWidget] = {}

        # Initialize theme manager
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        self._theme_manager = ThemeManager(app)

        self._settings = QSettings("NLSQ", "QtGUI")

        # Initialize autosave manager
        self._autosave = AutosaveManager(app_state, self)

        self._setup_window()
        self._setup_ui()
        self._setup_connections()
        self._setup_shortcuts()
        self._update_navigation_guards()
        self._restore_state()
        self._check_recovery()
        self._autosave.start()

    def _setup_window(self) -> None:
        """Configure window properties."""
        self.setWindowTitle("NLSQ Curve Fitting")
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.resize(1200, 800)

    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout: sidebar | content
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create sidebar
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        # Create page stack
        self._page_stack = QStackedWidget()
        self._create_pages()
        main_layout.addWidget(self._page_stack, stretch=1)

        # Create status bar
        self._setup_status_bar()

    def _create_sidebar(self) -> QWidget:
        """Create the sidebar widget with navigation and theme toggle."""
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setObjectName("sidebar")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # App title
        title = QLabel("NLSQ")
        title.setObjectName("sidebarTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 16px 0;")
        layout.addWidget(title)

        # Navigation list
        self._nav_list = QListWidget()
        self._nav_list.setObjectName("navList")
        self._nav_list.setSpacing(4)

        for page_name, display_label, icon in PAGE_CONFIG:
            item = QListWidgetItem(f"{icon}  {display_label}")
            item.setData(Qt.ItemDataRole.UserRole, page_name)
            self._nav_list.addItem(item)

        self._nav_list.setCurrentRow(0)
        layout.addWidget(self._nav_list, stretch=1)

        # Theme toggle
        theme_container = QWidget()
        theme_layout = QHBoxLayout(theme_container)
        theme_layout.setContentsMargins(0, 8, 0, 0)

        theme_label = QLabel("Dark Mode")
        self._theme_toggle = QCheckBox()
        self._theme_toggle.setChecked(True)  # Start in dark mode

        theme_layout.addWidget(theme_label)
        theme_layout.addStretch()
        theme_layout.addWidget(self._theme_toggle)

        layout.addWidget(theme_container)

        return sidebar

    def _create_pages(self) -> None:
        """Create and add all pages to the stack."""
        from nlsq.gui_qt.pages.data_loading import DataLoadingPage
        from nlsq.gui_qt.pages.export import ExportPage
        from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage
        from nlsq.gui_qt.pages.model_selection import ModelSelectionPage
        from nlsq.gui_qt.pages.results import ResultsPage

        # Create pages
        page_classes = {
            "data_loading": DataLoadingPage,
            "model_selection": ModelSelectionPage,
            "fitting_options": FittingOptionsPage,
            "results": ResultsPage,
            "export": ExportPage,
        }

        for page_name, page_class in page_classes.items():
            page = page_class(self._app_state)
            self._pages[page_name] = page
            self._page_stack.addWidget(page)

    def _setup_status_bar(self) -> None:
        """Set up the status bar."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        # Add permanent widgets
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)

        self._data_status = QLabel("")
        self._status_bar.addPermanentWidget(self._data_status)

    def _setup_connections(self) -> None:
        """Connect signals and slots."""
        # Navigation
        self._nav_list.currentRowChanged.connect(self._on_nav_changed)

        # Theme toggle
        self._theme_toggle.toggled.connect(self._on_theme_toggled)
        self._theme_manager.theme_changed.connect(self._on_theme_changed)

        # App state changes
        self._app_state.page_access_changed.connect(self._update_navigation_guards)
        self._app_state.data_changed.connect(self._update_status)
        self._app_state.fit_completed.connect(self._on_fit_completed)

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        # Page navigation shortcuts (Ctrl+1 through Ctrl+5)
        for i, (page_name, _, _) in enumerate(PAGE_CONFIG):
            action = QAction(self)
            action.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            action.triggered.connect(
                lambda checked, idx=i: self._nav_list.setCurrentRow(idx)
            )
            self.addAction(action)

        # Theme toggle (Ctrl+T)
        toggle_theme = QAction(self)
        toggle_theme.setShortcut(QKeySequence("Ctrl+T"))
        toggle_theme.triggered.connect(self._theme_manager.toggle)
        self.addAction(toggle_theme)

        # Run fit (Ctrl+R)
        run_fit = QAction(self)
        run_fit.setShortcut(QKeySequence("Ctrl+R"))
        run_fit.triggered.connect(self._on_run_fit_shortcut)
        self.addAction(run_fit)

        # Open file (Ctrl+O)
        open_file = QAction(self)
        open_file.setShortcut(QKeySequence.StandardKey.Open)
        open_file.triggered.connect(self._on_open_file_shortcut)
        self.addAction(open_file)

        # Next page (Ctrl+PgDown)
        next_page = QAction(self)
        next_page.setShortcut(QKeySequence("Ctrl+PgDown"))
        next_page.triggered.connect(self._navigate_next)
        self.addAction(next_page)

        # Previous page (Ctrl+PgUp)
        prev_page = QAction(self)
        prev_page.setShortcut(QKeySequence("Ctrl+PgUp"))
        prev_page.triggered.connect(self._navigate_prev)
        self.addAction(prev_page)

    def _on_run_fit_shortcut(self) -> None:
        """Handle run fit shortcut."""
        fitting_page = self._pages.get("fitting_options")
        if fitting_page is not None and hasattr(fitting_page, "run_fit"):
            # Navigate to fitting page first
            self._nav_list.setCurrentRow(2)
            fitting_page.run_fit()

    def _on_open_file_shortcut(self) -> None:
        """Handle open file shortcut."""
        # Navigate to data loading page
        self._nav_list.setCurrentRow(0)
        # Trigger file dialog in data loading page
        data_page = self._pages.get("data_loading")
        if data_page is not None and hasattr(data_page, "_on_browse"):
            data_page._on_browse()

    def _navigate_next(self) -> None:
        """Navigate to next page."""
        current = self._nav_list.currentRow()
        if current < len(PAGE_CONFIG) - 1:
            self._nav_list.setCurrentRow(current + 1)

    def _navigate_prev(self) -> None:
        """Navigate to previous page."""
        current = self._nav_list.currentRow()
        if current > 0:
            self._nav_list.setCurrentRow(current - 1)

    def _on_nav_changed(self, row: int) -> None:
        """Handle navigation list selection change."""
        if row >= 0:
            self._page_stack.setCurrentIndex(row)
            page_name = PAGE_CONFIG[row][1]
            self._status_label.setText(f"Page: {page_name}")

    def _on_theme_toggled(self, checked: bool) -> None:
        """Handle theme toggle."""
        theme_name = "dark" if checked else "light"
        self._theme_manager.set_theme(theme_name)
        self._app_state.set_theme_name(theme_name)

    def _on_theme_changed(self, theme) -> None:
        """Handle theme change signal."""
        # Propagate theme to all pages
        for page in self._pages.values():
            if hasattr(page, "set_theme"):
                page.set_theme(theme)

    def _update_navigation_guards(self) -> None:
        """Update navigation item enabled state based on workflow."""
        page_state = PageState.from_session_state(self._app_state.state)

        for i, (page_name, _, _) in enumerate(PAGE_CONFIG):
            item = self._nav_list.item(i)
            can_access = page_state.can_access(page_name)

            # Update item flags
            if can_access:
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
            else:
                item.setFlags(
                    item.flags()
                    & ~Qt.ItemFlag.ItemIsEnabled
                    & ~Qt.ItemFlag.ItemIsSelectable
                )

    def _update_status(self) -> None:
        """Update status bar with current data info."""
        state = self._app_state.state
        if state.xdata is not None and state.ydata is not None:
            n_points = len(state.xdata)
            self._data_status.setText(f"Data: {n_points:,} points")
        else:
            self._data_status.setText("")

    def _on_fit_completed(self, result: object) -> None:
        """Handle fit completion."""
        self._status_label.setText("Fit completed successfully")
        # Navigate to results page
        self._nav_list.setCurrentRow(3)  # Results page

    def navigate_to(self, page_name: str) -> None:
        """Navigate to a specific page.

        Args:
            page_name: Name of the page to navigate to
        """
        for i, (name, _, _) in enumerate(PAGE_CONFIG):
            if name == page_name:
                self._nav_list.setCurrentRow(i)
                break

    def set_theme(self, theme: str) -> None:
        """Set the application theme.

        Args:
            theme: "light" or "dark"
        """
        is_dark = theme == "dark"
        self._theme_toggle.setChecked(is_dark)
        self._theme_manager.set_theme(theme)

    def update_navigation_guards(self) -> None:
        """Public method to update navigation guards."""
        self._update_navigation_guards()

    def _save_state(self) -> None:
        """Save window state to settings."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        self._settings.setValue("currentPage", self._nav_list.currentRow())
        self._settings.setValue("theme", self._theme_manager.current_theme.name)

    def _restore_state(self) -> None:
        """Restore window state from settings."""
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        window_state = self._settings.value("windowState")
        if window_state is not None:
            self.restoreState(window_state)

        # Restore current page
        page_idx = self._settings.value("currentPage", 0, type=int)
        if 0 <= page_idx < len(PAGE_CONFIG):
            self._nav_list.setCurrentRow(page_idx)

        # Restore theme
        theme = self._settings.value("theme", "dark")
        if theme == "light":
            self._theme_toggle.setChecked(False)
            self._theme_manager.set_theme("light")

    def _check_recovery(self) -> None:
        """Check for and offer session recovery."""
        from PySide6.QtWidgets import QMessageBox

        if self._autosave.check_recovery():
            reply = QMessageBox.question(
                self,
                "Session Recovery",
                "A previous session was not saved properly.\n\n"
                "Would you like to recover it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                if self._autosave.recover():
                    self._status_label.setText("Session recovered")
                    self._update_navigation_guards()
                else:
                    QMessageBox.warning(
                        self,
                        "Recovery Failed",
                        "Failed to recover the previous session.",
                    )

            # Clear recovery file regardless of choice
            self._autosave.clear_recovery()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event.

        Args:
            event: Close event
        """
        # Stop autosave and clear recovery file (clean shutdown)
        self._autosave.stop()
        self._autosave.clear_recovery()

        # Save window state before closing
        self._save_state()
        event.accept()
