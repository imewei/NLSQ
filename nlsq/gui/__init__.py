"""NLSQ GUI - Streamlit-based desktop GUI for curve fitting.

This package provides a graphical user interface for the NLSQ library,
offering full feature parity with the CLI through an interactive visual interface.

Usage:
    # Run as Streamlit web app
    streamlit run nlsq/gui/app.py

    # Run as desktop application
    python -m nlsq.gui.run_desktop

    # Or from Python
    from nlsq.gui import run_desktop
    run_desktop()
"""

from nlsq.gui.app import main
from nlsq.gui.desktop_config import (
    DEFAULT_CONFIG,
    DesktopConfig,
    get_app_path,
    get_assets_path,
    get_desktop_config,
    get_streamlit_config_dict,
    get_webview_options,
)
from nlsq.gui.presets import (
    PRESETS,
    get_preset,
    get_preset_description,
    get_preset_names,
    get_preset_tolerances,
)
from nlsq.gui.state import (
    SessionState,
    apply_preset_to_state,
    get_current_config,
    initialize_state,
    reset_state,
)

__all__ = [
    # Desktop configuration
    "DEFAULT_CONFIG",
    # Presets
    "PRESETS",
    "DesktopConfig",
    # Session state
    "SessionState",
    "apply_preset_to_state",
    "get_app_path",
    "get_assets_path",
    "get_current_config",
    "get_desktop_config",
    "get_preset",
    "get_preset_description",
    "get_preset_names",
    "get_preset_tolerances",
    "get_streamlit_config_dict",
    "get_webview_options",
    "initialize_state",
    # Main entry points
    "main",
    "reset_state",
    "run_desktop",
]


def run_desktop(
    width: int = 1400,
    height: int = 900,
    browser_fallback: bool = True,
    debug: bool = False,
) -> None:
    """Run the NLSQ GUI as a desktop application.

    This function launches the Streamlit application in a native desktop
    window using pywebview. If pywebview is not available, it falls back
    to opening in the default web browser.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    browser_fallback : bool
        If True, fall back to browser mode when pywebview is unavailable.
    debug : bool
        Enable debug mode with developer tools.

    Examples
    --------
    >>> from nlsq.gui import run_desktop
    >>> run_desktop()  # Opens desktop window

    >>> run_desktop(width=1600, height=1000)  # Custom size
    """
    from nlsq.gui.run_desktop import run_in_browser, run_with_webview

    try:
        run_with_webview(width=width, height=height, debug=debug)
    except Exception as e:
        if browser_fallback:
            import logging

            logging.getLogger(__name__).warning(
                f"Desktop mode unavailable ({e}), using browser mode"
            )
            run_in_browser()
        else:
            raise
