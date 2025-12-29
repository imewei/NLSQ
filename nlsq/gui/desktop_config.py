"""Desktop application configuration for NLSQ GUI.

This module provides configuration settings for running the NLSQ Streamlit
application as a native desktop window using streamlit-desktop-app.

Configuration includes:
- Window title and dimensions
- Application icon path
- Server settings for local-only operation
- Security settings to disable web server exposure

Usage:
    from nlsq.gui.desktop_config import get_desktop_config, DesktopConfig
    config = get_desktop_config()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DesktopConfig:
    """Configuration for the NLSQ desktop application.

    Attributes
    ----------
    window_title : str
        Title displayed in the window title bar.
    window_width : int
        Initial window width in pixels.
    window_height : int
        Initial window height in pixels.
    icon_path : Path | None
        Path to the application icon file (PNG or ICO).
    splash_image_path : Path | None
        Path to the splash screen image (optional).
    server_port : int
        Local port for the Streamlit server.
    server_headless : bool
        Run server in headless mode (no browser auto-open).
    server_enable_cors : bool
        Enable CORS (disabled for security in desktop mode).
    server_enable_xsrf_protection : bool
        Enable XSRF protection.
    server_address : str
        Server bind address (localhost for local-only).
    browser_gather_usage_stats : bool
        Disable usage statistics collection.
    """

    window_title: str = "NLSQ Curve Fitting"
    window_width: int = 1400
    window_height: int = 900
    icon_path: Path | None = None
    splash_image_path: Path | None = None
    server_port: int = 8501
    server_headless: bool = True
    server_enable_cors: bool = False
    server_enable_xsrf_protection: bool = True
    server_address: str = "localhost"
    browser_gather_usage_stats: bool = False

    def __post_init__(self) -> None:
        """Initialize paths to assets directory."""
        assets_dir = Path(__file__).parent / "assets"

        # Set default icon path if not provided
        if self.icon_path is None:
            icon_png = assets_dir / "icon.png"
            icon_ico = assets_dir / "icon.ico"
            if icon_png.exists():
                self.icon_path = icon_png
            elif icon_ico.exists():
                self.icon_path = icon_ico

        # Set default splash image path if not provided
        if self.splash_image_path is None:
            splash_path = assets_dir / "splash.png"
            if splash_path.exists():
                self.splash_image_path = splash_path


def get_desktop_config(**overrides: Any) -> DesktopConfig:
    """Get the desktop application configuration.

    Parameters
    ----------
    **overrides : Any
        Configuration overrides to apply.

    Returns
    -------
    DesktopConfig
        The desktop configuration instance with any overrides applied.

    Examples
    --------
    >>> config = get_desktop_config()
    >>> config.window_title
    'NLSQ Curve Fitting'

    >>> config = get_desktop_config(window_width=1600, window_height=1000)
    >>> config.window_width
    1600
    """
    return DesktopConfig(**overrides)


def get_streamlit_config_dict(config: DesktopConfig | None = None) -> dict[str, Any]:
    """Convert desktop config to Streamlit configuration dictionary.

    This generates configuration suitable for passing to Streamlit's
    configuration system or writing to a config.toml file.

    Parameters
    ----------
    config : DesktopConfig | None
        The desktop configuration to convert. If None, uses default config.

    Returns
    -------
    dict[str, Any]
        Dictionary with Streamlit configuration sections.

    Examples
    --------
    >>> config = get_desktop_config()
    >>> st_config = get_streamlit_config_dict(config)
    >>> st_config["server"]["headless"]
    True
    """
    if config is None:
        config = get_desktop_config()

    return {
        "server": {
            "port": config.server_port,
            "headless": config.server_headless,
            "enableCORS": config.server_enable_cors,
            "enableXsrfProtection": config.server_enable_xsrf_protection,
            "address": config.server_address,
        },
        "browser": {
            "gatherUsageStats": config.browser_gather_usage_stats,
            "serverAddress": config.server_address,
            "serverPort": config.server_port,
        },
        "theme": {
            "base": "light",
            "primaryColor": "#1f77b4",
        },
    }


def get_webview_options(config: DesktopConfig | None = None) -> dict[str, Any]:
    """Get options for pywebview window creation.

    Parameters
    ----------
    config : DesktopConfig | None
        The desktop configuration to use. If None, uses default config.

    Returns
    -------
    dict[str, Any]
        Dictionary of options for pywebview.create_window().

    Examples
    --------
    >>> config = get_desktop_config()
    >>> options = get_webview_options(config)
    >>> options["title"]
    'NLSQ Curve Fitting'
    """
    if config is None:
        config = get_desktop_config()

    return {
        "title": config.window_title,
        "width": config.window_width,
        "height": config.window_height,
        "resizable": True,
        "min_size": (800, 600),
        "text_select": True,
    }


def get_app_path() -> Path:
    """Get the path to the main Streamlit app file.

    Returns
    -------
    Path
        Path to app.py in the GUI package.
    """
    return Path(__file__).parent / "app.py"


def get_assets_path() -> Path:
    """Get the path to the assets directory.

    Returns
    -------
    Path
        Path to the assets directory.
    """
    return Path(__file__).parent / "assets"


# Default configuration instance
DEFAULT_CONFIG = get_desktop_config()
