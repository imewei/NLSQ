#!/usr/bin/env python3
"""Desktop launcher for NLSQ Curve Fitting GUI.

This module provides the entry point for running the NLSQ Streamlit application
as a native desktop window. It uses pywebview (via streamlit-desktop-app) to
wrap the Streamlit server in a native window without exposing a web server.

Usage:
    # Direct execution
    python -m nlsq.gui.run_desktop

    # Or via the installed script
    nlsq-gui

    # Command line options
    python -m nlsq.gui.run_desktop --width 1600 --height 1000
"""

import argparse
import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_free_port(start_port: int = 8501, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.

    Parameters
    ----------
    start_port : int
        The port number to start searching from.
    max_attempts : int
        Maximum number of ports to try.

    Returns
    -------
    int
        An available port number.

    Raises
    ------
    RuntimeError
        If no available port is found within max_attempts.
    """
    for port in range(start_port, start_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_attempts}"
    )


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Wait for the Streamlit server to become available.

    Parameters
    ----------
    port : int
        The port to check.
    timeout : float
        Maximum time to wait in seconds.
    interval : float
        Time between checks in seconds.

    Returns
    -------
    bool
        True if server is available, False if timeout reached.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            result = sock.connect_ex(("localhost", port))
            if result == 0:
                return True
        time.sleep(interval)
    return False


def get_streamlit_args(port: int) -> list[str]:
    """Get command line arguments for launching Streamlit.

    Parameters
    ----------
    port : int
        The port to run the server on.

    Returns
    -------
    list[str]
        Command line arguments for Streamlit.
    """
    app_path = Path(__file__).parent / "app.py"

    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        "localhost",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "true",
        "--global.developmentMode",
        "false",
    ]


class StreamlitServer:
    """Manager for the Streamlit server process.

    This class handles starting, stopping, and monitoring the Streamlit
    server subprocess.

    Attributes
    ----------
    port : int
        The port the server is running on.
    process : subprocess.Popen | None
        The server process, or None if not running.
    """

    def __init__(self, port: int | None = None) -> None:
        """Initialize the server manager.

        Parameters
        ----------
        port : int | None
            The port to use. If None, finds an available port.
        """
        self.port = port if port is not None else find_free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self._output_thread: threading.Thread | None = None

    def start(self, capture_output: bool = True) -> bool:
        """Start the Streamlit server.

        Parameters
        ----------
        capture_output : bool
            Whether to capture and log server output.

        Returns
        -------
        bool
            True if server started successfully.
        """
        if self.process is not None:
            logger.warning("Server already running")
            return True

        args = get_streamlit_args(self.port)
        logger.info(f"Starting Streamlit server on port {self.port}")

        try:
            # Set up environment for bundled execution
            env = os.environ.copy()
            env["STREAMLIT_SERVER_HEADLESS"] = "true"
            env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.STDOUT if capture_output else None,
                env=env,
            )

            if capture_output:
                self._output_thread = threading.Thread(
                    target=self._log_output,
                    daemon=True,
                )
                self._output_thread.start()

            # Wait for server to be ready
            if wait_for_server(self.port):
                logger.info(f"Streamlit server ready on http://localhost:{self.port}")
                return True
            else:
                logger.error("Streamlit server failed to start within timeout")
                self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start Streamlit server: {e}")
            return False

    def _log_output(self) -> None:
        """Log output from the server process."""
        if self.process is None or self.process.stdout is None:
            return

        for line in iter(self.process.stdout.readline, b""):
            try:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    logger.debug(f"Streamlit: {decoded}")
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the Streamlit server."""
        if self.process is None:
            return

        logger.info("Stopping Streamlit server")
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, killing")
                self.process.kill()
                self.process.wait()
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
        finally:
            self.process = None

    @property
    def url(self) -> str:
        """Get the URL for the running server."""
        return f"http://localhost:{self.port}"

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.process is not None and self.process.poll() is None


def run_with_webview(
    width: int = 1400,
    height: int = 900,
    debug: bool = False,
) -> None:
    """Run the NLSQ GUI as a desktop application using pywebview.

    This creates a native window containing the Streamlit application,
    without exposing a web server to the network.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    debug : bool
        Enable debug mode with developer tools.
    """
    try:
        import webview
    except ImportError:
        logger.error(
            "pywebview is not installed. Install it with: pip install pywebview\n"
            "Or install all GUI dependencies with: pip install nlsq[gui]"
        )
        sys.exit(1)

    # Start the Streamlit server
    server = StreamlitServer()

    # Register cleanup handler
    def cleanup() -> None:
        server.stop()

    atexit.register(cleanup)

    # Handle signals for clean shutdown
    def signal_handler(signum: int, frame: Any) -> None:
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not server.start():
        logger.error("Failed to start Streamlit server")
        sys.exit(1)

    # Get icon path
    from nlsq.gui.desktop_config import get_desktop_config

    config = get_desktop_config()
    icon_path = (
        str(config.icon_path)
        if config.icon_path and config.icon_path.exists()
        else None
    )

    # Create and run the webview window
    logger.info("Opening desktop window")
    try:
        window = webview.create_window(
            title=config.window_title,
            url=server.url,
            width=width,
            height=height,
            resizable=True,
            min_size=(800, 600),
            text_select=True,
        )

        # Start webview event loop (blocks until window is closed)
        webview.start(debug=debug)
    finally:
        cleanup()


def run_in_browser(port: int | None = None) -> None:
    """Run the NLSQ GUI in the default web browser.

    This is a fallback mode when pywebview is not available.

    Parameters
    ----------
    port : int | None
        Port to run on. If None, finds an available port.
    """
    import webbrowser

    server = StreamlitServer(port)

    def cleanup() -> None:
        server.stop()

    atexit.register(cleanup)

    def signal_handler(signum: int, frame: Any) -> None:
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not server.start():
        logger.error("Failed to start Streamlit server")
        sys.exit(1)

    # Open browser
    webbrowser.open(server.url)

    logger.info(f"NLSQ GUI running at {server.url}")
    logger.info("Press Ctrl+C to stop")

    # Keep the main thread alive
    try:
        while server.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


def main() -> None:
    """Main entry point for the desktop application."""
    parser = argparse.ArgumentParser(
        description="NLSQ Curve Fitting Desktop Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run as desktop application
    python -m nlsq.gui.run_desktop

    # Run with custom window size
    python -m nlsq.gui.run_desktop --width 1600 --height 1000

    # Run in browser mode (fallback)
    python -m nlsq.gui.run_desktop --browser

    # Enable debug mode
    python -m nlsq.gui.run_desktop --debug
""",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1400,
        help="Window width in pixels (default: 1400)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Window height in pixels (default: 900)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for Streamlit server (default: auto-select)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Run in browser mode instead of desktop window",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with developer tools",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.browser:
        run_in_browser(args.port)
    else:
        try:
            run_with_webview(
                width=args.width,
                height=args.height,
                debug=args.debug,
            )
        except Exception as e:
            logger.warning(f"Desktop mode failed: {e}")
            logger.info("Falling back to browser mode")
            run_in_browser(args.port)


if __name__ == "__main__":
    main()
