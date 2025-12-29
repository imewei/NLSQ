"""Fast, non-Streamlit GUI utility tests."""

from __future__ import annotations

import importlib
import socket
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

gui_desktop = importlib.import_module("nlsq.gui.run_desktop")


@pytest.mark.gui
def test_find_free_port_runtime_error_when_blocked() -> None:
    """find_free_port should raise if start port is unavailable and max_attempts=1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

        with pytest.raises(RuntimeError):
            gui_desktop.find_free_port(start_port=port, max_attempts=1)


@pytest.mark.gui
def test_wait_for_server_times_out_fast() -> None:
    """wait_for_server should return False quickly for an unused port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    assert gui_desktop.wait_for_server(port, timeout=0.1, interval=0.05) is False


@pytest.mark.gui
def test_get_streamlit_args_contains_port() -> None:
    """get_streamlit_args should include the port flag and value."""
    port = 8601
    args = gui_desktop.get_streamlit_args(port)

    assert "--server.port" in args
    assert str(port) in args


@pytest.mark.gui
def test_streamlit_server_stop_kills_on_timeout() -> None:
    """stop should kill the process if terminate times out."""
    server = gui_desktop.StreamlitServer(port=8501)

    proc = MagicMock()
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
    server.process = proc

    server.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


@pytest.mark.gui
def test_run_with_webview_import_error_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_with_webview should exit with code 1 when pywebview is missing."""
    original_import = __import__

    def _import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name == "webview":
            raise ImportError("no webview")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    monkeypatch.setattr(builtins := sys.modules["builtins"], "__import__", _import)

    with pytest.raises(SystemExit) as excinfo:
        gui_desktop.run_with_webview()

    assert excinfo.value.code == 1


@pytest.mark.gui
def test_run_in_browser_exits_on_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_in_browser should exit when the server fails to start."""
    server = SimpleNamespace(start=lambda: False, stop=lambda: None)
    monkeypatch.setattr(gui_desktop, "StreamlitServer", lambda *_a, **_k: server)
    monkeypatch.setitem(
        sys.modules, "webbrowser", SimpleNamespace(open=lambda *_a, **_k: None)
    )

    with pytest.raises(SystemExit) as excinfo:
        gui_desktop.run_in_browser(port=8601)

    assert excinfo.value.code == 1


@pytest.mark.gui
def test_main_falls_back_to_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """main should fall back to browser mode if desktop mode fails."""
    calls: list[int | None] = []

    def _run_with_webview(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    def _run_in_browser(port: int | None = None) -> None:
        calls.append(port)

    monkeypatch.setattr(gui_desktop, "run_with_webview", _run_with_webview)
    monkeypatch.setattr(gui_desktop, "run_in_browser", _run_in_browser)
    monkeypatch.setattr(sys, "argv", ["prog"])

    gui_desktop.main()

    assert calls == [None]
