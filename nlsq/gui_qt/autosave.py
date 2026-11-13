"""
NLSQ Qt GUI Autosave Manager

This module provides automatic session state saving and crash recovery
functionality for the Qt GUI application.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import tempfile
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from nlsq.gui_qt.session_state import SessionState, n_points

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState

__all__ = ["AutosaveManager"]

logger = logging.getLogger(__name__)

# Autosave directory: per-user on POSIX (the system temp dir is shared, so a
# fixed name lets another local user pre-create it).
_UID_SUFFIX = f"_{os.getuid()}" if hasattr(os, "getuid") else ""
AUTOSAVE_DIR = Path(tempfile.gettempdir()) / f"nlsq_autosave{_UID_SUFFIX}"
AUTOSAVE_FILE = AUTOSAVE_DIR / "session_autosave.json"
AUTOSAVE_INTERVAL_MS = 60000  # 1 minute

# Scalar workflow settings are safe to recover without evaluating model code.
# Runtime state and callables must never be restored from a crash snapshot.
_RUNTIME_FIELDS = frozenset({"model_func", "fit_result", "fit_running", "fit_aborted"})
_SETTING_DEFAULTS = {
    f.name: f.default
    for f in fields(SessionState)
    if f.name not in _RUNTIME_FIELDS and isinstance(f.default, (str, bool, int, float))
}
_SETTING_TYPES = {name: (type(value),) for name, value in _SETTING_DEFAULTS.items()}
_SETTING_TYPES.update(
    {
        "data_file_name": (str, type(None)),
        "custom_file_path": (str, type(None)),
        "checkpoint_dir": (str, type(None)),
        "defense_preset": (str, type(None)),
        "batch_max_workers": (int, type(None)),
        **{
            name: (int, str, type(None))
            for name in ("x_column", "y_column", "z_column", "sigma_column")
        },
    }
)


def _valid_settings(settings: Any) -> dict[str, Any]:
    """Copy known JSON configuration values, excluding runtime objects."""
    if not isinstance(settings, dict):
        return {}
    result = {
        name: value
        for name, value in settings.items()
        if name in _SETTING_TYPES and type(value) in _SETTING_TYPES[name]
    }
    transforms = settings.get("transforms")
    if isinstance(transforms, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in transforms.items()
    ):
        result["transforms"] = dict(transforms)
    return result


def _restore_settings(state: SessionState, settings: Any) -> None:
    """Accept only known scalar settings with their original types."""
    for name, value in _valid_settings(settings).items():
        setattr(state, name, value)


def _restore_model(state: SessionState) -> None:
    """Rebuild shipped models; recovered custom code requires explicit Apply."""
    state.model_func = None
    if state.model_type not in {"builtin", "polynomial"} or not state.model_config:
        return
    from nlsq.cli.errors import ModelError
    from nlsq.gui_qt.adapters.model_adapter import get_model

    try:
        state.model_func = get_model(state.model_type, state.model_config)
    except (ValueError, TypeError, KeyError, AttributeError, ModelError) as exc:
        logger.warning("Could not restore model selection: %s", exc)


def _ensure_private_dir(path: Path) -> None:
    """Create ``path`` (mode 0o700) and verify we own it and it is no symlink.

    The ownership/symlink checks are POSIX-only; Windows temp dirs are
    already per-user.

    Raises:
        PermissionError: If ``path`` exists but is not a directory owned by us.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not hasattr(os, "getuid"):
        return
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise PermissionError(f"{path} is not a directory owned by the current user")
    if st.st_mode & 0o077:
        path.chmod(0o700)


def _write_autosave(data: dict[str, Any]) -> None:
    """Atomically replace the recovery file with a private temporary file."""
    fd, tmp_name = tempfile.mkstemp(
        dir=AUTOSAVE_FILE.parent, prefix=AUTOSAVE_FILE.stem + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_name, AUTOSAVE_FILE)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class AutosaveManager(QObject):
    """Manages automatic session saving and recovery.

    Provides:
    - Periodic autosave of session state
    - Crash recovery on startup
    - Clean shutdown handling
    """

    # Signals
    autosave_completed = Signal()
    recovery_available = Signal(str)  # timestamp of recovery file

    _autosave_warned: bool = False  # warn once per manager when autosave fails

    def __init__(self, app_state: AppState, parent: QObject | None = None) -> None:
        """Initialize the autosave manager.

        Args:
            app_state: Application state manager
            parent: Parent QObject
        """
        super().__init__(parent)
        self._app_state = app_state
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._do_autosave)

        # Ensure autosave directory exists and is private to this user; if the
        # shared location is unusable, fall back to a fresh private directory.
        global AUTOSAVE_DIR, AUTOSAVE_FILE  # noqa: PLW0603 - module-level paths are the test/crash-handler hook
        try:
            _ensure_private_dir(AUTOSAVE_DIR)
        except OSError as exc:
            AUTOSAVE_DIR = Path(tempfile.mkdtemp(prefix="nlsq_autosave_"))
            AUTOSAVE_FILE = AUTOSAVE_DIR / AUTOSAVE_FILE.name
            logger.warning(
                "Autosave directory unusable (%s); using %s. Crash recovery will "
                "not find this session after a restart.",
                exc,
                AUTOSAVE_DIR,
            )

    def start(self) -> None:
        """Start the autosave timer."""
        self._timer.start(AUTOSAVE_INTERVAL_MS)

    def stop(self) -> None:
        """Stop the autosave timer."""
        self._timer.stop()

    def check_recovery(self) -> bool:
        """Check if a recovery file exists.

        Returns:
            True if recovery file exists and is recent
        """
        if not AUTOSAVE_FILE.exists():
            return False

        try:
            data = json.loads(AUTOSAVE_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return False
            timestamp = data.get("timestamp")
            if timestamp:
                self.recovery_available.emit(timestamp)
                return True
        except (OSError, json.JSONDecodeError, ValueError, KeyError, RecursionError):
            pass

        return False

    def recover(self) -> bool:
        """Attempt to recover session from autosave.

        Returns:
            True if recovery was successful
        """
        if not AUTOSAVE_FILE.exists():
            return False

        try:
            data = json.loads(AUTOSAVE_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return False
            self._restore_state(data)
            return True
        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RecursionError,
        ):
            return False

    def clear_recovery(self) -> None:
        """Clear the recovery file."""
        if AUTOSAVE_FILE.exists():
            AUTOSAVE_FILE.unlink()

    def save_now(self) -> None:
        """Perform an immediate save."""
        self._do_autosave()

    def _do_autosave(self) -> None:
        """Perform the autosave operation."""
        try:
            data = self._serialize_state()
            _write_autosave(data)
            self.autosave_completed.emit()
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            # Best-effort: never interrupt the user, but say so once per manager.
            if not self._autosave_warned:
                self._autosave_warned = True
                logger.warning(
                    "Autosave failed (%s); crash recovery may be stale.", exc
                )

    def _serialize_state(self) -> dict[str, Any]:
        """Serialize current session state to a dictionary.

        Returns:
            Serializable dictionary
        """
        state = self._app_state.state
        data: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        # Serialize data arrays (skip large datasets to avoid OOM)
        max_autosave_points = 100_000
        if state.ydata is not None and len(state.ydata) > max_autosave_points:
            # Recovery will come back without data; make that explicit.
            data["data_skipped"] = len(state.ydata)
            logger.warning(
                "Autosave omits data: %d points exceed the %d-point autosave limit.",
                len(state.ydata),
                max_autosave_points,
            )
        # xdata is (n,) or, for 2-D surface data, (n_coords, n)
        if state.xdata is not None and n_points(state.xdata) <= max_autosave_points:
            data["xdata"] = np.asarray(state.xdata).tolist()
        if state.ydata is not None and len(state.ydata) <= max_autosave_points:
            data["ydata"] = np.asarray(state.ydata).tolist()
        if state.sigma is not None and len(state.sigma) <= max_autosave_points:
            data["sigma"] = np.asarray(state.sigma).tolist()

        # Serialize model config
        if state.model_config is not None:
            data["model_config"] = state.model_config

        # Serialize fitting options
        if state.p0 is not None:
            data["p0"] = [float(v) for v in state.p0]
        if state.bounds is not None:
            # None (= unbounded) is kept as JSON null; +/-inf use json's Infinity
            with contextlib.suppress(TypeError, ValueError):
                data["bounds"] = [
                    [None if v is None else float(v) for v in state.bounds[0]],
                    [None if v is None else float(v) for v in state.bounds[1]],
                ]

        # Serialize other state attributes
        data["auto_p0"] = getattr(state, "auto_p0", True)
        data["settings"] = _valid_settings(
            {
                name: getattr(state, name, None)
                for name in (*_SETTING_TYPES, "transforms")
            }
        )

        return data

    @staticmethod
    def _validate_restored_array(
        raw: Any, *, positive: bool = False, max_ndim: int = 1
    ) -> np.ndarray | None:
        """Validate an array restored from autosave JSON.

        Args:
            raw: Raw list from the serialized dictionary.
            max_ndim: 1 for y/sigma; 2 for xdata (2-D surface data).
            positive: If True, require strictly positive values (accepting
                +inf, the SciPy convention for "no weight" on sigma) instead
                of the default finite-values check.
        """
        arr = np.array(raw, dtype=np.float64)
        if not 1 <= arr.ndim <= max_ndim:
            return None
        valid = np.all(arr > 0) if positive else np.all(np.isfinite(arr))
        return arr if valid else None

    def _restore_state(self, data: dict[str, Any]) -> None:
        """Restore session state from a dictionary.

        Args:
            data: Serialized state dictionary
        """
        state = self._app_state.state

        # Restoring session state mutates `state` directly rather than going
        # through set_data()/set_model(), so it must invalidate any existing
        # fit result itself -- otherwise a fit_result from before recovery
        # would silently survive paired with the restored data/model.
        self._app_state._clear_fit_result()

        # Restore data arrays — validate before accepting to avoid emitting
        # data_changed with NaN/Inf-contaminated or malformed arrays.
        # All-or-nothing: a partial restore would pair the wrong x/y/sigma.
        raw = {key: data[key] for key in ("xdata", "ydata", "sigma") if key in data}
        restored: dict[str, np.ndarray] = {}
        for key, value in raw.items():
            try:
                arr = self._validate_restored_array(
                    value, positive=key == "sigma", max_ndim=2 if key == "xdata" else 1
                )
            except (TypeError, ValueError):
                arr = None
            if arr is not None:
                restored[key] = arr
        # Cross-validate lengths — independently-valid arrays of mismatched
        # length would silently pair the wrong sigma/y with the wrong x.
        lengths = {a.shape[-1] for a in restored.values()}
        if len(restored) < len(raw) or len(lengths) > 1:
            logger.warning(
                "Autosave recovery dropped all data arrays: %s invalid or "
                "inconsistent in the recovery file.",
                ", ".join(k for k in raw if k not in restored) or "lengths of arrays",
            )
        else:
            for key, arr in restored.items():
                setattr(state, key, arr)
        if "data_skipped" in data:
            logger.warning(
                "Autosave recovery has no data: the %s-point dataset exceeded "
                "the autosave limit and must be reloaded.",
                data["data_skipped"],
            )

        # Restore model config
        if isinstance(data.get("model_config"), dict):
            state.model_config = data["model_config"]

        # Restore fitting options
        p0 = data.get("p0")
        if isinstance(p0, list) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in p0
        ):
            state.p0 = [float(v) for v in p0]
        if "bounds" in data:
            try:
                lb, ub = data["bounds"]
                state.bounds = (
                    [None if v is None else float(v) for v in lb],
                    [None if v is None else float(v) for v in ub],
                )
            except (TypeError, ValueError, IndexError):
                pass

        # Restore other attributes
        if isinstance(data.get("auto_p0"), bool):
            state.auto_p0 = data["auto_p0"]
        _restore_settings(state, data.get("settings"))
        _restore_model(state)

        # Emit signals to update UI
        if state.xdata is not None:
            self._app_state.data_changed.emit()
        if state.model_config is not None:
            self._app_state.model_changed.emit()
        if state.xdata is not None or state.model_config is not None:
            self._app_state.page_access_changed.emit()
        self._app_state.session_restored.emit()
