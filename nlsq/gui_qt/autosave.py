"""
NLSQ Qt GUI Autosave Manager

This module provides automatic session state saving and crash recovery
functionality for the Qt GUI application.

Recovery safety: a snapshot chosen for recovery is identified by (inode,
mtime_ns, size) and owner token, and is removed by an atomic rename to a unique
``*.rm`` tombstone that is verified before deletion (a mismatching file is moved
back with ``os.link`` only if the path is still free, or an existence-checked
``os.rename`` on filesystems without hard links; leftover tombstones are adopted
at the next discovery when their canonical file is absent). Residual limit without a
lock file: for the microseconds between that rename and the restore, a live
instance's newer snapshot is absent from its path; if it autosaves in that
window its newer file wins and the older one stays as an ignored ``*.rm`` file,
and a crash of that instance inside the window loses its snapshot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import tempfile
import uuid
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


# File the running session writes; the crash hook updates this one, not the shared name.
_active_target: Path | None = None
_active_token: str | None = None  # owner token this session wrote into that file


class _ExclusiveUnsupported(OSError):
    """The filesystem cannot publish a file without replacing an existing one."""


def _write_autosave(
    data: dict[str, Any], path: Path | None = None, *, exclusive: bool = False
) -> None:
    """Atomically replace the recovery file with a private temporary file.

    ``exclusive`` publishes with a hard link, which fails with FileExistsError
    instead of replacing a file another session published meanwhile. Where hard
    links are unsupported it raises ``_ExclusiveUnsupported`` (nothing is
    written): the caller must then use a name no other session can choose.
    """
    path = path or AUTOSAVE_FILE
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=AUTOSAVE_FILE.stem + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # dumps() without indent uses the C encoder; dump() streams in pure
            # Python (~4x slower on a 100k-point snapshot, on the GUI thread)
            f.write(json.dumps(data))
            f.flush()
            os.fsync(f.fileno())  # data must reach disk before the rename does
        if exclusive:
            try:
                os.link(tmp_name, path)
            except FileExistsError:
                raise
            except OSError as exc:  # no hard links on this filesystem
                raise _ExclusiveUnsupported(str(exc)) from exc
            else:
                os.unlink(tmp_name)
        else:
            os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _read_snapshot(path: Path) -> dict[str, Any] | None:
    """The snapshot dict in ``path``, or None if absent, unreadable or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _file_identity(path: Path) -> tuple[int, int, int] | None:
    """(inode, mtime_ns, size) of ``path``; changes whenever it is rewritten."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _owned_by_live_session(data: dict[str, Any] | None, token: str) -> bool:
    """True if ``data`` was written by another still-running session.

    Total: a malformed owner/pid is "not alive" and never raises.

    A file from this very process is never "another session": one process owns
    one window, so a different token here means an earlier manager of ours.

    shortcut: pid liveness only (a recycled pid looks alive, which merely makes
    a stale snapshot ignored); upgrade to a lock file if that ever matters.
    """
    owner = (data or {}).get("owner")
    if not isinstance(owner, dict) or owner.get("token") == token:
        return False
    pid = owner.get("pid")
    # type() rejects bool/float/str; the range keeps psutil's C call from overflowing
    if type(pid) is not int or not 1 <= pid < 2**31 or pid == os.getpid():
        return False
    import psutil

    try:
        return psutil.pid_exists(pid)
    except (OverflowError, ValueError, OSError, psutil.Error):
        return False  # an unusable pid cannot be a live session


def _restore_tombstone(tomb: Path, path: Path) -> bool:
    """Put ``tomb`` back at ``path`` unless ``path`` is taken; never deletes ``tomb`` on failure.

    Uses a hard link (fails atomically if ``path`` exists). Filesystems without
    hard links (exFAT, some network mounts) fall back to a rename guarded by an
    existence check: that check-then-rename window is tiny but not atomic.
    """
    try:
        os.link(tomb, path)
    except FileExistsError:
        logger.warning(
            "Snapshot %s was replaced during recovery cleanup; its previous "
            "version stays as %s until its owner writes again.",
            path,
            tomb,
        )
        return False
    except OSError:  # no hard links here (EOPNOTSUPP, EPERM, EXDEV, ENOSYS, ...)
        if path.exists():
            logger.warning("Snapshot %s was replaced; %s kept.", path, tomb)
            return False
        try:
            os.rename(tomb, path)
        except OSError:
            logger.warning("Could not restore snapshot %s from %s.", path, tomb)
            return False
        return True
    with contextlib.suppress(OSError):
        tomb.unlink()
    return True


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
    # Single-owner contract: one manager per session. Class-level defaults keep
    # these valid for tests that build a manager without __init__.
    _token: str = ""  # identifies this session inside the files it writes
    _own_file: Path | None = None  # file this session writes (see _target_file)
    _own_file_base: Path | None = None  # AUTOSAVE_FILE the choice was made for
    _claimed_base: Path | None = None  # AUTOSAVE_FILE this session already published to
    _recovery_file: Path | None = None  # snapshot offered/adopted on startup
    _recovery_identity: tuple[int, int, int] | None = None  # its identity when chosen
    _recovery_token: Any = None  # owner token inside it when chosen

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

    def _session_token(self) -> str:
        if not self._token:
            self._token = uuid.uuid4().hex
        return self._token

    def _target_file(self) -> Path:
        """File this session writes: the shared one, or a private sibling.

        A second running instance must not overwrite (or, on close, delete) the
        first one's snapshot, so it writes ``<stem>.<token><suffix>`` instead.
        """
        if self._own_file is None or self._own_file_base != AUTOSAVE_FILE:
            token = self._session_token()
            held = _owned_by_live_session(_read_snapshot(AUTOSAVE_FILE), token)
            self._own_file_base = AUTOSAVE_FILE
            self._own_file = (
                AUTOSAVE_FILE.with_name(
                    f"{AUTOSAVE_FILE.stem}.{token}{AUTOSAVE_FILE.suffix}"
                )
                if held
                else AUTOSAVE_FILE
            )
        return self._own_file

    def _adopt_orphan_tombstones(self) -> None:
        """Move leftover ``*.rm`` tombstones back to their canonical name.

        A failed restore (or a crash between rename and restore) must not make a
        snapshot permanently undiscoverable. Only when the canonical file is
        absent; a tombstone of ours is never touched.
        """
        token = self._session_token()
        for tomb in AUTOSAVE_FILE.parent.glob(f"{AUTOSAVE_FILE.stem}.*.rm"):
            canonical = tomb.with_name(tomb.name.rsplit(".", 2)[0])
            data = _read_snapshot(tomb)
            owner = (data or {}).get("owner")
            if data is None or (
                isinstance(owner, dict) and owner.get("token") == token
            ):
                continue
            if not canonical.exists():
                _restore_tombstone(tomb, canonical)

    def _recovery_candidates(self) -> list[Path]:
        """Snapshots of sessions that are no longer running, newest first."""
        self._adopt_orphan_tombstones()
        token = self._session_token()
        paths = [
            AUTOSAVE_FILE,
            *AUTOSAVE_FILE.parent.glob(f"{AUTOSAVE_FILE.stem}.*{AUTOSAVE_FILE.suffix}"),
        ]
        found = []
        for path in paths:
            data = _read_snapshot(path)
            if data is None or _owned_by_live_session(data, token):
                continue
            owner = data.get("owner")  # a malformed owner counts as absent
            if isinstance(owner, dict) and owner.get("token") == token:
                continue  # our own file is not a crash to recover from
            found.append((str(data.get("timestamp") or ""), path))
        return [path for _, path in sorted(found, reverse=True)]

    def _select_recovery(self, path: Path | None) -> None:
        self._recovery_file = path
        self._recovery_identity = None if path is None else _file_identity(path)
        owner = None if path is None else (_read_snapshot(path) or {}).get("owner")
        self._recovery_token = owner.get("token") if isinstance(owner, dict) else None

    def _remove_selected_recovery(self) -> None:
        """Delete the selected snapshot only if it is still the file selected.

        The rename is atomic, so exactly the file at the path at that instant is
        taken; it is deleted only when its identity and owner match the
        selection, otherwise (a live instance replaced it) it is put back.
        """
        path = self._recovery_file
        if path is None:
            return
        tomb = path.with_name(f"{path.name}.{uuid.uuid4().hex}.rm")
        try:
            os.rename(path, tomb)
        except OSError:
            return  # already gone
        data = _read_snapshot(tomb)
        owner = (data or {}).get("owner")
        token = owner.get("token") if isinstance(owner, dict) else None
        if (
            _file_identity(tomb) == self._recovery_identity
            and token == self._recovery_token
            and not _owned_by_live_session(data, self._session_token())
        ):
            with contextlib.suppress(OSError):
                tomb.unlink()
            return
        _restore_tombstone(tomb, path)

    def _remove_if_owned(self, path: Path) -> None:
        """Delete ``path`` only if it carries this session's owner token.

        The file is first renamed away (atomic), checked, and then deleted or
        put back, so a snapshot another session published at that path is
        never removed.
        """
        tomb = path.with_name(f"{path.name}.{uuid.uuid4().hex}.rm")
        try:
            os.rename(path, tomb)
        except OSError:
            return  # already gone
        owner = (_read_snapshot(tomb) or {}).get("owner")
        if isinstance(owner, dict) and owner.get("token") == self._session_token():
            with contextlib.suppress(OSError):
                tomb.unlink()
        else:
            _restore_tombstone(tomb, path)

    def _recovery_still_selected(self) -> bool:
        """True if the chosen snapshot is the very file chosen and still unowned.

        Another instance may have replaced it (new inode/mtime) since the prompt
        opened; acting on the path alone would then touch ITS snapshot.
        """
        path = self._recovery_file
        if path is None or _file_identity(path) != self._recovery_identity:
            return False
        return not _owned_by_live_session(_read_snapshot(path), self._session_token())

    def check_recovery(self) -> bool:
        """Check if a recovery file exists.

        Returns:
            True if a snapshot of an ended session exists
        """
        for path in self._recovery_candidates():
            timestamp = (_read_snapshot(path) or {}).get("timestamp")
            if timestamp:
                self._select_recovery(path)
                self.recovery_available.emit(timestamp)
                return True
        return False

    def recover(self) -> bool:
        """Attempt to recover session from autosave.

        The snapshot stays on disk until this session's first autosave has
        replaced it (see ``_do_autosave``), so a second crash right after
        recovery does not lose it.

        Returns:
            True if recovery was successful
        """
        if self._recovery_file is None:
            candidates = self._recovery_candidates()
            self._select_recovery(candidates[0] if candidates else None)
        elif not self._recovery_still_selected():
            logger.warning("Recovery snapshot changed since it was offered; skipped.")
            self._select_recovery(None)
        path = self._recovery_file
        if path is None:
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return False
            if _file_identity(path) != self._recovery_identity:
                return False  # replaced while we read it: do not adopt
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
        """Remove this session's own snapshot and any snapshot it adopted.

        Another running session's file is never touched.
        """
        own = self._own_file if self._own_file_base == AUTOSAVE_FILE else None
        if own is not None:
            self._remove_if_owned(own)
        if self._recovery_file != own:
            self._remove_selected_recovery()
        self._select_recovery(None)

    def save_now(self) -> None:
        """Perform an immediate save."""
        self._do_autosave()

    def _do_autosave(self) -> None:
        """Perform the autosave operation."""
        try:
            data = self._serialize_state()
            data["owner"] = {"pid": os.getpid(), "token": self._session_token()}
            target = self._target_file()
            if target == AUTOSAVE_FILE and self._claimed_base != AUTOSAVE_FILE:
                # First publish to the shared name: claim it atomically so two
                # sessions starting together cannot both think they own it.
                try:
                    _write_autosave(data, target, exclusive=True)
                except (FileExistsError, _ExclusiveUnsupported):
                    # Never replace a file this session did not create (it may
                    # be another session's, live or stale): use the private
                    # name; a stale snapshot is cleaned up identity-aware via
                    # the recovery selection below.
                    target = self._own_file = AUTOSAVE_FILE.with_name(
                        f"{AUTOSAVE_FILE.stem}.{self._session_token()}"
                        f"{AUTOSAVE_FILE.suffix}"
                    )
                    _write_autosave(data, target)
                self._claimed_base = AUTOSAVE_FILE if target == AUTOSAVE_FILE else None
            else:
                _write_autosave(data, target)
            global _active_target, _active_token
            _active_target, _active_token = target, self._session_token()
            # The recovered snapshot is superseded only once this one is on disk.
            if self._recovery_file != target:
                self._remove_selected_recovery()
            self._select_recovery(None)
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

    @staticmethod
    def _clear_data(state: Any) -> None:
        state.xdata = state.ydata = state.sigma = None

    def _restore_data(self, state: Any, data: dict[str, Any]) -> None:
        """Restore x/y/sigma as one tuple (see ``_restore_state``)."""
        # Restore data arrays — validate before accepting to avoid emitting
        # data_changed with NaN/Inf-contaminated or malformed arrays.
        # All-or-nothing: a partial restore would pair the wrong x/y/sigma.
        raw = {key: data[key] for key in ("xdata", "ydata", "sigma") if key in data}
        if ("xdata" in raw) != ("ydata" in raw):
            # half a pair (and any sigma with it) must never meet the old
            # y/x: keep the existing data untouched
            logger.warning(
                "Autosave recovery ignored its data: it has %s but not %s.",
                "xdata" if "xdata" in raw else "ydata",
                "ydata" if "xdata" in raw else "xdata",
            )
            raw = {}
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
        # sigma is only meaningful with the x/y it was recorded for: never pair
        # a recovered sigma with the current data.
        if "sigma" in restored and not {"xdata", "ydata"} & restored.keys():
            logger.warning("Autosave recovery dropped sigma: no x/y data with it.")
            del restored["sigma"]
            raw = {k: v for k, v in raw.items() if k != "sigma"}
        # Cross-validate lengths — independently-valid arrays of mismatched
        # length would silently pair the wrong sigma/y with the wrong x.
        lengths = {a.shape[-1] for a in restored.values()}
        if len(restored) < len(raw) or len(lengths) > 1:
            logger.warning(
                "Autosave recovery dropped all data arrays: %s invalid or "
                "inconsistent in the recovery file.",
                ", ".join(k for k in raw if k not in restored) or "lengths of arrays",
            )
            self._clear_data(state)  # never pair the recovered model with old data
        else:
            if "xdata" in restored or "ydata" in restored:
                # a recovered dataset replaces the old x/y/sigma as one tuple
                state.sigma = None
            for key, arr in restored.items():
                setattr(state, key, arr)
        if "data_skipped" in data:
            logger.warning(
                "Autosave recovery has no data: the %s-point dataset exceeded "
                "the autosave limit and must be reloaded.",
                data["data_skipped"],
            )
            self._clear_data(state)

    def _restore_state(self, data: dict[str, Any]) -> None:
        """Restore session state from a dictionary.

        Args:
            data: Serialized state dictionary
        """
        state = self._app_state.state

        # Convert the fit options first: a malformed value must not leave the
        # session half-restored (data replaced, result cleared, then an error).
        p0 = data.get("p0")
        restored_p0 = None
        if isinstance(p0, list) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in p0
        ):
            try:
                restored_p0 = [float(v) for v in p0]
            except OverflowError:  # an int too large for float64
                logger.warning("Autosave recovery ignored an out-of-range p0.")
        restored_bounds = None
        if "bounds" in data:
            try:
                lb, ub = data["bounds"]
                restored_bounds = (
                    [None if v is None else float(v) for v in lb],
                    [None if v is None else float(v) for v in ub],
                )
            except (TypeError, ValueError, IndexError, OverflowError):
                pass

        # Restoring session state mutates `state` directly rather than going
        # through set_data()/set_model(), so it must invalidate any existing
        # fit result itself -- otherwise a fit_result from before recovery
        # would silently survive paired with the restored data/model.
        self._app_state._clear_fit_result()
        # Inputs change here without going through set_data()/set_model(): bump
        # the epoch so a fit still running cannot attach its result to them.
        self._app_state.input_epoch += 1

        self._restore_data(state, data)

        # Restore model config
        if isinstance(data.get("model_config"), dict):
            state.model_config = data["model_config"]

        # Restore fitting options
        if restored_p0 is not None:
            state.p0 = restored_p0
        if restored_bounds is not None:
            state.bounds = restored_bounds

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
