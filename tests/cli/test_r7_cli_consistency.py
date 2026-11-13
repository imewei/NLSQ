"""R7: CLI load cap default, umask-free atomic write, absolute_sigma rule."""

import os
import stat
import sys
import zipfile

import numpy as np
import pytest

from nlsq.cli.data_loaders import DataLoader, _max_load_elements
from nlsq.cli.errors import ConfigError, DataLoadError
from nlsq.cli.result_exporter import _atomic_write
from nlsq.cli.workflow_runner import WorkflowRunner


def test_default_load_cap_supports_100m_plus_points(monkeypatch, tmp_path):
    monkeypatch.delenv("NLSQ_MAX_LOAD_ELEMENTS", raising=False)
    assert _max_load_elements(tmp_path / "f", "hdf5") >= 2_000_000_000


def test_env_still_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("NLSQ_MAX_LOAD_ELEMENTS", "7")
    assert _max_load_elements(tmp_path / "f", "hdf5") == 7


def test_hdf5_boundary_uses_total_elements_of_2d_dataset(monkeypatch, tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("/data/x", data=np.ones((2, 50)))
        f.create_dataset("/data/y", data=np.ones(100))
    cfg = {"format": "hdf5", "hdf5": {"x_path": "/data/x", "y_path": "/data/y"}}
    monkeypatch.setenv("NLSQ_MAX_LOAD_ELEMENTS", "100")
    DataLoader().load(p, cfg)  # exactly at the limit: accepted
    monkeypatch.setenv("NLSQ_MAX_LOAD_ELEMENTS", "99")
    with pytest.raises(DataLoadError, match="load limit"):
        DataLoader().load(p, cfg)


def test_npz_member_uncompressed_size_is_checked(monkeypatch, tmp_path):
    """A header that understates the member's real payload is still rejected."""
    p = tmp_path / "lie.npz"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in ("x", "y"):
            with zf.open(key + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array_header_1_0(
                    fh,
                    {"descr": "<f8", "fortran_order": False, "shape": (3,)},
                )
                fh.write(np.ones(3).tobytes())
                if key == "x":
                    fh.write(np.zeros(500).tobytes())  # undeclared trailing payload
    cfg = {"format": "npz", "npz": {"x_key": "x", "y_key": "y"}}
    monkeypatch.setenv("NLSQ_MAX_LOAD_ELEMENTS", "100")
    with pytest.raises(DataLoadError, match="load limit"):
        DataLoader().load(p, cfg)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_atomic_write_mode_follows_umask_without_toggling_it(monkeypatch, tmp_path):
    old = os.umask(0o022)
    try:

        def boom(*a, **k):
            raise AssertionError("os.umask must not be called at write time")

        monkeypatch.setattr(os, "umask", boom)
        target = tmp_path / "out.json"
        _atomic_write(target, "{}")
        assert stat.S_IMODE(target.stat().st_mode) == 0o644
        os.chmod(target, 0o600)
        _atomic_write(target, "{}")  # existing file keeps its mode
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        monkeypatch.undo()
        os.umask(old)


def _extract(fitting):
    x = np.linspace(0, 1, 10)

    def model(x, a, b):
        return a * x + b

    return WorkflowRunner()._extract_fit_params(
        {"fitting": {"p0": [1.0, 0.0], **fitting}}, x, 2 * x + 1, model
    )


@pytest.mark.parametrize("value", [True, False, 0, 1, np.bool_(True)])
def test_absolute_sigma_accepts_bool_like(value):
    _, _, kw = _extract({"absolute_sigma": value})
    assert kw["absolute_sigma"] is bool(value)


@pytest.mark.parametrize("value", ["false", 2, -1, 1.0, 0.0, [True]])
def test_absolute_sigma_rejects_everything_else(value):
    with pytest.raises(ConfigError, match="absolute_sigma"):
        _extract({"absolute_sigma": value})
