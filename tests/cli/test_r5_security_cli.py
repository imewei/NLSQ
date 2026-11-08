"""Round-5 security regressions: HDF5/NPZ loader limits and link rejection."""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import DataLoadError

h5py = pytest.importorskip("h5py")


def _hdf5_cfg(**kw):
    return {"hdf5": {"x_path": "/x", "y_path": "/y", **kw}}


class TestS3LoadLimit:
    def test_hdf5_oversized_dataset_rejected_before_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NLSQ_MAX_LOAD_ELEMENTS", "1000")
        p = tmp_path / "bomb.h5"
        with h5py.File(p, "w") as f:  # sparse: tiny file, huge declared shape
            f.create_dataset("x", shape=(5000,), dtype="f8", chunks=(100,))
            f.create_dataset("y", shape=(5000,), dtype="f8", chunks=(100,))
        with pytest.raises(DataLoadError, match="NLSQ_MAX_LOAD_ELEMENTS"):
            DataLoader()._load_hdf5(p, _hdf5_cfg())

    def test_npz_oversized_member_rejected_before_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NLSQ_MAX_LOAD_ELEMENTS", "1000")
        p = tmp_path / "bomb.npz"
        np.savez_compressed(p, x=np.zeros(5000), y=np.zeros(5000))
        with pytest.raises(DataLoadError, match="NLSQ_MAX_LOAD_ELEMENTS"):
            DataLoader()._load_npz(p, {})

    def test_small_files_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NLSQ_MAX_LOAD_ELEMENTS", raising=False)
        h = tmp_path / "ok.h5"
        with h5py.File(h, "w") as f:
            f["x"] = np.arange(5.0)
            f["y"] = np.arange(5.0) * 2
        x, y, _ = DataLoader()._load_hdf5(h, _hdf5_cfg())
        assert x.size == y.size == 5
        n = tmp_path / "ok.npz"
        np.savez(n, x=np.arange(5.0), y=np.arange(5.0))
        x, y, _ = DataLoader()._load_npz(n, {})
        assert x.size == y.size == 5

    def test_bad_env_value_is_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NLSQ_MAX_LOAD_ELEMENTS", "lots")
        n = tmp_path / "ok.npz"
        np.savez(n, x=np.arange(5.0), y=np.arange(5.0))
        with pytest.raises(DataLoadError, match="NLSQ_MAX_LOAD_ELEMENTS"):
            DataLoader()._load_npz(n, {})


class TestS4Links:
    def test_external_link_rejected(self, tmp_path):
        other = tmp_path / "other.h5"
        with h5py.File(other, "w") as f:
            f["x"] = np.arange(5.0)
            f["y"] = np.arange(5.0)
        p = tmp_path / "link.h5"
        with h5py.File(p, "w") as f:
            f["x"] = h5py.ExternalLink(str(other), "/x")
            f["y"] = h5py.ExternalLink(str(other), "/y")
        with pytest.raises(DataLoadError, match="external"):
            DataLoader()._load_hdf5(p, _hdf5_cfg())

    def test_external_group_link_rejected(self, tmp_path):
        other = tmp_path / "other.h5"
        with h5py.File(other, "w") as f:
            f["g/x"] = np.arange(5.0)
            f["g/y"] = np.arange(5.0)
        p = tmp_path / "glink.h5"
        with h5py.File(p, "w") as f:
            f["g"] = h5py.ExternalLink(str(other), "/g")
        with pytest.raises(DataLoadError, match="external"):
            DataLoader()._load_hdf5(p, _hdf5_cfg(x_path="/g/x", y_path="/g/y"))

    def test_virtual_dataset_rejected(self, tmp_path):
        src = tmp_path / "src.h5"
        with h5py.File(src, "w") as f:
            f["d"] = np.arange(5.0)
        p = tmp_path / "virt.h5"
        with h5py.File(p, "w") as f:
            for name in ("x", "y"):
                layout = h5py.VirtualLayout(shape=(5,), dtype="f8")
                layout[:] = h5py.VirtualSource(str(src), "d", shape=(5,))
                f.create_virtual_dataset(name, layout)
        with pytest.raises(DataLoadError, match="virtual"):
            DataLoader()._load_hdf5(p, _hdf5_cfg())
