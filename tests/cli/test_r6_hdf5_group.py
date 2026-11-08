"""R6-02: an HDF5 x_path that names a group is a DataLoadError."""

import pytest

h5py = pytest.importorskip("h5py")

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import DataLoadError


def _cfg(x_path):
    return {
        "format": "hdf5",
        "hdf5": {"x_path": x_path, "y_path": "/data/y"},
    }


@pytest.fixture
def h5file(tmp_path):
    p = tmp_path / "g.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("/data/x", data=[1.0, 2.0, 3.0])
        f.create_dataset("/data/y", data=[1.0, 2.0, 3.0])
        f["/dangling"] = h5py.SoftLink("/nowhere")
    return p


def test_group_path_raises_dataload_error(h5file):
    with pytest.raises(DataLoadError, match="not a dataset"):
        DataLoader().load(h5file, _cfg("/data"))


def test_dangling_softlink_raises_dataload_error(h5file):
    with pytest.raises(DataLoadError):
        DataLoader().load(h5file, _cfg("/dangling"))
