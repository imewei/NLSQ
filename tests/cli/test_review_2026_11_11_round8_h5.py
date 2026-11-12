"""Round 8: the HDF5 unallocated-chunk check applies only to large declared sizes."""

import h5py
import numpy as np
import pytest

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import DataLoadError


def _cfg():
    return {"format": "hdf5", "hdf5": {"x_path": "x", "y_path": "y"}}


def test_small_partially_written_chunked_dataset_loads(tmp_path):
    p = tmp_path / "small.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=np.linspace(0, 1, 1000), chunks=(100,))
        dy = f.create_dataset(
            "y", shape=(1000,), dtype="f8", chunks=(100,), fillvalue=0.0
        )
        dy[500:] = 1.0
    x, y, _ = DataLoader().load(p, _cfg())
    assert x.shape == (1000,)
    assert y[:500].sum() == 0.0 and y[500:].sum() == 500.0


@pytest.mark.serial
def test_large_sparse_bomb_still_rejected(tmp_path):
    p = tmp_path / "bomb.h5"
    with h5py.File(p, "w") as f:
        for name in ("x", "y"):
            d = f.create_dataset(
                name,
                shape=(100_000_000,),
                dtype="f8",
                chunks=(1024,),
                compression="gzip",
            )
            d[:1024] = 1.0
    with pytest.raises(DataLoadError, match=r"unallocated"):
        DataLoader().load(p, _cfg())
