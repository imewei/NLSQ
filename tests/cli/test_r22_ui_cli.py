"""Round-22 regressions: CSV quoting, validator blocklist, YAML scalars, export."""

import numpy as np
import pytest

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.model_validation import validate_model


def _load(tmp_path, text, cfg):
    p = tmp_path / "q.csv"
    p.write_text(text)
    return DataLoader().load(p, cfg)


def test_csv_quoted_field_with_delimiter_does_not_shift_columns(tmp_path):
    x, y, _ = _load(
        tmp_path,
        'label,x,y\n"a, b",1,2\n"c, d",2,4\n"e, f",3,6\n',
        {
            "format": "csv",
            "csv": {"header": False, "skip_header": 1},
            "columns": {"x": 1, "y": 2},
        },
    )
    np.testing.assert_allclose(x, [1, 2, 3])
    np.testing.assert_allclose(y, [2, 4, 6])


def test_csv_quoted_numeric_with_thousands_separator_is_not_silently_split(tmp_path):
    try:
        x, y, _ = _load(
            tmp_path,
            '"1,000",2.5\n"2,000",3.5\n"3,000",4.5\n',
            {"format": "csv", "csv": {"header": False}},
        )
    except Exception:  # loud failure is fine
        return
    # y must keep its own column; x must not have been truncated to 1, 2, 3
    np.testing.assert_allclose(y, [2.5, 3.5, 4.5])
    assert not np.any(np.isfinite(x))


@pytest.mark.parametrize(
    "body",
    [
        'from PySide6.QtCore import QProcess\nQProcess.execute("/bin/true", [])\n',
        'import bz2\nbz2.BZ2File("/tmp/x", "w")\n',
        'import gzip\ngzip.GzipFile("/tmp/x", "wb")\n',
        'import lzma\nlzma.LZMAFile("/tmp/x", "w")\n',
        'from pathlib import Path\nPath("/tmp/x").mkdir()\n',
        'import matplotlib.pyplot as plt\nplt.savefig("/tmp/x")\n',
    ],
)
def test_validator_rejects_process_and_file_write_primitives(tmp_path, body):
    p = tmp_path / "m.py"
    p.write_text(body + "\ndef model(x, a):\n    return a * x\n")
    assert not validate_model(p).is_valid


def test_hdf5_unwritten_dataset_is_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "bomb.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("/data/x", shape=(10_000_000,), dtype="f8")
        f.create_dataset("/data/y", shape=(10_000_000,), dtype="f8")
    with pytest.raises(Exception, match="no stored data"):
        DataLoader().load(p, {"format": "hdf5"})


def test_unknown_top_level_yaml_key_is_warned(tmp_path, caplog):
    import logging

    from nlsq.cli.workflow_runner import WorkflowRunner

    p = tmp_path / "d.csv"
    p.write_text("x,y\n0,1\n1,2\n2,3\n3,4\n")
    cfg = {
        "data": {"input_file": str(p), "format": "csv"},
        "model": {"type": "builtin", "name": "linear"},
        "fiting": {"p0": [1.0, 0.0]},
    }
    with caplog.at_level(logging.WARNING):
        WorkflowRunner().run(cfg)
    assert any("fiting" in r.getMessage() for r in caplog.records)


def test_npz_complex_array_is_rejected(tmp_path):
    p = tmp_path / "c.npz"
    np.savez(p, x=np.arange(5.0), y=np.arange(5.0) * (1 + 2j))
    with pytest.raises(Exception, match="complex"):
        DataLoader().load(p, {"format": "npz"})
