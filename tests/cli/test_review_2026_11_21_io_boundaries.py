"""CLI input boundaries: bare .npy named .npz, NUL in results_file."""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import DataLoadError


def test_bare_npy_named_npz_is_rejected_without_loading_it(tmp_path, monkeypatch):
    path = tmp_path / "bare.npz"
    np.save(tmp_path / "bare.npy", np.zeros(5000))
    (tmp_path / "bare.npy").rename(path)
    monkeypatch.setenv("NLSQ_MAX_LOAD_ELEMENTS", "1000")
    with pytest.raises(DataLoadError, match=r"Not a valid \.npz archive"):
        DataLoader().load(path, {"format": "npz"})


def test_valid_npz_still_loads(tmp_path):
    path = tmp_path / "ok.npz"
    np.savez(path, x=np.arange(4.0), y=np.arange(4.0) * 2)
    x, y, sigma = DataLoader().load(path, {"format": "npz"})
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert y.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert sigma is None


def test_nul_in_results_file_is_a_config_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.csv").write_text("x,y\n1,2\n2,4\n3,6.1\n4,8\n")
    (tmp_path / "w.yaml").write_text(
        "data:\n  input_file: d.csv\n  format: csv\n"
        "model:\n  type: builtin\n  name: linear\n"
        'export:\n  results_file: "r\\x00.json"\n'
    )
    from nlsq.cli.main import main

    assert main(["fit", "w.yaml"]) != 0
    assert "NUL" in capsys.readouterr().err
