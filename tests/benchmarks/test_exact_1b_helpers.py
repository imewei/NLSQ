"""Unit tests for the Stage 3 1B-point run helpers (small scale only)."""

import argparse
import json

import h5py
import numpy as np
import pytest

import benchmarks.studies.exact_1b_run as run_mod
from benchmarks.compare.timing import residual_cost
from benchmarks.studies.exact_1b_run import (
    PROBLEM,
    THETA_TRUE,
    TailOffsetY,
    _count_ok,
    calibrate_sentinel,
    hessian_at,
    model_jnp,
    model_np,
    noisy_gate,
    rel_size,
    streaming_cost,
    write_dataset,
)
from nlsq.streaming.exact_lm import exact_lm_fit


def test_streaming_cost_matches_residual_cost_small():
    # both use the solver's convention F = 0.5 * sum(r**2), so they are compared directly
    x, y = PROBLEM.data(100_000, seed=3)
    theta = THETA_TRUE * (1 + 1e-3)
    want = residual_cost(PROBLEM, x, y, theta)
    got = streaming_cost((x, y), model_np, theta, 7919)  # tail chunk is short
    assert abs(got - want) <= 1e-12 * abs(want)

    sigma = np.linspace(0.5, 2.0, x.size)
    r = (y - model_np(x, *theta)) / sigma
    wanted = 0.5 * float(np.sum(r**2))
    gotw = streaming_cost((x, y, sigma), model_np, theta, 7919)
    assert abs(gotw - wanted) <= 1e-12 * wanted


def test_generator_is_independent_of_chunk_write_order(tmp_path):
    n, c = 10_000, 1024  # 10 chunks, short tail
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    assert write_dataset(a, n, 7, 0.05, c) == "written"
    assert write_dataset(b, n, 7, 0.05, c, order=list(range(9, -1, -1))) == "written"
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        np.testing.assert_array_equal(fa["x"][:], fb["x"][:])
        np.testing.assert_array_equal(fa["y"][:], fb["y"][:])
        np.testing.assert_array_equal(fa["x"][:], np.linspace(0.0, 4.0, n))
        assert np.std(fa["y"][:] - model_np(fa["x"][:], *THETA_TRUE)) > 0.04
    assert write_dataset(a, n, 7, 0.05, c) == "exists"
    assert not (tmp_path / "a.h5.tmp").exists()


def test_fingerprint_mismatch_refuses_overwrite(tmp_path):
    p = tmp_path / "d.h5"
    write_dataset(p, 5000, 0, 0.0, 1000)
    before = p.read_bytes()
    for kw in ({"seed": 1}, {"noise": 0.05}, {"chunk_rows": 999}, {"n": 5001}):
        args = {"n": 5000, "seed": 0, "noise": 0.0, "chunk_rows": 1000} | kw
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            write_dataset(p, **args)
    assert p.read_bytes() == before
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not hdf5")
    with pytest.raises(FileExistsError):
        write_dataset(junk, 5000, 0, 0.0, 1000)


def test_calibrate_sentinel_gives_stated_displacement():
    n, chunk, target = 50_000, 8192, 1e-3
    x = np.linspace(0.0, 4.0, n)
    y = model_np(x, *THETA_TRUE)
    H, n_seen = hessian_at(THETA_TRUE, x, y, chunk)
    assert n_seen == n
    delta, pred = calibrate_sentinel(THETA_TRUE, x[-1000:], H, target)
    assert rel_size(pred, THETA_TRUE) == pytest.approx(target, rel=1e-12)

    free_y, sent_y = TailOffsetY(y), TailOffsetY(y, delta)
    free = exact_lm_fit(model_jnp, x, free_y, PROBLEM.p0, chunk_size=chunk)
    sent = exact_lm_fit(model_jnp, x, sent_y, PROBLEM.p0, chunk_size=chunk)
    assert free["success"] and sent["success"]
    moved = sent["x"] - free["x"]
    assert 0.8 <= rel_size(moved, free["x"]) / target <= 1.2
    assert float(moved @ pred / (pred @ pred)) == pytest.approx(1.0, abs=0.2)
    # every pass read every y row
    assert free_y.rows_read == n * free["nfev"]
    assert sent_y.rows_read == n * sent["nfev"]
    np.testing.assert_array_equal(sent_y[0 : n - 1000], y[: n - 1000])
    np.testing.assert_array_equal(sent_y[n - 1500 : n][500:], y[-1000:] + delta)


def test_noisy_gate_fails_without_a_covariance():
    p = len(THETA_TRUE)
    assert (
        noisy_gate(THETA_TRUE, np.full((p, p), np.inf))[0] is False
    )  # z = 0 is not a pass
    assert noisy_gate(THETA_TRUE, np.full((p, p), np.nan))[0] is False
    assert noisy_gate(THETA_TRUE, np.zeros((p, p)))[0] is False  # se = 0
    good = np.diag(np.full(p, 1e-4))
    assert noisy_gate(THETA_TRUE + 4e-2, good)[0] is True  # z = 4
    assert noisy_gate(THETA_TRUE + 6e-2, good)[0] is False  # z = 6


def test_count_ok_truth_table():
    n = 100
    ok = {"pass_n_seen": [n, n, n], "nfev": 3, "y_rows_read": 3 * n}
    assert _count_ok(ok, n)
    assert not _count_ok({**ok, "pass_n_seen": [n, n - 1, n]}, n)  # a short pass
    assert not _count_ok({**ok, "y_rows_read": 3 * n - 1}, n)  # y rows missed
    assert not _count_ok({**ok, "pass_n_seen": [n, n]}, n)  # pass log != nfev
    assert not _count_ok({**ok, "nfev": 4}, n)


def _args(tmp_path):
    return argparse.Namespace(
        n=20_000,
        chunk_rows=4096,
        seed=0,
        noise=0.05,
        data_dir=tmp_path,
        out=str(tmp_path / "out.json"),
    )


def _verdicts(tmp_path):
    return {
        g["gate"]: g["pass"]
        for g in json.loads((tmp_path / "out.json").read_text())["gates"]
    }


def test_run_small_scale_all_gates_pass(tmp_path):
    assert run_mod.run(_args(tmp_path)) == 0
    verdicts = _verdicts(tmp_path)
    assert len(verdicts) == 8 and all(verdicts.values()), verdicts


def test_run_fails_when_the_sentinel_is_not_felt(tmp_path, monkeypatch):
    real = run_mod.TailOffsetY
    # drops the tail offset: the fit cannot see the sentinel, so the gate must say so
    monkeypatch.setattr(run_mod, "TailOffsetY", lambda y, offset=0.0: real(y, 0.0))
    assert run_mod.run(_args(tmp_path)) == 1
    verdicts = _verdicts(tmp_path)
    assert verdicts["3iii sentinel"] is False
    assert verdicts["1 noiseless recovery"] is True
