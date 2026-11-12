"""Review 2026-11-12 f9: telemetry dedup, Hybrid(verbose) honoured, checkpoint DoS."""

import contextlib

import h5py
import numpy as np
import pytest

from nlsq.streaming.telemetry import DefenseLayerTelemetry


def test_line_search_failed_steps_counted_once_out_of_order():
    t = DefenseLayerTelemetry()
    t.record_warmup_start()
    for i in range(3):
        t.record_lbfgs_iteration()
        t.record_lbfgs_line_search_failure(i)
        if i > 0:  # deferred loss-increase check reports the previous step
            t.record_lbfgs_line_search_failure(i - 1)
    assert t.lbfgs_line_search_failed_steps == 3
    assert t.get_trigger_rates()["lbfgs_line_search_failure_rate"] <= 100.0


@pytest.mark.slow
def test_hybrid_verbose_without_config_is_honoured(monkeypatch):
    from nlsq import curve_fit_large
    from nlsq.streaming import strategy
    from nlsq.streaming.specs import Hybrid

    seen = {}
    real = strategy.build_large_strategy

    def spy(kind, config):
        seen["verbose"] = config.verbose
        return real(kind, config)

    monkeypatch.setattr(strategy, "build_large_strategy", spy)
    x = np.linspace(0, 1, 2000)
    with contextlib.suppress(Exception):  # only the engine config matters
        curve_fit_large(
            lambda x, a, b: a * x + b,
            x,
            2.0 * x + 1.0,
            p0=[1.0, 0.0],
            strategy=Hybrid(verbose=2),
        )
    assert seen["verbose"] == 2
    seen.clear()
    with contextlib.suppress(Exception):  # only the engine config matters
        curve_fit_large(
            lambda x, a, b: a * x + b,
            x,
            2.0 * x + 1.0,
            p0=[1.0, 0.0],
            strategy=Hybrid(),
        )
    assert seen["verbose"] == 1  # default config untouched


def _crafted(path, group, name, shape, dtype="i8"):
    with h5py.File(path, "w") as f:
        f.attrs["version"] = "3.0"
        f.create_group(group).create_dataset(
            name, shape=shape, dtype=dtype, chunks=(1024,)
        )


def test_hybrid_checkpoint_rejects_huge_scalar_dataset(tmp_path):
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.phases.checkpoint import CheckpointManager

    p = tmp_path / "bad.h5"
    _crafted(p, "phase_state", "current_phase", (2**31,))
    with pytest.raises(ValueError, match="current_phase"):
        CheckpointManager(HybridStreamingConfig()).load(str(p))


def test_hybrid_checkpoint_rejects_huge_bytes_dataset(tmp_path):
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.phases.checkpoint import CheckpointManager

    p = tmp_path / "bad2.h5"
    with h5py.File(p, "w") as f:
        f.attrs["version"] = "3.0"
        g = f.create_group("phase_state")
        g.create_dataset("current_phase", data=1)
    with h5py.File(p, "a") as f:
        f["phase_state"].create_dataset("best_cost_global", data=1.0)
        f["phase_state"].create_dataset(
            "phase_history", shape=(2**31,), dtype="V1", chunks=(1024,)
        )
    with pytest.raises(ValueError, match="phase_history"):
        CheckpointManager(HybridStreamingConfig()).load(str(p))


def test_scalar_and_bytes_readers_reject_wrong_shape(tmp_path):
    """Shared by the CMA-ES checkpoint loader (generation_counter, popsize, fitness_history)."""
    from nlsq.streaming.phases.checkpoint import (
        read_checkpoint_bytes,
        read_checkpoint_scalar,
    )

    p = tmp_path / "cma.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("n", shape=(2**31,), dtype="i8", chunks=(1024,))
        f.create_dataset("b", shape=(2**31,), dtype="V1", chunks=(1024,))
        f.create_dataset("ok", data=np.int64(3))
        f.create_dataset("okb", data=np.void(b"[1]"))
    with h5py.File(p, "r") as f:
        with pytest.raises(ValueError, match="generation_counter"):
            read_checkpoint_scalar(f["n"], "generation_counter")
        with pytest.raises(ValueError, match="fitness_history"):
            read_checkpoint_bytes(f["b"], "fitness_history")
        assert read_checkpoint_scalar(f["ok"], "ok") == 3
        assert read_checkpoint_bytes(f["okb"], "okb") == b"[1]"
