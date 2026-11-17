"""Regression tests for core batch D review (2026-11-15).

Covers nlsq/core/sparse_jacobian.py and nlsq/core/workflow.py.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.sparse_jacobian import (
    SparseJacobianComputer,
    SparseOptimizer,
    detect_jacobian_sparsity,
    detect_sparsity_at_p0,
)
from nlsq.core.workflow import (
    ClusterDetector,
    ClusterInfo,
    MemoryBudget,
    MemoryBudgetSelector,
    OptimizationGoal,
    calculate_adaptive_tolerances,
    create_distributed_config,
)


def _linear(x, a, b):
    return a + b * x


def _dense_jac(n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(1.0, 2.0, size=(n, 2))


# --- W1: sample_size > 100 must cover the whole range ------------------------


def _step_model(x, a, b, c):
    return a * (x < 0.1) + (b + c) * (x >= 0.1)


def test_sample_size_above_100_samples_whole_range():
    x = np.linspace(0, 1, 10_000)
    p0 = np.array([1.0, 1.0, 1.0])
    ref, _ = detect_sparsity_at_p0(_step_model, p0, x, 10_000, sample_size=100)
    big, _ = detect_sparsity_at_p0(_step_model, p0, x, 10_000, sample_size=1000)
    # Both are evenly spaced over [0, 1]; the true value is ~0.367 (dense).
    assert abs(big - 0.367) < 0.03
    assert abs(big - ref) < 0.05


def test_detect_jacobian_sparsity_logs_truncation(caplog):
    x = np.linspace(0, 1, 500)
    with caplog.at_level(logging.INFO, logger="nlsq.core.sparse_jacobian"):
        detect_jacobian_sparsity(_linear, np.array([1.0, 2.0]), x, n_samples=50)
    assert any("first 50 of 500" in r.getMessage() for r in caplog.records)


def test_sample_size_validation():
    x = np.linspace(0, 1, 200)
    with pytest.raises(ValueError):
        detect_sparsity_at_p0(_linear, np.array([1.0, 2.0]), x, 200, sample_size=0)
    with pytest.raises(TypeError):
        detect_sparsity_at_p0(_linear, np.array([1.0, 2.0]), x, 200, sample_size=2.5)  # type: ignore[arg-type]


# --- W2: threshold validation ------------------------------------------------


@pytest.mark.parametrize("bad", [1.0, 2.0, -0.1, float("nan"), float("inf")])
def test_threshold_rejected(bad):
    with pytest.raises(ValueError, match="threshold"):
        SparseJacobianComputer(bad)
    with pytest.raises(ValueError, match="threshold"):
        detect_sparsity_at_p0(
            _linear, np.array([1.0, 2.0]), np.linspace(0, 1, 50), 50, threshold=bad
        )
    with pytest.raises(ValueError, match="threshold"):
        SparseOptimizer(sparsity_threshold=bad)


def test_threshold_zero_accepted():
    SparseJacobianComputer(0.0)


# --- W3: chunk_size validation -----------------------------------------------


@pytest.mark.parametrize("bad", [-5, 0])
def test_chunk_size_rejected(bad):
    J = _dense_jac()
    comp = SparseJacobianComputer()
    with pytest.raises(ValueError, match="chunk_size"):
        comp.compute_sparse_jacobian(
            lambda x, xd, yd, m, _: J[: len(yd)],
            np.ones(2),
            np.arange(50.0),
            np.zeros(50),
            chunk_size=bad,
        )


def test_chunk_size_non_int_rejected():
    comp = SparseJacobianComputer()
    with pytest.raises(TypeError, match="chunk_size"):
        comp.compute_sparse_jacobian(
            lambda *a: _dense_jac(),
            np.ones(2),
            np.arange(50.0),
            np.zeros(50),
            chunk_size=10.5,  # type: ignore[arg-type]
        )


# --- W4: MemoryBudget n_features / dtype_bytes -------------------------------


@pytest.mark.parametrize(
    "kw",
    [{"n_features": -1}, {"n_features": 0}, {"dtype_bytes": 0}, {"dtype_bytes": -8}],
)
def test_memory_budget_rejects_nonpositive(kw):
    with pytest.raises(ValueError):
        MemoryBudget.compute(n_points=10**9, n_params=3, memory_limit_gb=4.0, **kw)


# --- W5: list xdata in SparseOptimizer ---------------------------------------


def test_sparse_optimizer_list_xdata(caplog):
    n = 20_000

    def model(xy, *p):
        return sum(pi * xy[0] ** 0 for pi in p) + xy[1]

    X = [np.linspace(0, 1, n), np.linspace(0, 1, n)]
    opt = SparseOptimizer()
    # Dense model: detection runs, then the dense fallback (stubbed) is used.
    with (
        patch("nlsq.core.minpack.curve_fit", return_value="dense"),
        caplog.at_level(logging.INFO, logger="nlsq.core.sparse_jacobian"),
    ):
        out = opt.optimize_with_sparsity(model, np.ones(120), X, np.zeros(n))
    assert out == "dense"
    assert any("Detected sparsity" in r.getMessage() for r in caplog.records)


# --- W6: integer p0 ----------------------------------------------------------


def test_integer_p0_detection():
    x = np.linspace(0, 1, 200)
    s_int, _ = detect_jacobian_sparsity(_linear, np.array([1, 2]), x)
    s_float, _ = detect_jacobian_sparsity(_linear, np.array([1.0, 2.0]), x)
    assert s_int == pytest.approx(s_float)
    assert s_int < 0.1


def test_integer_x_finite_diff_jacobian():
    comp = SparseJacobianComputer()
    x = np.linspace(0, 1, 30)
    J = comp._finite_diff_jacobian(
        _linear, np.array([1, 2]), x, np.zeros(30), np.ones(30, bool)
    )
    assert np.abs(J).max() > 0.5


# --- W7: JAX (k, N) xdata ----------------------------------------------------


def test_jax_points_last_xdata_chunks_like_numpy():
    n = 50
    x1, x2 = np.linspace(0, 1, n), np.linspace(1, 2, n)
    seen: list[int] = []

    def jac(x, xd, yd, mask, _):
        assert xd.shape[0] == 2
        seen.append(xd.shape[1])
        return np.ones((len(yd), 2))

    comp = SparseJacobianComputer()
    out_np = comp.compute_sparse_jacobian(
        jac, np.ones(2), np.stack([x1, x2]), np.zeros(n), chunk_size=10
    )
    out_jx = comp.compute_sparse_jacobian(
        jac, np.ones(2), jnp.stack([x1, x2]), np.zeros(n), chunk_size=10
    )
    assert out_jx.nnz == out_np.nnz == 2 * n
    assert seen == [10] * 10


# --- S2: PBS nodefile parsing ------------------------------------------------


def test_pbs_nodefile_blank_and_whitespace_lines(tmp_path):
    nf = tmp_path / "nodes"
    nf.write_bytes(b"node01\nnode01 \n\nnode02\r\nnode02\n")
    with patch.dict(os.environ, {"PBS_NODEFILE": str(nf)}):
        det = ClusterDetector(default_gpus_per_node=2)
        with patch.object(det, "_detect_gpus_per_node", return_value=1):
            info = det.detect_pbs()
    assert info is not None
    assert info.node_list == ["node01", "node02"]
    assert info.node_count == 2


def test_pbs_blank_only_nodefile_is_none(tmp_path):
    nf = tmp_path / "nodes"
    nf.write_text("\n  \n")
    with patch.dict(os.environ, {"PBS_NODEFILE": str(nf)}):
        assert ClusterDetector().detect_pbs() is None


# --- S8: default GPUs substitution is not silent -----------------------------


def test_pbs_default_gpu_substitution_warns(tmp_path, caplog):
    nf = tmp_path / "nodes"
    nf.write_text("n1\nn2\n")
    with patch.dict(os.environ, {"PBS_NODEFILE": str(nf)}):
        det = ClusterDetector(default_gpus_per_node=8)
        with (
            patch.object(det, "_detect_gpus_per_node", return_value=0),
            caplog.at_level(logging.WARNING, logger="nlsq"),
        ):
            info = det.detect_pbs()
    assert info is not None and info.gpus_per_node == 8
    assert any("default_gpus_per_node" in r.getMessage() for r in caplog.records)


# --- S4: per-node GPU count uses local devices -------------------------------


def test_gpu_count_uses_local_devices():
    class Dev:
        platform = "gpu"

    with (
        patch("jax.devices", return_value=[Dev()] * 16),
        patch("jax.local_devices", return_value=[Dev()] * 4),
    ):
        assert ClusterDetector()._detect_gpus_per_node() == 4


# --- S3: no zero starts with multistart enabled ------------------------------


def test_distributed_config_zero_gpus_keeps_one_start():
    cfg = create_distributed_config(ClusterInfo.from_dict({}))
    assert cfg["enable_multistart"] and cfg["n_starts"] >= 1


# --- S5: string goal -----------------------------------------------------------


def test_adaptive_tolerances_string_goal_raises():
    with pytest.raises(TypeError, match="OptimizationGoal"):
        calculate_adaptive_tolerances(5_000_000, "quality")  # type: ignore[arg-type]
    assert (
        calculate_adaptive_tolerances(5_000_000, OptimizationGoal.QUALITY)["gtol"]
        == 1e-8
    )


# =============================================================================
# Round 2
# =============================================================================


# --- R2-W1: parameter-square allocations in the peak estimate -----------------


def test_peak_includes_parameter_square_terms():
    b = MemoryBudget.compute(100, 100_000, memory_limit_gb=1.0)
    p2_gib = 100_000**2 * 8 / 1024**3
    # covariance (p x p) + V (p x min(m, p)); here m << p so V is small
    assert p2_gib <= b.fixed_gb < 1.01 * p2_gib
    assert b.peak_gb >= b.fixed_gb
    bb = MemoryBudget.compute(100, 100_000, memory_limit_gb=1.0, bounded=True)
    assert bb.fixed_gb > b.fixed_gb


def test_selector_not_standard_when_covariance_cannot_fit(caplog):
    with caplog.at_level(logging.WARNING, logger="nlsq"):
        strategy, _ = MemoryBudgetSelector().select(100, 100_000, memory_limit_gb=1.0)
    assert strategy != "standard"
    assert any(
        "Parameter-dependent allocations" in r.getMessage() for r in caplog.records
    )


def test_small_problem_budget_unchanged_by_fixed_terms():
    b = MemoryBudget.compute(1000, 3, memory_limit_gb=8.0)
    assert b.fixed_gb < 1e-6
    assert MemoryBudgetSelector().select(1000, 3, memory_limit_gb=8.0)[0] == "standard"


# --- R2-W2: list xdata forms ---------------------------------------------------

_FLAT = [1.0, 2.0, 3.0]
_COORD = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


@pytest.mark.parametrize(
    "xdata", [_FLAT, tuple(_FLAT), np.array(_FLAT), jnp.array(_FLAT)]
)
def test_flat_xdata_forms_detect(xdata):
    s, _ = detect_jacobian_sparsity(
        lambda x, a: a * np.asarray(x), np.array([1.0]), xdata
    )
    assert s == 0.0


@pytest.mark.parametrize(
    "xdata",
    [
        _COORD,
        tuple(tuple(c) for c in _COORD),
        [np.array(c) for c in _COORD],
        np.array(_COORD),
        jnp.array(_COORD),
    ],
)
def test_coordinate_xdata_forms_at_p0(xdata):
    s, _ = detect_sparsity_at_p0(
        lambda x, a: a * np.asarray(x[0]), np.array([1.0]), xdata, 3, sample_size=2
    )
    assert s == 0.0


def test_flat_list_in_optimizer_sampling_and_chunks(caplog):
    n = 20_000
    X = list(np.linspace(0, 1, n))
    opt = SparseOptimizer()
    with (
        patch("nlsq.core.minpack.curve_fit", return_value="dense"),
        caplog.at_level(logging.INFO, logger="nlsq.core.sparse_jacobian"),
    ):
        out = opt.optimize_with_sparsity(
            lambda x, *p: sum(p) * np.asarray(x), np.ones(120), X, np.zeros(n)
        )
    assert out == "dense"
    J = SparseJacobianComputer().compute_sparse_jacobian(
        lambda *a: np.ones((len(a[2]), 1)),
        np.ones(1),
        [1.0, 2.0, 3.0, 4.0],
        np.zeros(4),
        chunk_size=3,
    )
    assert J.nnz == 4


# --- R2-W3: non-finite values are rejected ------------------------------------


def test_nan_model_rejected():
    with pytest.raises(ValueError, match="not finite"):
        detect_jacobian_sparsity(
            lambda x, a: np.full(len(x), np.nan), np.array([1.0]), np.arange(3.0)
        )


def test_nan_after_probe_rejected():
    # float(a) is untraceable, so the finite-difference probe path is used.
    def f(x, a):
        return np.where(float(a) > 1.0, np.nan, 1.0) * np.ones(len(x))

    with pytest.raises(ValueError, match="parameter 0"):
        detect_jacobian_sparsity(f, np.array([1.0]), np.arange(3.0))


# --- R2-W4: cancellation does not erase a real column -------------------------


def test_cancellation_column_not_reported_zero():
    s, _ = detect_jacobian_sparsity(
        lambda x, a: 1e8 + a * np.ones_like(x), np.array([1.0]), np.arange(3.0)
    )
    assert s == 0.0


def test_genuine_zero_column_still_sparse():
    s, _ = detect_jacobian_sparsity(
        lambda x, a, b: 1e8 + a * np.ones_like(x), np.array([1.0, 2.0]), np.arange(3.0)
    )
    assert s == pytest.approx(0.5)


# --- R2-W5: empty sampling ----------------------------------------------------


def test_detect_jacobian_sparsity_empty_cases_raise_cleanly(recwarn):
    with pytest.raises(ValueError, match="n_samples"):
        detect_jacobian_sparsity(
            lambda x, a: a * x, np.array([1.0]), np.arange(3.0), n_samples=0
        )
    with pytest.raises(ValueError, match="empty"):
        detect_jacobian_sparsity(lambda x, a: a * x, np.array([1.0]), np.array([]))
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_pattern_empty_sample_returns_empty_without_warning(recwarn):
    pattern, sparsity = SparseJacobianComputer().detect_sparsity_pattern(
        lambda x, a: a * x, np.array([1.0]), np.array([]), n_samples=0
    )
    assert pattern.size == 0 and sparsity == 0.0
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# --- R2-W6: NaN / non-integral counts -----------------------------------------


@pytest.mark.parametrize("name", ["n_points", "n_params", "n_features", "dtype_bytes"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2.5])
def test_budget_rejects_nonfinite_or_fractional(name, bad):
    kw = {"n_points": 10, "n_params": 2, "memory_limit_gb": 1.0, name: bad}
    with pytest.raises(ValueError, match=name):
        MemoryBudget.compute(**kw)


@pytest.mark.parametrize("name", ["n_points", "n_params", "n_features", "dtype_bytes"])
def test_budget_rejects_bool_and_str(name):
    for bad in (True, "3"):
        kw = {"n_points": 10, "n_params": 2, "memory_limit_gb": 1.0, name: bad}
        with pytest.raises(TypeError, match=name):
            MemoryBudget.compute(**kw)


def test_budget_accepts_integral_float_and_numpy_int():
    b = MemoryBudget.compute(1e6, np.int64(3), memory_limit_gb=8.0)
    assert np.isfinite(b.peak_gb)


# --- R2-W7: TPU is not a GPU --------------------------------------------------


def _dev(platform):
    class D:
        pass

    d = D()
    d.platform = platform
    return d


@pytest.mark.parametrize(
    ("platforms", "expected"),
    [
        (["tpu"], 0),
        (["cpu", "gpu", "tpu"], 1),
        (["cuda", "rocm", "gpu", "cpu"], 3),
        (["cpu"], 0),
    ],
)
def test_gpu_count_excludes_tpu(platforms, expected):
    with patch("jax.local_devices", return_value=[_dev(p) for p in platforms]):
        assert ClusterDetector()._detect_gpus_per_node() == expected


def test_tpu_only_node_has_no_local_gpu_config():
    with patch("jax.local_devices", return_value=[_dev("tpu")]):
        assert ClusterDetector().detect_local_gpus() is None


# --- R2-W8: callable Jacobian honors data_mask --------------------------------


def test_callable_jacobian_applies_mask_like_fd_branch():
    comp = SparseJacobianComputer()
    mask = np.array([True, False, True])
    y = np.zeros(3)
    x = np.arange(3.0)
    J_call = comp.compute_sparse_jacobian(
        lambda *a: np.ones((3, 1)), np.ones(1), x, y, data_mask=mask
    )
    J_fd = comp.compute_sparse_jacobian(
        None, np.ones(1), x, y, data_mask=mask, func=lambda x, a: a * np.ones_like(x)
    )
    assert J_call.toarray().ravel().tolist() == [1.0, 0.0, 1.0]
    np.testing.assert_allclose(J_fd.toarray().ravel(), [1.0, 0.0, 1.0], atol=1e-6)


def test_callable_jacobian_wrong_row_count_rejected():
    with pytest.raises(ValueError, match="shape"):
        SparseJacobianComputer().compute_sparse_jacobian(
            lambda *a: np.ones((2, 1)), np.ones(1), np.arange(3.0), np.zeros(3)
        )


# =============================================================================
# Round 3
# =============================================================================


# --- R3-W1: identical probe outputs are not a proven zero ----------------------


def test_offset_swamped_column_uses_autodiff():
    s, info = detect_jacobian_sparsity(
        lambda x, a: 1e8 + 1e-10 * a * x, np.array([0.0]), jnp.array([1.0, 2.0, 3.0])
    )
    assert s == 0.0 and info["nnz"] == 3


def test_traceable_model_ignoring_parameter_is_sparse():
    s, _ = detect_jacobian_sparsity(
        lambda x, a, b: 1e8 + 1e-10 * a * x,
        np.array([0.0, 1.0]),
        jnp.array([1.0, 2.0, 3.0]),
    )
    assert s == pytest.approx(0.5)


def test_numpy_only_model_with_unused_parameter_is_dense_when_untraceable():
    def f(x, a, b):
        return np.exp(np.asarray(1e8 * 0.0 + float(a) * 0.0)) * np.ones_like(x)

    # float(tracer) fails under jvp, so autodiff is unavailable: both columns dense.
    s, _ = detect_jacobian_sparsity(f, np.array([1.0, 1.0]), np.arange(3.0))
    assert s == 0.0


# --- R3-W2: callback shape must be exactly (points, n_params) -------------------


@pytest.mark.parametrize("shape", [(3, 1), (3,), (3, 2, 1), (2, 2), (3, 3)], ids=str)
def test_callable_jacobian_wrong_shape_rejected(shape):
    with pytest.raises(ValueError, match=r"shape .* expected \(3, 2\)"):
        SparseJacobianComputer().compute_sparse_jacobian(
            lambda *a: np.ones(shape), np.ones(2), np.arange(3.0), np.zeros(3)
        )


def test_callable_jacobian_correct_shape_and_masked_chunks():
    mask = np.array([True, False, True, True])
    J = SparseJacobianComputer().compute_sparse_jacobian(
        lambda p, xd, yd, m, _: np.ones((len(yd), 2)),
        np.ones(2),
        np.arange(4.0),
        np.zeros(4),
        data_mask=mask,
        chunk_size=3,
    )
    np.testing.assert_array_equal(J.toarray(), np.outer(mask, [1.0, 1.0]))


# --- R3-W3: peak counts J, scaled J and U (same accounting as MemoryManager) ----


def test_peak_counts_three_jacobian_sized_buffers():
    b = MemoryBudget.compute(1_000_000, 50, memory_limit_gb=1.0)
    gib = 1024**3
    assert b.peak_gb >= 3 * 1_000_000 * 50 * 8 / gib  # J + scaled J + U
    assert MemoryBudgetSelector().select(1_000_000, 50, memory_limit_gb=1.0)[0] != (
        "standard"
    )


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize(("m", "n"), [(10_000, 20), (50, 200)])
def test_peak_matches_memory_manager_prediction(bounded, m, n):
    from nlsq.caching.memory_manager import MemoryManager

    b = MemoryBudget.compute(m, n, memory_limit_gb=8.0, bounded=bounded)
    predicted = MemoryManager(safety_factor=1.0).predict_memory_requirement(
        m, n, "trf", bounded=bounded
    )
    covariance = n * n * 8
    ours = (b.peak_gb - 0.1) * 1024**3 - covariance
    assert ours == pytest.approx(predicted, rel=1e-9)


# --- R3-W5: chunk / batch sizes respect the fixed p x p allocations -------------


def _total_gib(budget, rows):
    return budget.fixed_gb + rows * budget.bytes_per_point / 1024**3


def test_streaming_chunk_size_accounts_for_fixed_allocations():
    strategy, cfg = MemoryBudgetSelector().select(
        100_000_000, 6000, memory_limit_gb=1.0
    )
    b = MemoryBudget.compute(100_000_000, 6000, memory_limit_gb=1.0)
    assert strategy == "streaming"
    assert _total_gib(b, cfg.chunk_size) <= b.threshold_gb


def test_chunked_size_accounts_for_fixed_allocations():
    b = MemoryBudget.compute(10_000_000, 3000, memory_limit_gb=1.0)
    rows = MemoryBudgetSelector()._compute_chunk_size(b, 3000)
    assert _total_gib(b, rows) <= b.threshold_gb


def test_minimum_chunk_warns_when_fixed_blocks_leave_no_room(caplog):
    b = MemoryBudget.compute(100_000_000, 20_000, memory_limit_gb=1.0)
    with caplog.at_level(logging.WARNING, logger="nlsq.core.workflow"):
        rows = MemoryBudgetSelector()._compute_streaming_batch_size(b, 20_000)
    assert rows == 1000
    assert any("minimum" in r.getMessage() for r in caplog.records)


# =============================================================================
# Round 4 (Claude round 2): exact derivative first
# =============================================================================


def test_enlarged_step_secant_cannot_hide_a_dense_column():
    x = jnp.array([2 * np.pi] * 50 + [np.pi / 2] * 50)
    s, info = detect_jacobian_sparsity(
        lambda x, a: 1e14 + 3 * jnp.sin(a * x), np.array([0.0]), x
    )
    assert s == 0.0 and info["nnz"] == 100


def test_untraceable_unresolved_column_is_dense_not_sparse():
    x = np.array([2 * np.pi] * 50 + [np.pi / 2] * 50)
    s, _ = detect_jacobian_sparsity(
        lambda x, a: 1e14 + 3 * np.sin(float(a) * x), np.array([0.0]), x
    )
    assert s == 0.0


def test_pattern_matches_jacfwd_on_random_models():
    import jax

    rng = np.random.default_rng(7)
    for _ in range(15):
        n_data, n_par = int(rng.integers(20, 60)), int(rng.integers(2, 6))
        x = jnp.asarray(rng.uniform(-3, 3, n_data))
        offset = float(10.0 ** rng.integers(0, 15))
        coef = jnp.asarray(rng.normal(size=n_par))
        active = rng.random(n_par) < 0.6  # inactive parameters are ignored
        mask = jnp.asarray(active, dtype=float)
        x0 = rng.normal(size=n_par)

        def f(xd, *p, coef=coef, mask=mask, offset=offset):
            p = jnp.stack(p)
            return offset + jnp.sin(xd[:, None] * (p * mask) * coef).sum(axis=1) * (
                xd > 0
            )

        J = np.abs(
            np.asarray(jax.jacfwd(lambda p, x=x, f=f: f(x, *p))(jnp.asarray(x0)))
        )
        expected = 0.01 * np.max(J, axis=0, initial=0.0) < J
        pattern, _ = SparseJacobianComputer().detect_sparsity_pattern(f, x0, x, n_data)
        np.testing.assert_array_equal(pattern, expected)
