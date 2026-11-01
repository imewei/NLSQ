"""Stage 3 out-of-memory sources for exact_lm: array-likes read chunk by chunk.

memmap, ``h5py.Dataset``, Zarr and any object with ``shape``/``len``/slicing are streamed
from the source on every pass; nothing reads or places the whole source. The basic
ndarray view test (``test_cpu_chunks_are_views``) lives in test_exact_lm.py.
"""

import tracemalloc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.streaming.exact_lm as lm
from benchmarks.compare.problems import PROBLEMS
from nlsq import curve_fit_large

EXP = PROBLEMS["exp_decay"]
N, CHUNK = 3000, 1024  # 3 chunks, padded tail


def exp_model(x, a, b, c):
    return EXP.model(jnp, x, a, b, c)


@pytest.fixture(scope="module")
def data():
    return EXP.data(N)


class Counting:
    """Array-like (``shape``, ``len``, slicing) that records every read as ``(start, stop)``.

    Converting it whole (``np.asarray``) fails the test.
    """

    def __init__(self, a, log):
        self._a, self.shape, self.log = a, a.shape, log

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, k):
        assert isinstance(k, slice), f"non-slice read {k!r}"
        start, stop, _ = k.indices(self.shape[0])
        self.log.append((start, stop))
        return self._a[k]

    def __array__(self, *a, **k):
        raise AssertionError("source was materialised whole")


@pytest.fixture
def passes(monkeypatch):
    """Spy on run_pass: records one entry per full data pass."""
    calls = []
    real = lm.run_pass

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(lm, "run_pass", spy)
    return calls


def _sizes(log):
    return [b - a for a, b in log]


# ---- finite check covers every block; probe on the first block only ------------------


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("which", ["x", "y"])
def test_invalid_final_block_raises(data, passes, lazy, which):
    x, y = (a.copy() for a in data)
    (x if which == "x" else y)[-1] = np.nan  # only the (padded) last block is bad
    if lazy:
        x, y = Counting(x, []), Counting(y, [])
    with pytest.raises(ValueError, match=r"^x and y must be finite"):
        lm.exact_lm_fit(exp_model, x, y, EXP.p0, chunk_size=CHUNK)
    assert len(passes) == 1  # raised after pass 1, before any LM step


def test_probe_first_block_only_finite_every_block(data, passes, monkeypatch):
    probes = []
    real = lm.check_model_contract

    def spy(model, theta, x_chunk):
        probes.append(len(x_chunk))
        return real(model, theta, x_chunk)

    monkeypatch.setattr(lm, "check_model_contract", spy)
    out = lm.exact_lm_fit(exp_model, *data, EXP.p0, chunk_size=CHUNK)
    assert out["nfev"] > 1
    assert probes == [CHUNK]  # one probe, on one block
    assert passes[0]["check_finite"] is True  # pass 1 checks every block...
    assert not any(k["check_finite"] for k in passes[1:])  # ...later passes do not


# ---- lazy sources: chunk-sized reads, one pass = one read of the data ----------------


def _check_reads(log, n, nfev, chunk, extra):
    sizes = _sizes(log)
    assert max(sizes) <= chunk
    assert n * nfev <= sum(sizes) <= n * nfev + extra


@pytest.mark.parametrize("chunk", [CHUNK, N])
def test_reads_are_chunk_sized(data, passes, monkeypatch, chunk):
    xl, yl = [], []
    before = []
    real = lm.run_pass

    def first(*a, **k):  # reads that happen before the first pass begins
        if not before:
            before.append((list(xl), list(yl)))
        return real(*a, **k)

    monkeypatch.setattr(lm, "run_pass", first)
    out = lm.exact_lm_fit(
        exp_model,
        Counting(data[0], xl),
        Counting(data[1], yl),
        EXP.p0,
        chunk_size=chunk,
    )
    ref = lm.exact_lm_fit(exp_model, *data, EXP.p0, chunk_size=chunk)
    np.testing.assert_array_equal(out["x"], ref["x"])
    assert out["cost"] == ref["cost"]
    nfev = out["nfev"]
    # validation reads nothing; the model probe reads one chunk of x, once
    assert before == [([(0, chunk)], [])]
    _check_reads(xl, N, nfev, chunk, extra=chunk)
    _check_reads(yl, N, nfev, chunk, extra=0)
    one_pass = [(s, min(s + chunk, N)) for s in range(0, N, chunk)]
    assert yl == one_pass * nfev  # sequential, in order, no shuffle, every point
    assert xl == one_pass[:1] + one_pass * nfev


def test_lazy_sigma_read_chunkwise(data):
    x, y = data
    s = 0.01 * (1.0 + np.arange(N) % 7)
    logs = [], [], []
    src = [Counting(a, log) for a, log in zip((x, y, s), logs, strict=True)]
    out = lm.exact_lm_fit(
        exp_model, src[0], src[1], EXP.p0, sigma=src[2], chunk_size=CHUNK
    )
    ref = lm.exact_lm_fit(exp_model, x, y, EXP.p0, sigma=s, chunk_size=CHUNK)
    np.testing.assert_array_equal(out["x"], ref["x"])
    np.testing.assert_array_equal(out["pcov"], ref["pcov"])
    _check_reads(logs[2], N, out["nfev"], CHUNK, extra=0)


def test_lazy_source_through_curve_fit_large(data):
    xl, yl = [], []
    popt, _ = curve_fit_large(
        exp_model,
        Counting(data[0], xl),
        Counting(data[1], yl),
        p0=EXP.p0,
        method="exact_streaming",
        chunk_size=CHUNK,
    )
    ref = lm.exact_lm_fit(exp_model, *data, EXP.p0, chunk_size=CHUNK)
    np.testing.assert_array_equal(popt, ref["x"])
    assert max(_sizes(xl + yl)) <= CHUNK


def test_multi_feature_xdata_rejected():
    y = np.ones(10)
    pat = r"multi-feature \(2-D\) xdata is not supported by exact_streaming"
    with pytest.raises(ValueError, match=pat):
        curve_fit_large(
            exp_model, np.ones((10, 2)), y, p0=EXP.p0, method="exact_streaming"
        )
    with pytest.raises(ValueError, match=pat):
        lm.exact_lm_fit(exp_model, np.ones((10, 2)), np.ones((10, 2)), EXP.p0)


class ZarrLike:
    """zarr v3 ``Array`` semantics: ``shape``/``dtype``/slicing, no ``__len__``, no
    ``__array__``; slices clamp at the end and return a fresh ``np.ndarray``."""

    def __init__(self, a, log):
        self._a, self.shape, self.dtype, self.log = a, a.shape, a.dtype, log

    def __getitem__(self, k):
        assert isinstance(k, slice), f"non-slice read {k!r}"
        start, stop, _ = k.indices(self.shape[0])
        self.log.append((start, stop))
        return np.array(self._a[k])


def test_zarr_like_source_without_len(data):
    x, y = data
    s = 0.01 * (1.0 + np.arange(N) % 7)
    logs = [], [], []
    src = [ZarrLike(a, log) for a, log in zip((x, y, s), logs, strict=True)]
    out = lm.exact_lm_fit(
        exp_model, src[0], src[1], EXP.p0, sigma=src[2], chunk_size=CHUNK
    )
    ref = lm.exact_lm_fit(exp_model, x, y, EXP.p0, sigma=s, chunk_size=CHUNK)
    np.testing.assert_array_equal(out["x"], ref["x"])
    assert out["cost"] == ref["cost"] and out["n_points"] == N
    for log in logs:
        assert max(_sizes(log)) <= CHUNK
    _check_reads(logs[1], N, out["nfev"], CHUNK, extra=0)
    popt, _ = curve_fit_large(
        exp_model,
        ZarrLike(x, []),
        ZarrLike(y, []),
        p0=EXP.p0,
        sigma=ZarrLike(s, []),
        method="exact_streaming",
        chunk_size=CHUNK,
    )
    np.testing.assert_array_equal(popt, ref["x"])


class Strict(Counting):
    """Array-like that raises ``IndexError`` on a slice running past the end."""

    def __getitem__(self, k):
        if k.stop is not None and k.stop > self.shape[0]:
            raise IndexError(f"slice stop {k.stop} > length {self.shape[0]}")
        return super().__getitem__(k)


def test_strict_array_like_last_block_clamped(data):
    x, y = data
    s = 0.01 * (1.0 + np.arange(N) % 7)
    src = [Strict(a, []) for a in (x, y, s)]
    out = lm.exact_lm_fit(
        exp_model, src[0], src[1], EXP.p0, sigma=src[2], chunk_size=CHUNK
    )
    ref = lm.exact_lm_fit(exp_model, x, y, EXP.p0, sigma=s, chunk_size=CHUNK)
    np.testing.assert_array_equal(out["x"], ref["x"])
    assert out["cost"] == ref["cost"]


def test_place_zarr_like_not_placed(gpu):
    z = ZarrLike(np.ones(1000), [])
    assert lm._place(z, z, None)[0] is z


def test_memmap_chunks_are_views(data, tmp_path):
    """A float64 memmap chunk is a view of the mapping: no per-pass host copy."""
    s = np.full(N, 0.1)
    mm = [_memmap(tmp_path / f"{i}.bin", a) for i, a in enumerate((*data, s))]
    xc, yc, sc, mask = next(lm.iter_blocks(*mm, CHUNK))
    assert all(np.shares_memory(b, a) for b, a in zip((xc, yc, sc), mm, strict=True))
    assert bool(mask.all())


# ---- JAX arrays stay on device --------------------------------------------------------


def test_jax_array_stays_resident(data):
    """Under ``transfer_guard_device_to_host("disallow")`` an implicit host round trip
    raises on an accelerator. Explicit ``jax.device_get`` (the per-pass p- and p^2-sized
    statistics) is allowed by that guard and is the only transfer the fit makes. On the CPU
    backend the guard does not fire (CPU arrays are host memory), so there only the
    ``jax.Array`` type checks are meaningful.
    """
    x, y = (jnp.asarray(a) for a in data)
    s = jnp.full(N, 0.1)
    with jax.transfer_guard_device_to_host("disallow"):
        for xc, yc, sc, mask in lm.iter_blocks(x, y, s, CHUNK):
            assert all(isinstance(b, jax.Array) for b in (xc, yc, sc, mask))
        out = lm.exact_lm_fit(exp_model, x, y, EXP.p0, chunk_size=CHUNK)
    assert out["success"]


# ---- device placement: never whole-place a lazy source, honour memory_limit_gb -------


class Lazy:
    shape = (1000,)

    def __len__(self):
        return 1000

    def __getitem__(self, k):
        return np.ones(1000)[k]


@pytest.fixture
def gpu(monkeypatch):
    """Pretend to be an accelerator with 1 TB free (the placement itself runs on CPU)."""
    monkeypatch.setattr(lm, "_free_device_bytes", lambda: 10**12)


def test_place_never_whole_places_lazy(gpu, tmp_path):
    lazy, arr = Lazy(), np.ones(1000)
    mm = np.memmap(tmp_path / "m.bin", np.float64, "w+", shape=(1000,))
    for x, y, s in [
        (lazy, arr, None),
        (arr, lazy, None),
        (arr, arr, lazy),
        (mm, mm, None),
    ]:
        out = lm._place(x, y, s)
        assert out[0] is x and out[1] is y and out[2] is s
    placed = lm._place(arr, arr, arr)
    assert all(isinstance(a, jax.Array) for a in placed)


def test_place_honours_memory_limit(gpu):
    x = np.ones(1000)
    need = 2 * 8 * 1000 + 5000  # x and y in float64 + chunk workspace
    fits = lm._place(x, x, None, workspace_bytes=5000, memory_limit_gb=need / 2**30)
    assert isinstance(fits[0], jax.Array)
    over = lm._place(
        x, x, None, workspace_bytes=5000, memory_limit_gb=(need - 1) / 2**30
    )
    assert over[0] is x


def test_place_respects_device_fraction(monkeypatch):
    x = np.ones(1000)
    monkeypatch.setattr(lm, "_free_device_bytes", lambda: 2 * 16000)  # 50% = data size
    assert isinstance(lm._place(x, x, None)[0], jax.Array)
    monkeypatch.setattr(lm, "_free_device_bytes", lambda: 2 * 16000 - 2)
    assert lm._place(x, x, None)[0] is x
    monkeypatch.setattr(lm, "_free_device_bytes", lambda: None)  # CPU / no stats
    assert lm._place(x, x, None)[0] is x


@pytest.mark.parametrize(("limit", "placed"), [(400_000, True), (300_000, False)])
def test_fit_passes_memory_limit_to_place(
    data, gpu, passes, monkeypatch, limit, placed
):
    seen = []
    real = lm.run_pass

    def spy(stats, theta, source, chunk, **k):
        seen.append(isinstance(source[0], jax.Array))
        return real(stats, theta, source, chunk, **k)

    monkeypatch.setattr(lm, "run_pass", spy)
    # data 48 kB; chunk = 3000 -> workspace 3000 * 3 * 8 * 4 = 288 kB
    lm.exact_lm_fit(exp_model, *data, EXP.p0, memory_limit_gb=limit / 2**30)
    assert set(seen) == {placed}


# ---- memmap / HDF5 / Zarr give the in-memory result ------------------------------------

BIG = 500_000
BIG_CHUNK = 65_536  # 8 chunks, padded tail


@pytest.fixture(scope="module")
def big():
    return EXP.data(BIG)


@pytest.fixture(scope="module")
def big_ref(big):
    return lm.exact_lm_fit(exp_model, *big, EXP.p0, chunk_size=BIG_CHUNK)


def _assert_same(out, ref):
    np.testing.assert_allclose(out["x"], ref["x"], rtol=1e-9, atol=0)
    assert abs(out["cost"] - ref["cost"]) <= 1e-12 * abs(ref["cost"])
    assert out["nfev"] == ref["nfev"]


def _memmap(path, a):
    mm = np.memmap(path, np.float64, "w+", shape=a.shape)
    mm[:] = a
    mm.flush()
    del mm
    return np.memmap(path, np.float64, "r", shape=a.shape)


@pytest.mark.serial
@pytest.mark.slow
def test_memmap_h5py_match_in_memory(big, big_ref, tmp_path):
    import h5py

    x, y = big
    mx, my = _memmap(tmp_path / "x.bin", x), _memmap(tmp_path / "y.bin", y)
    _assert_same(
        lm.exact_lm_fit(exp_model, mx, my, EXP.p0, chunk_size=BIG_CHUNK), big_ref
    )
    with h5py.File(tmp_path / "d.h5", "w") as f:
        f["x"], f["y"] = x, y
    with h5py.File(tmp_path / "d.h5", "r") as f:
        hx, hy = f["x"], f["y"]
        _assert_same(
            lm.exact_lm_fit(exp_model, hx, hy, EXP.p0, chunk_size=BIG_CHUNK), big_ref
        )
        popt, _ = curve_fit_large(
            exp_model, hx, hy, p0=EXP.p0, method="exact_streaming", chunk_size=BIG_CHUNK
        )
    np.testing.assert_allclose(popt, big_ref["x"], rtol=1e-9, atol=0)


@pytest.mark.serial
@pytest.mark.slow
def test_zarr_matches_in_memory(big, big_ref, tmp_path):
    zarr = pytest.importorskip("zarr")
    x, y = big
    zx = zarr.open(str(tmp_path / "x.zarr"), mode="w", shape=x.shape, dtype="f8")
    zy = zarr.open(str(tmp_path / "y.zarr"), mode="w", shape=y.shape, dtype="f8")
    zx[:], zy[:] = x, y
    _assert_same(
        lm.exact_lm_fit(exp_model, zx, zy, EXP.p0, chunk_size=BIG_CHUNK), big_ref
    )


# ---- host memory stays chunk-bounded on a 50M-point memmap ------------------------------


@pytest.mark.serial
@pytest.mark.slow
@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="host-memory bound, CPU only"
)
def test_tracemalloc_peak_bounded(tmp_path):
    n, chunk, block = 50_000_000, 1_000_000, 1_000_000
    a, b, c = 2.0, 0.5, 0.3
    paths = tmp_path / "x50m.bin", tmp_path / "y50m.bin"
    mx = np.memmap(paths[0], np.float64, "w+", shape=(n,))
    my = np.memmap(paths[1], np.float64, "w+", shape=(n,))
    for s in range(0, n, block):  # written in blocks, never held in RAM
        xs = 5.0 * np.arange(s, min(s + block, n)) / (n - 1)
        mx[s : s + block] = xs
        my[s : s + block] = a * np.exp(-b * xs) + c
    mx.flush(), my.flush()
    del mx, my
    try:
        x = np.memmap(paths[0], np.float64, "r", shape=(n,))
        y = np.memmap(paths[1], np.float64, "r", shape=(n,))
        tracemalloc.start()
        try:
            out = lm.exact_lm_fit(exp_model, x, y, [1.5, 0.4, 0.2], chunk_size=chunk)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        del x, y
        assert out["success"], out["message"]
        np.testing.assert_allclose(out["x"], [a, b, c], rtol=1e-6)
        assert peak < 8 * chunk * 8 + 64 * 2**20, f"peak {peak / 2**20:.1f} MiB"
    finally:
        for p in paths:
            p.unlink(missing_ok=True)
