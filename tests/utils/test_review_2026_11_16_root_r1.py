"""Regression tests for root-module review round 1 (2026-11-16)."""

import io
import subprocess
import sys
import threading
import time
import typing
import warnings

import numpy as np
import pytest

import nlsq
from nlsq import config as nlsq_config
from nlsq.callbacks import (
    CallbackChain,
    IterationLogger,
    ProgressBar,
    StopOptimization,
)
from nlsq.config import (
    JAXConfig,
    LargeDatasetConfig,
    MemoryConfig,
    configure_for_large_datasets,
    memory_context,
    set_memory_limits,
)
from nlsq.streaming import Chunked, Hybrid

INFO = {"gradient_norm": 1.0}


class _FakeBar:
    instances: typing.ClassVar[list] = []

    def __init__(self, total=None, desc=None, **_):
        self.n = 0
        self.closed = False
        _FakeBar.instances.append(self)

    def set_postfix(self, *_a, **_k):
        pass

    def update(self, k):
        self.n += k

    def close(self):
        self.closed = True


@pytest.fixture
def restore_config():
    saved = (JAXConfig._memory_config, JAXConfig._large_dataset_config)
    JAXConfig()  # make sure both are initialised
    saved = (
        nlsq_config.get_memory_config(),
        nlsq_config.get_large_dataset_config(),
    )
    yield
    JAXConfig.set_memory_config(saved[0])
    JAXConfig.set_large_dataset_config(saved[1])


# --- callbacks ---------------------------------------------------------------


@pytest.mark.serial
def test_unclosed_progressbar_exits_cleanly_after_jax():
    """An open tqdm bar at interpreter exit used to segfault once JAX was loaded."""
    pytest.importorskip("tqdm")
    code = (
        "import jax.numpy as jnp, numpy as np\n"
        "jnp.ones(3) + 1\n"
        "from nlsq.callbacks import ProgressBar\n"
        "pb = ProgressBar(max_nfev=10)\n"
        "pb(1, 1.0, np.zeros(2), {'nfev': 3, 'gradient_norm': 1.0})\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
        env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"},
    )
    assert proc.returncode == 0, proc.stderr[-500:]


def test_progressbar_reused_across_fits_restarts_bar():
    _FakeBar.instances = []
    pb = ProgressBar(max_nfev=100)
    pb.tqdm = _FakeBar
    for it in (1, 2, 3):
        pb(it, 1.0, np.zeros(1), {**INFO, "nfev": it + 1})
    first = _FakeBar.instances[0]
    for it in (1, 2):  # second fit: counters restart
        pb(it, 1.0, np.zeros(1), {**INFO, "nfev": it + 1})
    assert len(_FakeBar.instances) == 2
    assert first.closed
    assert first.n == 4  # first fit ran nfev 2..4
    assert _FakeBar.instances[1].n == 3  # second fit's own progress, not net zero
    pb.close()


def test_iteration_logger_reused_writes_header_per_fit():
    buf = io.StringIO()
    log = IterationLogger(file=buf)
    for it in (1, 2, 3, 1, 2):
        log(it, 1.0, np.zeros(1), INFO)
    assert buf.getvalue().count("NLSQ Optimization Log") == 2


@pytest.mark.filterwarnings("ignore:Callback raised exception:RuntimeWarning")
def test_callback_chain_failure_does_not_starve_later_callbacks():
    seen = []

    def bad(iteration, cost, params, info):
        raise RuntimeError("logger broke")

    def watcher(iteration, cost, params, info):
        seen.append(iteration)

    def stopper(iteration, cost, params, info):
        raise StopOptimization

    chain = CallbackChain(bad, watcher)
    with pytest.raises(RuntimeError, match="logger broke"):
        chain(1, 1.0, np.zeros(1), INFO)
    assert seen == [1]

    chain = CallbackChain(bad, stopper)
    with pytest.raises(StopOptimization):
        chain(1, 1.0, np.zeros(1), INFO)


def test_callback_base_docstring_says_one_indexed():
    from nlsq.callbacks import CallbackBase

    assert "0-indexed" not in (CallbackBase.__call__.__doc__ or "")


# --- device / types ----------------------------------------------------------


def test_get_device_info_survives_backend_failure(monkeypatch):
    import jax

    from nlsq.device import get_device_info

    def boom():
        raise RuntimeError("Unable to initialize backend 'bogus'")

    monkeypatch.setattr(jax, "default_backend", boom)
    info = get_device_info()
    assert "bogus" in info["jax_error"]
    assert info["jax_backend"] is None


def test_method_literal_includes_lm_and_callback_signature():
    from nlsq.types import CallbackFunction, MethodLiteral

    assert "lm" in typing.get_args(MethodLiteral)
    assert len(typing.get_args(CallbackFunction)[0]) == 4


# --- fit() routing -----------------------------------------------------------


def _model(x, a, b):
    import jax.numpy as jnp

    return a * jnp.exp(-b * x)


@pytest.fixture
def spy_fit_with_strategy(monkeypatch):
    calls = []

    def spy(strategy, *args, **kwargs):
        calls.append(strategy)
        return np.array([1.0, 1.0]), np.eye(2)

    monkeypatch.setattr("nlsq.streaming.large_fit.fit_with_strategy", spy)
    return calls


@pytest.mark.parametrize("n_points", [2000, 3000])
def test_fit_honours_explicit_strategy_at_both_sizes(spy_fit_with_strategy, n_points):
    x = np.linspace(0, 4, n_points)
    strategy = Chunked()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nlsq.fit(_model, x, x, p0=[1, 1], strategy=strategy, size_threshold=2500)
    assert spy_fit_with_strategy == [strategy]


def test_fit_strategy_rejects_conflicting_engine_option(spy_fit_with_strategy):
    x = np.linspace(0, 4, 3000)
    with pytest.raises(TypeError, match="strategy"):
        nlsq.fit(_model, x, x, p0=[1, 1], strategy=Chunked(), n_starts=3)


@pytest.mark.filterwarnings("ignore:goal tolerances are not applied:UserWarning")
def test_fit_hybrid_with_goal_above_threshold(spy_fit_with_strategy):
    x = np.linspace(0, 4, 3000)
    nlsq.fit(
        _model,
        x,
        x,
        p0=[1, 1],
        size_threshold=1000,
        method="hybrid_streaming",
        goal="fast",
    )
    assert isinstance(spy_fit_with_strategy[0], Hybrid)


def test_hybrid_legacy_options_map_to_config():
    strategy = nlsq._legacy_to_strategy(
        method="hybrid_streaming", chunk_size=777, multistart=True, n_starts=4
    )
    assert strategy.config.chunk_size == 777
    assert strategy.config.enable_multistart is True
    assert strategy.config.n_starts == 4


def test_hybrid_warns_on_options_it_cannot_honour():
    with pytest.warns(UserWarning, match="no effect with method='hybrid_streaming'"):
        nlsq._legacy_to_strategy(method="hybrid_streaming", stability="check")


@pytest.mark.serial
def test_force_cpu_zero_keeps_gpu_memory_default():
    import os

    if sys.platform != "linux":
        pytest.skip("XLA env defaults are set on Linux only")
    code = "import os, nlsq; print(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'))"
    env = {k: v for k, v in os.environ.items() if k != "XLA_PYTHON_CLIENT_MEM_FRACTION"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
        env={**env, "NLSQ_FORCE_CPU": "0", "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().splitlines()[-1] == "0.8"


# --- config ------------------------------------------------------------------


def test_configure_for_large_datasets_keeps_safety_factor(restore_config):
    set_memory_limits(4.0, gpu_memory_fraction=0.5, safety_factor=0.5)
    configure_for_large_datasets()
    cfg = nlsq_config.get_memory_config()
    assert cfg.safety_factor == 0.5
    assert cfg.gpu_memory_fraction == 0.5
    assert cfg.memory_limit_gb is None


def test_unconsumed_options_warn(restore_config, monkeypatch):
    with pytest.warns(UserWarning, match="no effect"):
        configure_for_large_datasets(progress_reporting=False)
    JAXConfig()._memory_config = None
    monkeypatch.setenv("NLSQ_CHUNK_SIZE_MB", "64")
    with pytest.warns(UserWarning, match="NLSQ_CHUNK_SIZE_MB"):
        JAXConfig.get_memory_config()
    JAXConfig()._large_dataset_config = None
    monkeypatch.setenv("NLSQ_DISABLE_AUTO_SOLVER_SELECTION", "1")
    with pytest.warns(UserWarning, match="NLSQ_DISABLE_AUTO_SOLVER_SELECTION"):
        JAXConfig.get_large_dataset_config()


@pytest.mark.parametrize("which", ["memory", "large_dataset"])
def test_overlapping_contexts_on_threads_do_not_leak(restore_config, which):
    if which == "memory":
        original = nlsq_config.get_memory_config()
        make = lambda v: memory_context(MemoryConfig(memory_limit_gb=v))
        current = lambda: nlsq_config.get_memory_config().memory_limit_gb
        original_value = original.memory_limit_gb
    else:
        original_value = (
            nlsq_config.get_large_dataset_config().solver_selection_thresholds
        )
        make = lambda v: nlsq_config.large_dataset_context(
            LargeDatasetConfig(
                solver_selection_thresholds={
                    "direct": int(v),
                    "iterative": int(v) + 1,
                    "chunked": int(v) + 2,
                }
            )
        )
        current = lambda: (
            nlsq_config.get_large_dataset_config().solver_selection_thresholds
        )

    started, release = threading.Event(), threading.Event()

    def t1():
        with make(1.0):
            started.set()
            release.wait(10)

    def t2():
        started.wait(10)
        with make(2.0):
            pass

    threads = [threading.Thread(target=t1), threading.Thread(target=t2)]
    for t in threads:
        t.start()
    started.wait(10)
    time.sleep(0.2)  # let T2 reach its context while T1 is still inside
    release.set()
    for t in threads:
        t.join(20)
    assert current() == original_value


# --- round 2 (Agy) -----------------------------------------------------------


def test_callback_chain_propagates_true_stop_signal():
    def stop_true(iteration, cost, params, info):
        return True

    def observer(iteration, cost, params, info):
        observer.seen = iteration

    chain = CallbackChain(stop_true, observer)
    assert chain(1, 1.0, np.zeros(1), INFO) is True
    assert observer.seen == 1  # later callbacks still ran
    assert not CallbackChain(observer)(1, 1.0, np.zeros(1), INFO)


def test_callback_chain_true_stops_a_real_fit():
    calls = []

    def stop_true(iteration, cost, params, info):
        calls.append(iteration)
        return True

    x = np.linspace(0, 1, 100)
    nlsq.curve_fit(
        lambda x, a: a * x,
        x,
        2.0 * x,
        p0=[1.0],
        bounds=([0], [10]),
        callback=CallbackChain(stop_true),
    )
    assert calls == [1]


@pytest.mark.filterwarnings("ignore:goal tolerances are not applied:UserWarning")
def test_fit_strategy_with_goal_does_not_forward_tolerances(spy_fit_with_strategy):
    x = np.linspace(0, 1, 100)
    strategy = Chunked()
    nlsq.fit(_model, x, x, p0=[1, 1], strategy=strategy, goal="fast")
    assert spy_fit_with_strategy == [strategy]


@pytest.mark.filterwarnings("ignore:goal tolerances are not applied:UserWarning")
def test_fit_hybrid_with_goal_below_threshold(monkeypatch):
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return np.array([1.0, 1.0]), np.eye(2)

    monkeypatch.setattr(nlsq, "curve_fit", spy)
    x = np.linspace(0, 1, 100)
    nlsq.fit(
        _model,
        x,
        x,
        p0=[1, 1],
        method="hybrid_streaming",
        goal="fast",
        size_threshold=1000,
    )
    assert not {"ftol", "gtol", "xtol"} & set(seen)


def test_reuse_detected_after_one_iteration_fit():
    buf = io.StringIO()
    log = IterationLogger(file=buf)
    for it in (1, 1, 2):  # fit 1 stopped after one iteration, fit 2 follows
        log(it, 1.0, np.zeros(1), INFO)
    assert buf.getvalue().count("NLSQ Optimization Log") == 2

    _FakeBar.instances = []
    pb = ProgressBar(max_nfev=100)
    pb.tqdm = _FakeBar
    pb(1, 1.0, np.zeros(1), {**INFO, "nfev": 2})
    pb(1, 1.0, np.zeros(1), {**INFO, "nfev": 2})
    assert len(_FakeBar.instances) == 2
    pb.close()


def test_every_lazy_name_is_exported_and_importable():
    missing = sorted(set(nlsq._LAZY_MODULES) - set(nlsq.__all__))
    assert not missing
    for name in nlsq.__all__:
        assert hasattr(nlsq, name), name


def test_stale_constants_removed_or_corrected():
    from nlsq import constants

    for name in ("STEP_ACCEPTANCE_THRESHOLD", "TRUST_RADIUS_DECREASE_FACTOR"):
        assert not hasattr(constants, name)
        assert name not in constants.__all__
    assert constants.MEMORY_SAFETY_FACTOR == 0.75
    assert set(constants.__all__) <= set(dir(constants))


@pytest.mark.skipif(sys.platform != "linux", reason="GPU check is Linux-only")
def test_check_gpu_availability_warns_when_backend_fails(monkeypatch):
    import jax

    from nlsq.device import check_gpu_availability

    def boom():
        raise RuntimeError("backend init failed")

    monkeypatch.delenv("NLSQ_SKIP_GPU_CHECK", raising=False)
    monkeypatch.setattr("nlsq.device.get_gpu_info", lambda: ("Fake GPU", 8.0))
    monkeypatch.setattr("nlsq.device.get_system_cuda_version", lambda: ("12.4", 12))
    monkeypatch.setattr(jax, "devices", boom)
    with pytest.warns(UserWarning, match="backend init failed"):
        assert check_gpu_availability(warn=True) is False


@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "linux", reason="GPU check is Linux-only")
def test_import_is_quiet_when_gpu_check_fails(tmp_path):
    """A failing import-time GPU check must not warn (python -W error -c 'import nlsq')."""
    import os

    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\necho 'Fake GPU, 8.9'\n")
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "NLSQ_SKIP_GPU_CHECK"}
    env.update(PATH=f"{tmp_path}:{env['PATH']}", JAX_PLATFORMS="bogus")
    proc = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import nlsq"],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-600:]


def test_goal_tolerances_dropped_with_hybrid_or_strategy_warn(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "nlsq.streaming.large_fit.fit_with_strategy",
        lambda strategy, *a, **k: calls.append(strategy) or (np.ones(2), np.eye(2)),
    )
    x = np.linspace(0, 4, 3000)
    with pytest.warns(UserWarning, match="goal.*not applied"):
        nlsq.fit(
            _model,
            x,
            x,
            p0=[1, 1],
            strategy=Chunked(),
            goal="fast",
        )
    with pytest.warns(UserWarning, match="goal.*not applied"):
        nlsq.fit(
            _model,
            x,
            x,
            p0=[1, 1],
            size_threshold=1000,
            method="hybrid_streaming",
            goal="fast",
        )
    assert len(calls) == 2


def test_lazy_getattr_keeps_inner_attribute_error_text(monkeypatch):
    import importlib

    def boom(path, *a, **k):
        raise AttributeError("inner boom while importing")

    monkeypatch.delitem(nlsq._LAZY_CACHE, "HealthStatus", raising=False)
    monkeypatch.setattr(importlib, "import_module", boom)
    with pytest.raises(AttributeError, match="inner boom while importing"):
        nlsq.HealthStatus  # noqa: B018


def test_context_locality_is_documented():
    from nlsq import config

    for obj in (
        config.set_memory_limits,
        config.configure_for_large_datasets,
        config.memory_context,
        config.large_dataset_context,
    ):
        assert "copy_context" in (obj.__doc__ or ""), obj.__name__
