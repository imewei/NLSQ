"""Round-7 review fixes: consistent boolean env parsing in nlsq.config."""

import pytest

from nlsq import config as cfg


class _FakeJaxConfig:
    def __init__(self):
        self.jax_compilation_cache_dir = None
        self.updates = {}

    def update(self, key, value):
        self.updates[key] = value


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "On", " 1 "])
def test_env_flag_truthy(monkeypatch, val):
    monkeypatch.setenv("NLSQ_TEST_FLAG", val)
    assert cfg._env_flag("NLSQ_TEST_FLAG")


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "2"])
def test_env_flag_falsy(monkeypatch, val):
    monkeypatch.setenv("NLSQ_TEST_FLAG", val)
    assert not cfg._env_flag("NLSQ_TEST_FLAG")


def test_env_flag_unset(monkeypatch):
    monkeypatch.delenv("NLSQ_TEST_FLAG", raising=False)
    assert not cfg._env_flag("NLSQ_TEST_FLAG")


def test_disable_persistent_cache_true(monkeypatch):
    monkeypatch.setenv("NLSQ_DISABLE_PERSISTENT_CACHE", "true")
    fake = _FakeJaxConfig()
    cfg.JAXConfig._configure_persistent_cache(object.__new__(cfg.JAXConfig), fake)
    assert fake.updates == {}


def test_empty_nlsq_cache_dir_means_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("NLSQ_DISABLE_PERSISTENT_CACHE", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("NLSQ_CACHE_MIN_COMPILE_TIME_SECS", raising=False)
    monkeypatch.setenv("NLSQ_JAX_CACHE_DIR", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    fake = _FakeJaxConfig()
    cfg.JAXConfig._configure_persistent_cache(object.__new__(cfg.JAXConfig), fake)
    expected = str(tmp_path / ".cache" / "nlsq" / "jax_cache")
    assert fake.updates.get("jax_compilation_cache_dir") == expected


def test_force_cpu_true(monkeypatch):
    import jax

    fake = _FakeJaxConfig()
    monkeypatch.setattr(jax, "config", fake)
    monkeypatch.setenv("NLSQ_FORCE_CPU", "TRUE")
    monkeypatch.setenv("NLSQ_DISABLE_PERSISTENT_CACHE", "1")
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    inst = object.__new__(cfg.JAXConfig)
    monkeypatch.setattr(inst, "_x64_enabled", True, raising=False)
    monkeypatch.setattr(inst, "_configure_gpu_memory", lambda: None, raising=False)
    inst._initialize_jax()
    assert fake.updates.get("jax_platform_name") == "cpu"
