"""Review 2026-11-12 f9: callback dispatch through decorators, NLSQ_SKIP_GPU_CHECK flag."""

import functools

import numpy as np
import pytest

from nlsq.callbacks import call_user_callback


def _dec(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def _a(params, cost, iteration, info):
    return (iteration, cost, params, info)


def test_wrapped_callback_with_reordered_names_gets_right_values():
    p = np.array([1.0])
    for cb in (_a, _dec(_a), _dec(_dec(_a))):
        assert call_user_callback(cb, 7, 1.5, p, {"k": 1}) == (7, 1.5, p, {"k": 1})


def test_wrapped_positional_callback_unchanged():
    def pos(a, b, c, d):
        return (a, b, c, d)

    assert call_user_callback(_dec(pos), 1, 2.0, "P", {}) == (1, 2.0, "P", {})


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE ", "On"])
def test_skip_gpu_check_truthy(monkeypatch, value):
    import nlsq.device as dev

    monkeypatch.setenv("NLSQ_SKIP_GPU_CHECK", value)
    monkeypatch.setattr(dev.sys, "platform", "linux")

    def boom():
        raise AssertionError("GPU probe ran")

    monkeypatch.setattr(dev, "get_gpu_info", boom)
    assert dev.check_gpu_availability(warn=False) is False  # returned before probing


@pytest.mark.parametrize("value", ["0", "off", ""])
def test_skip_gpu_check_falsy_probes(monkeypatch, value):
    import nlsq.device as dev

    monkeypatch.setenv("NLSQ_SKIP_GPU_CHECK", value)
    monkeypatch.setattr(dev.sys, "platform", "linux")
    called = []
    monkeypatch.setattr(
        dev, "get_gpu_info", lambda: (called.append(1), (None, None))[1]
    )
    dev.check_gpu_availability(warn=False)
    assert called
