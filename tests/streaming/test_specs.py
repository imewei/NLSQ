"""Tests for the large-fit strategy dataclasses."""

import dataclasses

import pytest

from nlsq.streaming.specs import Auto, Chunked, Exact, Hybrid, InMemory, Multistart


def test_auto_resolves_below_threshold_to_in_memory():
    auto = Auto()
    assert auto.resolve(auto.size_threshold - 1) is auto.in_memory


def test_auto_resolves_at_threshold_to_chunked():
    auto = Auto()
    assert auto.resolve(auto.size_threshold) is auto.chunked


def test_auto_without_size_detection_is_always_chunked():
    auto = Auto(auto_size_detection=False)
    assert auto.resolve(10) is auto.chunked


def test_exact_has_no_loss_field():
    with pytest.raises(TypeError):
        Exact(loss="huber")  # type: ignore[call-arg]


def test_strategies_are_frozen():
    for obj in (Multistart(), InMemory(), Chunked(), Hybrid(), Exact(), Auto()):
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.foo = 1  # type: ignore[attr-defined]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Auto().size_threshold = 5  # type: ignore[misc]


def test_multistart_defaults():
    ms = Multistart()
    assert (ms.n_starts, ms.sampler, ms.center_on_p0, ms.scale_factor, ms.enabled) == (
        10,
        "lhs",
        True,
        1.0,
        True,
    )


def test_inmemory_and_chunked_default_multistart_disabled():
    assert InMemory().multistart.enabled is False
    assert Chunked().multistart.enabled is False


def test_chunked_has_no_stability_fields():
    with pytest.raises(TypeError):
        Chunked(stability="auto")  # type: ignore[call-arg]


def test_options_defaults_not_shared():
    assert InMemory().options == {}
    assert InMemory().options is not InMemory().options


def test_streaming_exports():
    from nlsq.streaming import Auto, Chunked, Exact, Hybrid, InMemory, Multistart

    assert all((Auto, Chunked, Exact, Hybrid, InMemory, Multistart))
