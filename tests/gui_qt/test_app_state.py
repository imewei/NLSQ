"""Tests for AppState — stale fit_result invalidation on data/model change."""

import numpy as np


class TestFitResultInvalidation:
    """set_data()/set_model() must clear a stale fit_result."""

    def test_set_data_clears_completed_fit_result(self, app_state):
        app_state.set_fit_result(object())
        assert app_state.state.fit_result is not None

        app_state.set_data(np.array([1.0, 2.0]), np.array([3.0, 4.0]))

        assert app_state.state.fit_result is None

    def test_set_model_clears_completed_fit_result(self, app_state):
        app_state.set_fit_result(object())
        assert app_state.state.fit_result is not None

        app_state.set_model("builtin", config={"name": "linear"})

        assert app_state.state.fit_result is None

    def test_set_data_emits_fit_completed_none_when_clearing(self, app_state, qtbot):
        app_state.set_fit_result(object())

        with qtbot.waitSignal(app_state.fit_completed, timeout=1000) as blocker:
            app_state.set_data(np.array([1.0]), np.array([2.0]))

        assert blocker.args == [None]

    def test_set_data_without_prior_fit_does_not_emit_fit_completed(
        self, app_state, qtbot
    ):
        received = []
        app_state.fit_completed.connect(received.append)

        app_state.set_data(np.array([1.0]), np.array([2.0]))

        assert received == []
