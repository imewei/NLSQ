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

    def test_reset_clears_completed_fit_result_and_emits(self, app_state, qtbot):
        app_state.set_fit_result(object())

        with qtbot.waitSignal(app_state.fit_completed, timeout=1000) as blocker:
            app_state.reset()

        assert blocker.args == [None]
        assert app_state.state.fit_result is None

    def test_reset_without_prior_fit_does_not_emit_fit_completed(
        self, app_state, qtbot
    ):
        received = []
        app_state.fit_completed.connect(received.append)

        app_state.reset()

        assert received == []

    def test_set_fit_running_clears_stale_result_before_refit(self, app_state, qtbot):
        """Re-running a fit on already-loaded data (no data/model change)
        must invalidate the previous fit_result immediately, so an error or
        abort during the re-fit doesn't leave Results/Export showing the
        prior run's now-unrelated numbers -- PageState.can_access() gates
        purely on fit_result is not None, with no regard for fit_running.
        """
        app_state.set_fit_result(object())

        with qtbot.waitSignal(app_state.fit_completed, timeout=1000) as blocker:
            app_state.set_fit_running(True)

        assert blocker.args == [None]
        assert app_state.state.fit_result is None

    def test_set_fit_running_without_prior_fit_does_not_emit_fit_completed(
        self, app_state, qtbot
    ):
        received = []
        app_state.fit_completed.connect(received.append)

        app_state.set_fit_running(True)

        assert received == []
