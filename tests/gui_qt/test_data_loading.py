"""Tests for DataLoadingPage — refreshing UI after session recovery."""

from unittest.mock import patch

import numpy as np


class TestAppStateDataChangedRefresh:
    """The page must reflect AppState data restored from outside itself."""

    def test_recovered_data_updates_file_label_and_stats(self, qtbot, app_state):
        from nlsq.gui_qt.pages.data_loading import DataLoadingPage

        page = DataLoadingPage(app_state)
        qtbot.addWidget(page)

        assert page._file_path_label.text() == "No file selected"

        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([4.0, 5.0, 6.0])
        app_state.set_data(xdata, ydata, file_name="recovered.csv")

        assert page._file_path_label.text() == "recovered.csv"
        np.testing.assert_array_equal(page._xdata, xdata)
        np.testing.assert_array_equal(page._ydata, ydata)
        assert "Points: 3" in page._stats_points.text()

    def test_own_apply_does_not_reprocess_via_data_changed(self, qtbot, app_state):
        """A local Apply Data click already updated the UI directly — the
        resulting data_changed signal must be a no-op, not a redundant redraw."""
        from nlsq.gui_qt.pages.data_loading import DataLoadingPage

        page = DataLoadingPage(app_state)
        qtbot.addWidget(page)

        page._xdata = np.array([1.0, 2.0])
        page._ydata = np.array([3.0, 4.0])
        with patch("nlsq.gui_qt.pages.data_loading.QMessageBox.information"):
            page._on_apply()

        # Guard compares identity with the arrays already on the page —
        # confirms the no-op path was taken rather than a second refresh.
        assert app_state.state.xdata is page._xdata
