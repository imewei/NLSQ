"""Current CLI data workflows must select the same measurements in the GUI."""

import numpy as np
import pytest

from nlsq.gui_qt.pages.data_loading import DataLoadingPage

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("path_key", ["input_file", "file"])
@pytest.mark.parametrize("named_columns", [False, True])
def test_yaml_data_import_resolves_relative_file_and_columns(
    app_state, qtbot, tmp_path, path_key, named_columns
):
    config_dir = tmp_path / "workflow"
    config_dir.mkdir()
    (config_dir / "data.csv").write_text(
        "signal,error,time\n3,0.1,1\n5,0.2,2\n7,0.3,3\n"
    )
    columns = (
        "{x: time, y: signal, sigma: error}"
        if named_columns
        else "{x: 2, y: 0, sigma: 1}"
    )
    workflow = config_dir / "workflow.yaml"
    workflow.write_text(
        f"data:\n  {path_key}: data.csv\n  format: csv\n  columns: {columns}\n"
    )
    page = DataLoadingPage(app_state)
    qtbot.addWidget(page)

    page.load_yaml_config(workflow)

    np.testing.assert_array_equal(page._xdata, [1, 2, 3])
    np.testing.assert_array_equal(page._ydata, [3, 5, 7])
    np.testing.assert_array_equal(page._sigma, [0.1, 0.2, 0.3])


def test_yaml_surface_columns_enable_2d_mode(app_state, qtbot, tmp_path):
    (tmp_path / "surface.csv").write_text("z,x,y\n5,1,2\n8,2,3\n11,3,4\n")
    workflow = tmp_path / "surface.yaml"
    workflow.write_text(
        "data:\n  input_file: surface.csv\n  columns: {x: x, y: y, z: z, sigma: null}\n"
    )
    page = DataLoadingPage(app_state)
    qtbot.addWidget(page)

    page.load_yaml_config(workflow)

    np.testing.assert_array_equal(page._xdata, [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(page._ydata, [5, 8, 11])
    assert page._sigma is None
