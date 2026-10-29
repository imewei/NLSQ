"""Regression tests for the 2026-10-29 GUI review (nlsq/gui_qt).

Covers: 2-D surface data (point count is the LAST axis), abort stopping the
fit, generated-script correctness, clipboard blank cells, autosave None
bounds / 2-D round trip / private directory.

Tests that build a real Qt widget carry ``@pytest.mark.serial`` (OOM on CI).
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.gui_qt.adapters.data_adapter import (
    load_from_clipboard,
    parse_numeric_rows,
)
from nlsq.gui_qt.adapters.fit_adapter import (
    is_large_dataset,
    run_fit,
    validate_fit_inputs,
)
from nlsq.gui_qt.autosave import AutosaveManager, _ensure_private_dir
from nlsq.gui_qt.session_state import SessionState, n_points


def _surface(n=50):
    xx = np.vstack([np.linspace(0, 1, n), np.linspace(0, 1, n) ** 2])
    return xx, 2 * xx[0] + 3 * xx[1]


# ---------------------------------------------------------------------------
# 1. 2-D surface data
# ---------------------------------------------------------------------------


class TestSurfacePointCount:
    def test_n_points(self):
        assert n_points(np.zeros(7)) == 7
        assert n_points(np.zeros((2, 9))) == 9
        assert n_points([1.0, 2.0, 3.0]) == 3

    def test_validate_accepts_2d_xdata(self):
        xx, yy = _surface()
        validate_fit_inputs(xx, yy, None)  # raised "got 2 and 50" before the fix

    def test_validate_still_rejects_mismatch_and_empty(self):
        xx, yy = _surface()
        with pytest.raises(ValueError, match="same length"):
            validate_fit_inputs(xx, yy[:-1], None)
        with pytest.raises(ValueError, match="empty"):
            validate_fit_inputs(np.zeros((2, 0)), np.zeros(0), None)

    def test_is_large_dataset_counts_points_not_rows(self):
        assert is_large_dataset(np.zeros((2, 10)), threshold=5)

    def test_run_fit_on_surface(self):
        xx, yy = _surface()
        s = SessionState()
        s.xdata, s.ydata = xx, yy
        s.model_func = lambda xy, a, b: a * xy[0] + b * xy[1]
        s.p0 = [1.0, 1.0]
        res = run_fit(s)
        np.testing.assert_allclose(np.asarray(res.popt), [2.0, 3.0], rtol=1e-6)


def _mock_app_state(**attrs):
    base = {
        "xdata": None,
        "ydata": None,
        "sigma": None,
        "model_config": None,
        "p0": None,
        "bounds": None,
        "auto_p0": True,
    }
    base.update(attrs)
    app_state = MagicMock()
    app_state.state = SimpleNamespace(**base)
    return app_state


def _manager(app_state):
    mgr = AutosaveManager.__new__(AutosaveManager)
    mgr._app_state = app_state
    return mgr


class TestAutosave2DAndBounds:
    def test_2d_round_trip(self):
        x = np.vstack([np.arange(10.0), np.arange(10.0) * 2])
        y = np.arange(10.0)
        payload = json.loads(
            json.dumps(_manager(_mock_app_state(xdata=x, ydata=y))._serialize_state())
        )
        target = _mock_app_state()
        _manager(target)._restore_state(payload)
        np.testing.assert_array_equal(target.state.xdata, x)
        np.testing.assert_array_equal(target.state.ydata, y)

    def test_size_guard_counts_points_not_rows(self):
        n = 100_001
        x = np.vstack([np.zeros(n), np.zeros(n)])
        data = _manager(_mock_app_state(xdata=x, ydata=np.zeros(n)))._serialize_state()
        assert "xdata" not in data and "ydata" not in data

    def test_none_bounds_round_trip(self):
        bounds = ([None, 0.0], [10.0, None])
        data = _manager(_mock_app_state(bounds=bounds))._serialize_state()
        assert data["bounds"] == [[None, 0.0], [10.0, None]]

        target = _mock_app_state()
        _manager(target)._restore_state(json.loads(json.dumps(data)))
        assert target.state.bounds == ([None, 0.0], [10.0, None])

    def test_infinite_bounds_round_trip(self):
        bounds = ([-np.inf, 0.0], [np.inf, 1.0])
        data = _manager(_mock_app_state(bounds=bounds))._serialize_state()
        target = _mock_app_state()
        _manager(target)._restore_state(json.loads(json.dumps(data)))
        assert target.state.bounds == ([-np.inf, 0.0], [np.inf, 1.0])

    def test_old_float_only_bounds_still_load(self):
        target = _mock_app_state()
        _manager(target)._restore_state({"bounds": [[0, 1], [2, 3]]})
        assert target.state.bounds == ([0.0, 1.0], [2.0, 3.0])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ownership/symlink checks")
class TestAutosavePrivateDir:
    def test_dir_created_0700_and_loose_mode_tightened(self, tmp_path):
        d = tmp_path / "a"
        _ensure_private_dir(d)
        assert (d.stat().st_mode & 0o777) == 0o700
        d.chmod(0o755)
        _ensure_private_dir(d)
        assert (d.stat().st_mode & 0o777) == 0o700

    def test_symlinked_dir_rejected(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        with pytest.raises(PermissionError):
            _ensure_private_dir(link)

    def test_foreign_owner_rejected(self, tmp_path):
        d = tmp_path / "d"
        d.mkdir()
        with (
            patch("os.getuid", return_value=os.getuid() + 1),
            pytest.raises(PermissionError),
        ):
            _ensure_private_dir(d)

    def test_planted_symlink_tmp_is_not_written_through(self, tmp_path):
        victim = tmp_path / "victim"
        victim.write_text("ORIGINAL")
        (tmp_path / "session_autosave.tmp").symlink_to(victim)
        mgr = _manager(_mock_app_state(xdata=np.arange(3.0), ydata=np.arange(3.0)))
        mgr.autosave_completed = MagicMock()
        with (
            patch("nlsq.gui_qt.autosave.AUTOSAVE_DIR", tmp_path),
            patch(
                "nlsq.gui_qt.autosave.AUTOSAVE_FILE",
                tmp_path / "session_autosave.json",
            ),
        ):
            mgr._do_autosave()
        assert victim.read_text() == "ORIGINAL"
        assert json.loads((tmp_path / "session_autosave.json").read_text())["xdata"]
        # no stray temp files other than the planted link
        assert [p.name for p in tmp_path.glob("*.tmp")] == ["session_autosave.tmp"]


# ---------------------------------------------------------------------------
# 2. Abort
# ---------------------------------------------------------------------------


def _decay_state():
    s = SessionState()
    x = np.linspace(0, 10, 500)
    s.xdata = x
    s.ydata = (
        3.0 * np.exp(-0.4 * x)
        + 0.5
        + 0.01 * np.random.default_rng(0).normal(size=x.size)
    )
    s.model_func = lambda x, a, b, c: a * jnp.exp(-b * x) + c
    s.p0 = [30.0, 5.0, -10.0]
    s.gtol = s.ftol = s.xtol = 1e-15
    return s


class TestAbort:
    def test_callback_false_stops_fit_after_one_iteration(self):
        calls = []

        def cb(it, cost):
            calls.append(it)
            return False

        res = run_fit(_decay_state(), progress_callback=cb)
        assert len(calls) == 1
        assert res["aborted"] is True

    def test_uninterrupted_fit_runs_many_iterations(self):
        calls = []
        res = run_fit(
            _decay_state(), progress_callback=lambda it, c: calls.append(it) or True
        )
        assert len(calls) > 1
        assert not res.get("aborted", False)

    @pytest.mark.serial
    def test_worker_emits_aborted_not_error(self, qtbot):
        from nlsq.gui_qt.pages.fitting_options import FitWorker

        worker = FitWorker(_decay_state())
        worker.progress.connect(lambda it, c: worker.abort())
        got = {"aborted": 0, "error": [], "finished": 0}
        worker.aborted.connect(lambda: got.__setitem__("aborted", got["aborted"] + 1))
        worker.error.connect(got["error"].append)
        worker.finished.connect(lambda r: got.__setitem__("finished", 1))
        worker.run()
        assert got == {"aborted": 1, "error": [], "finished": 0}

    @pytest.mark.serial
    def test_error_text_containing_aborted_is_still_an_error(self, qtbot):
        from nlsq.gui_qt.pages.fitting_options import FitWorker

        def boom(*a, **k):
            raise RuntimeError("solver aborted: singular matrix")

        worker = FitWorker(_decay_state())
        errors, aborted = [], []
        worker.error.connect(errors.append)
        worker.aborted.connect(lambda: aborted.append(1))
        with patch("nlsq.gui_qt.adapters.fit_adapter.run_fit", boom):
            worker.run()
        assert len(errors) == 1 and not aborted


# ---------------------------------------------------------------------------
# 3. Generated script / 1. export of 2-D data
# ---------------------------------------------------------------------------


def _run_generated(code: str) -> dict:
    ns: dict = {}
    exec(compile(code, "<generated>", "exec"), ns)
    return ns


@pytest.mark.serial
class TestGeneratedScript:
    def _page(self, app_state, qtbot):
        from nlsq.gui_qt.pages.export import ExportPage

        page = ExportPage(app_state)
        qtbot.addWidget(page)
        return page

    def test_polynomial_script_fits_all_coefficients(self, app_state, qtbot):
        from nlsq.gui_qt.adapters.model_adapter import get_model

        x = np.linspace(0, 2, 15)
        app_state.set_data(x, 1 + 2 * x + 3 * x**2)
        app_state.set_model(
            "polynomial", {"degree": 2}, get_model("polynomial", {"degree": 2})
        )
        ns = _run_generated(self._page(app_state, qtbot).generate_python_code())
        np.testing.assert_allclose(ns["popt"], [1, 2, 3], rtol=1e-6)

    def test_data_embedded_at_full_precision(self, app_state, qtbot):
        xp = np.array([0.123456789012, 1.000000001, 2.5])
        app_state.set_data(xp, xp)
        app_state.set_model("builtin", {"name": "linear"}, lambda x, a, b: a * x + b)
        ns = _run_generated(self._page(app_state, qtbot).generate_python_code())
        np.testing.assert_array_equal(ns["xdata"], xp)

    def test_custom_model_with_other_name_runs(self, app_state, qtbot):
        code = "def my_line(x, a, b):\n    return a * x + b\n"
        fn = {}
        exec(code, fn)
        x = np.linspace(0, 4, 10)
        app_state.set_data(x, 2.0 * x + 1.0)
        app_state.set_model(
            "custom", {"code": code, "function": "my_line"}, fn["my_line"]
        )
        ns = _run_generated(self._page(app_state, qtbot).generate_python_code())
        np.testing.assert_allclose(ns["popt"], [2.0, 1.0], rtol=1e-6)

    def test_unbounded_params_script_runs(self, app_state, qtbot):
        code = "def model(x, a, b):\n    return a * x + b\n"
        fn = {}
        exec(code, fn)
        x = np.linspace(0, 4, 10)
        app_state.set_data(x, 2.0 * x + 1.0)
        app_state.set_model("custom", {"code": code, "function": "model"}, fn["model"])
        app_state.state.p0 = [1.0, 0.5]
        app_state.state.bounds = ([None, 0.0], [10.0, None])
        script = self._page(app_state, qtbot).generate_python_code()
        bounds_line = next(ln for ln in script.splitlines() if ln.startswith("bounds"))
        assert "None" not in bounds_line
        ns = _run_generated(script)
        np.testing.assert_allclose(ns["popt"], [2.0, 1.0], rtol=1e-5)

    def test_surface_csv_json_and_script(self, app_state, qtbot, tmp_path):
        n = 1500
        xx, yy = _surface(n)

        def surf(xy, a, b):
            return a * xy[0] + b * xy[1]

        app_state.set_data(xx, yy)
        app_state.set_model("builtin", {"name": "linear"}, surf)
        app_state.set_fit_result(
            SimpleNamespace(
                x=np.array([2.0, 3.0]), pcov=None, success=True, nfev=3, popt=None
            )
        )
        page = self._page(app_state, qtbot)

        csv_path = tmp_path / "surf.csv"
        page.export_csv(str(csv_path), include_data=True)
        rows = csv_path.read_text().splitlines()
        assert rows[0] == "x0,x1,y,y_fit,residuals"
        assert len(rows) == n + 1
        assert "..." not in "\n".join(rows)
        first = [float(v) for v in rows[1].split(",")]
        assert first[:3] == [0.0, 0.0, 0.0]

        assert json.loads(page._generate_json())["statistics"]["n_points"] == n
        assert "..." not in page.generate_python_code().split("# Fitting options")[0]

    def test_1d_csv_header_unchanged(self, app_state, qtbot, tmp_path):
        app_state.set_data(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        app_state.set_model("custom", model_func=lambda x, a: a * x)
        page = self._page(app_state, qtbot)
        p = tmp_path / "d.csv"
        page.export_csv(str(p), include_data=True)
        assert p.read_text().splitlines() == [
            "x,y,y_fit,residuals",
            "1.0,2.0,,",
            "2.0,4.0,,",
        ]


# ---------------------------------------------------------------------------
# 4. Clipboard blank cells
# ---------------------------------------------------------------------------


class TestClipboardBlankCells:
    def test_blank_cell_is_nan_and_later_values_keep_their_column(self):
        rows = parse_numeric_rows(["1\t\t0.1", "2\t4\t"], "\t")
        assert np.isnan(rows[0][1]) and rows[0][2] == 0.1
        assert rows[1][:2] == [2.0, 4.0] and np.isnan(rows[1][2])

    def test_whitespace_delimiter_unchanged(self):
        assert parse_numeric_rows(["1  2   3"], None) == [[1.0, 2.0, 3.0]]

    def test_non_numeric_names_the_line(self):
        with pytest.raises(ValueError, match="line 2"):
            parse_numeric_rows(["1,2", "a,b"], ",")

    def test_load_from_clipboard_shares_parser_and_keeps_trailing_cell(self):
        cfg = {"columns": {"x": 0, "y": 1, "sigma": 2}}
        _, y, sigma = load_from_clipboard("1\t4\t0.1\n2\t5\t", cfg)
        np.testing.assert_array_equal(y, [4.0, 5.0])
        assert sigma[0] == 0.1 and np.isnan(sigma[1])


@pytest.mark.serial
class TestClipboardPage:
    def test_blank_sigma_cell_keeps_columns_and_is_rejected(self, qtbot, app_state):
        from nlsq.gui_qt.pages.data_loading import DataLoadingPage

        page = DataLoadingPage(app_state)
        qtbot.addWidget(page)
        page._clipboard_text.setPlainText("1\t\t0.1\n2\t4\t")
        with patch("nlsq.gui_qt.pages.data_loading.QMessageBox.warning") as warn:
            page._on_parse_clipboard()

        raw = page._raw_data
        assert raw.shape == (2, 3)
        assert raw[0, 0] == 1.0 and np.isnan(raw[0, 1]) and raw[0, 2] == 0.1
        assert raw[1, 1] == 4.0 and np.isnan(raw[1, 2])
        # default x=col0, y=col1: the NaN y cell must be flagged, not silently
        # replaced by the sigma value that used to shift left into it
        warn.assert_called_once()
        assert not page._apply_btn.isEnabled()


class TestGuiModelValidatorParity:
    """The GUI validator must reject the same os-alias / file-writer payloads as the CLI one."""

    @pytest.mark.parametrize(
        "body",
        [
            "import posix\n    posix.posix_spawn('/bin/sh', ['sh'], {})",
            "import io\n    io.FileIO('m', 'w')",
            "import numpy as np\n    np.save('m.npy', x)",
        ],
    )
    def test_rejects_bypass_payloads(self, body):
        from nlsq.gui_qt.adapters.model_adapter import (
            SecurityError,
            _SafeASTValidator,
        )

        code = f"def model(x, a):\n    {body}\n    return a * x\n"
        with pytest.raises(SecurityError):
            _SafeASTValidator().validate(code)


# ---------------------------------------------------------------------------
# 7. Varargs-model export, list-typed data, 0-d n_points
# ---------------------------------------------------------------------------


def _export_page(model_func, popt, xdata, ydata):
    """ExportPage with a mocked app state (no Qt widget is constructed)."""
    from nlsq.gui_qt.pages.export import ExportPage

    page = ExportPage.__new__(ExportPage)
    page._app_state = _mock_app_state(
        xdata=xdata,
        ydata=ydata,
        model_func=model_func,
        model_config=None,
        fit_result=SimpleNamespace(
            x=np.asarray(popt),
            pcov=np.diag([0.01] * len(popt)),
            success=True,
            nfev=5,
        ),
    )
    return page


class TestVarargsExport:
    @staticmethod
    def _page():
        x = np.linspace(0.0, 1.0, 10)
        return _export_page(lambda x, *c: c[0] + c[1] * x, [1.0, 2.0], x, 1.0 + 2.0 * x)

    def test_param_names_padded_to_popt_length(self):
        assert self._page()._get_param_names() == ["c0", "c1"]

    def test_csv_has_parameter_rows(self, tmp_path):
        out = tmp_path / "p.csv"
        self._page().export_csv(str(out), include_data=False)
        rows = out.read_text().splitlines()
        assert [r.split(",")[0] for r in rows] == ["parameter", "c0", "c1"]
        assert float(rows[2].split(",")[1]) == 2.0

    def test_json_has_parameters(self):
        params = json.loads(self._page()._generate_json())["parameters"]
        assert set(params) == {"c0", "c1"}
        assert params["c1"]["value"] == 2.0

    def test_named_model_names_unchanged(self):
        x = np.linspace(0.0, 1.0, 5)
        page = _export_page(lambda x, a, b: a + b * x, [1.0, 2.0], x, x)
        assert page._get_param_names() == ["a", "b"]


class TestListDataExport:
    def test_json_with_list_data_and_jax_fit(self):
        page = _export_page(
            lambda x, a: jnp.array([a * xi for xi in x]),
            [2.0],
            [1.0, 2.0],
            [2.0, 4.5],
        )
        stats = json.loads(page._generate_json())["statistics"]
        assert stats["rmse"] == pytest.approx(np.sqrt(0.25 / 2))


class TestAutosaveListData:
    def test_list_arrays_serialize(self):
        data = _manager(
            _mock_app_state(
                xdata=[1.0, 2.0, 3.0], ydata=[4.0, 5.0, 6.0], sigma=[0.1, 0.1, 0.1]
            )
        )._serialize_state()
        assert data["xdata"] == [1.0, 2.0, 3.0]
        assert data["ydata"] == [4.0, 5.0, 6.0]
        assert data["sigma"] == [0.1, 0.1, 0.1]


def test_n_points_zero_dim():
    assert n_points(np.float64(3.0)) == 1
    assert n_points(np.array(5.0)) == 1
    assert n_points(np.zeros(4)) == 4
    assert n_points(np.zeros((2, 6))) == 6
