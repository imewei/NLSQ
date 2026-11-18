"""Regression tests for the 2026-11-16 diagnostics review (round 1)."""

from __future__ import annotations

import json
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.diagnostics import (
    DiagnosticLevel,
    DiagnosticsConfig,
    DiagnosticsReport,
    GradientMonitor,
    HealthStatus,
    IdentifiabilityAnalyzer,
    IdentifiabilityReport,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
    ParameterSensitivityAnalyzer,
    PluginRegistry,
    PluginResult,
    create_health_report,
    run_plugins,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    PluginRegistry.clear()
    yield
    PluginRegistry.clear()


def _jac(cols):
    return np.column_stack(cols)


# ---------------------------------------------------------------- correlation
class TestCorrelationSign:
    def test_tradeoff_pair_is_negative_one_jacobian_path(self):
        x = np.linspace(0, 5, 60)
        c = np.exp(-0.7 * x)
        # (a + b) * exp(-c x): d/da == d/db exactly, d/dc independent.
        j = _jac([c, c, -x * 2.0 * c])
        rep = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze(j)
        corr = rep.correlation_matrix
        assert corr is not None
        # a and b trade off one-for-one: correlation must not be +1.
        assert corr[0, 1] == pytest.approx(-1.0, abs=1e-6)
        assert all(r < 0 for _, _, r in rep.highly_correlated_pairs)

    def test_tradeoff_pair_is_negative_one_fim_path(self):
        x = np.linspace(0, 5, 60)
        c = np.exp(-0.7 * x)
        j = _jac([c, c, -x * c])
        rep = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze_from_fim(j.T @ j)
        assert rep.correlation_matrix is not None
        assert rep.correlation_matrix[0, 1] == pytest.approx(-1.0, abs=1e-6)

    def test_curve_fit_rank_deficient_reports_negative_corr(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 4, 80)
        y = 3.0 * np.exp(-0.8 * x) + rng.normal(0, 0.01, x.size)

        def model(x, a, b, c):
            return (a + b) * jnp.exp(-c * x)

        r = curve_fit(model, x, y, p0=[1.0, 2.0, 0.7], compute_diagnostics=True)
        corr = r.diagnostics.identifiability.correlation_matrix
        assert corr is not None
        assert corr[0, 1] < -0.99

    def test_zero_column_does_not_suppress_other_correlations(self):
        rng = np.random.default_rng(1)
        c = rng.normal(size=50)
        j = _jac([c, c + 1e-4 * rng.normal(size=50), np.zeros(50)])
        rep = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze(j)
        assert rep.correlation_matrix is not None
        assert any(p[:2] == (0, 1) for p in rep.highly_correlated_pairs)

    def test_zero_column_fim_path(self):
        rng = np.random.default_rng(2)
        c = rng.normal(size=50)
        j = _jac([c, c + 1e-4 * rng.normal(size=50), np.zeros(50)])
        rep = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze_from_fim(j.T @ j)
        assert rep.correlation_matrix is not None
        assert any(p[:2] == (0, 1) for p in rep.highly_correlated_pairs)

    def test_full_rank_correlation_unchanged(self):
        rng = np.random.default_rng(3)
        j = rng.normal(size=(40, 3))
        corr = (
            IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze(j).correlation_matrix
        )
        cov = np.linalg.inv(j.T @ j)
        d = np.sqrt(np.diag(cov))
        np.testing.assert_allclose(corr, cov / np.outer(d, d), atol=1e-8)


# ------------------------------------------------------------------ gradients
class TestGradientMonitor:
    def test_no_vanishing_false_positive_on_small_scale_convergence(self):
        m = GradientMonitor(DiagnosticsConfig())
        # Healthy convergence in tiny units: steady decay, cost drops.
        for i in range(12):
            m.record_gradient(
                np.array([3e-7, 3e-7]) * 0.3**i, cost=1e-3 * 0.5**i + 1e-5
            )
        rep = m.get_report()
        assert not rep.vanishing_detected

    def test_no_vanishing_false_positive_on_noise_floor_convergence(self):
        m = GradientMonitor(DiagnosticsConfig())
        for i in range(40):
            g = 10.0 * 0.6**i
            m.record_gradient(np.array([g, g]), cost=0.5 + 5.0 * 0.6**i)
        assert not m.get_report().vanishing_detected

    def test_grad002_names_scaled_parameters(self):
        m = GradientMonitor(DiagnosticsConfig(imbalance_threshold=1e3))
        m.record_gradient(
            np.array([1.0, 1e-3, 1e-1]), 1.0, params=np.array([1e-8, 1, 1])
        )
        issue = next(i for i in m.get_report().issues if i.code == "GRAD-002")
        # |g*p| = [1e-8, 1e-3, 1e-1]: params 0 (smallest) and 2 (largest).
        assert set(issue.affected_parameters) == {0, 2}

    def test_grad004_message_is_consistent(self):
        m = GradientMonitor(DiagnosticsConfig(exploding_threshold=1e8))
        m.record_gradient(np.array([1e-3, 1e-3]), cost=1.0)
        m.record_gradient(np.array([1e6, 1e6]), cost=1.0)
        issue = next(i for i in m.get_report().issues if i.code == "GRAD-004")
        assert "exceeds threshold" not in issue.message
        assert "initial" in issue.message
        assert "initial_gradient_norm" in issue.details

    def test_callback_forwards_user_stop_request(self):
        m = GradientMonitor(DiagnosticsConfig())
        cb = m.create_callback(user_callback=lambda it, cost, p, info: True)
        assert cb(iteration=1, cost=1.0, params=np.array([1.0]), info={}) is True

    def test_record_gradient_shape_change_leaves_state_untouched(self):
        m = GradientMonitor(DiagnosticsConfig())
        m.record_gradient(np.ones(3), 1.0)
        with pytest.raises(ValueError, match="shape"):
            m.record_gradient(np.ones(2), 1.0)
        assert m.iteration_count == 1
        assert len(m._gradient_norm_history) == 1


# ----------------------------------------------------------------- to_dict/json
class TestStrictJson:
    def test_rank_deficient_full_report_is_strict_json(self):
        x = np.linspace(0, 4, 60)
        y = 3.0 * np.exp(-0.8 * x)

        def model(x, a, b, c):
            return (a + b) * jnp.exp(-c * x)

        r = curve_fit(
            model,
            x,
            y,
            p0=[1.0, 2.0, 0.7],
            compute_diagnostics=True,
            diagnostics_level=DiagnosticLevel.FULL,
        )
        json.dumps(r.diagnostics.to_dict(), allow_nan=False)

    def test_plugin_numpy_data_is_json_safe(self):
        class P:
            name = "np-plugin"

            def analyze(self, jacobian, parameters, residuals, **ctx):
                return PluginResult(
                    plugin_name="np-plugin",
                    data={"arr": np.arange(3), "x": np.float64(np.nan)},
                )

        PluginRegistry.register(P())
        results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        rep = create_health_report(plugin_results=results)
        json.dumps(rep.to_dict(), allow_nan=False)


# ---------------------------------------------------------- level / plugin wiring
def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data():
    rng = np.random.default_rng(5)
    x = np.linspace(0, 4, 60)
    return x, 2.0 * np.exp(-0.9 * x) + rng.normal(0, 0.01, x.size)


class TestCurveFitWiring:
    def test_level_kwarg_honoured_with_config(self):
        x, y = _data()
        r = curve_fit(
            _model,
            x,
            y,
            p0=[1.0, 1.0],
            compute_diagnostics=True,
            diagnostics_level=DiagnosticLevel.FULL,
            diagnostics_config=DiagnosticsConfig(verbose=False),
        )
        assert r.diagnostics.config.level == DiagnosticLevel.FULL
        assert r.diagnostics.sloppy_model is not None

    def test_config_level_kept_when_kwarg_default(self):
        x, y = _data()
        r = curve_fit(
            _model,
            x,
            y,
            p0=[1.0, 1.0],
            compute_diagnostics=True,
            diagnostics_config=DiagnosticsConfig(
                level=DiagnosticLevel.FULL, verbose=False
            ),
        )
        assert r.diagnostics.sloppy_model is not None

    def test_registered_plugin_runs_in_curve_fit(self):
        calls = []

        class P:
            name = "tiny"

            def analyze(self, jacobian, parameters, residuals, **ctx):
                calls.append(jacobian.shape)
                return PluginResult(
                    plugin_name="tiny",
                    data={"n": len(parameters)},
                    issues=[
                        ModelHealthIssue(
                            category=IssueCategory.IDENTIFIABILITY,
                            severity=IssueSeverity.WARNING,
                            code="TINY-001",
                            message="domain check failed",
                            affected_parameters=None,
                            details={},
                            recommendation="fix it",
                        )
                    ],
                )

        PluginRegistry.register(P())
        x, y = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = curve_fit(_model, x, y, p0=[1.0, 1.0], compute_diagnostics=True)
        assert len(calls) == 1
        assert "tiny" in r.diagnostics.plugin_results
        assert any(i.code == "TINY-001" for i in r.diagnostics.all_issues)


# ---------------------------------------------------------------------- plugins
class _Plugin:
    def __init__(self, name, issues=None, data=None):
        self._name = name
        self._issues = issues if issues is not None else []
        self._data = data or {}

    @property
    def name(self):
        return self._name

    def analyze(self, jacobian, parameters, residuals, **ctx):
        return PluginResult(
            plugin_name=self._name, data=self._data, issues=self._issues
        )


class TestPlugins:
    def test_malformed_issue_is_dropped_not_fatal(self):
        PluginRegistry.register(_Plugin("bad", issues=["oops"]))
        with pytest.warns(UserWarning, match="invalid issue"):
            results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        assert results["bad"].issues == []
        create_health_report(plugin_results=results)  # must not raise

    def test_results_keyed_by_registered_name(self):
        class Shifty(_Plugin):
            @property
            def name(self):
                return self._name if not getattr(self, "_seen", False) else "same"

        a, b = Shifty("a", data={"who": "a"}), Shifty("b", data={"who": "b"})
        PluginRegistry.register(a)
        PluginRegistry.register(b)
        a._seen = b._seen = True
        results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        assert set(results) == {"a", "b"}

    def test_register_reads_name_outside_lock(self):
        class Reentrant:
            @property
            def name(self):
                PluginRegistry.get("anything")  # takes the lock
                return "reentrant"

            def analyze(self, jacobian, parameters, residuals, **ctx):
                return PluginResult(plugin_name="reentrant")

        PluginRegistry.register(Reentrant())  # deadlocks if the lock is held
        assert PluginRegistry.get("reentrant") is not None

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            PluginRegistry.register(_Plugin(""))


# --------------------------------------------------------------- types / report
class TestTypes:
    def test_legacy_report_failed_analysis_not_healthy(self):
        rep = DiagnosticsReport(
            identifiability=IdentifiabilityReport(
                available=False, error_message="SVD failed"
            )
        )
        assert rep.overall_status != HealthStatus.HEALTHY
        modern = create_health_report(
            identifiability=IdentifiabilityReport(
                available=False, error_message="SVD failed"
            )
        )
        assert modern.status == rep.overall_status

    def test_empty_fim_rejected_cleanly(self):
        rep = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze_from_fim(
            np.empty((0, 0))
        )
        assert rep.error_message == "Empty FIM matrix"
        sens = ParameterSensitivityAnalyzer(DiagnosticsConfig()).analyze_from_fim(
            np.empty((0, 0))
        )
        assert not sens.available

    def test_sloppy_section_shown_at_basic_level(self):
        from nlsq.diagnostics import ParameterSensitivityReport

        sloppy = ParameterSensitivityReport(
            available=True, is_sloppy=True, eigenvalue_range=4.5
        )
        rep = create_health_report(
            sloppy_model=sloppy,
            config=DiagnosticsConfig(level=DiagnosticLevel.BASIC),
        )
        assert "--- Sloppy Model ---" in rep.summary()


# ------------------------------------------------------------ round 2 (Codex)
class TestRound2:
    def test_small_but_resolved_null_component_still_correlated(self):
        j = np.array([[1, 1, 0], [0, 1e-6, 1], [0, 0, 0.0]])
        for rep in (
            IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze(j),
            IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze_from_fim(j.T @ j),
        ):
            pairs = {p[:2] for p in rep.highly_correlated_pairs}
            assert {(0, 1), (0, 2), (1, 2)} <= pairs

    def test_grad001_invariant_to_parameter_units(self):
        def run(g, p):
            m = GradientMonitor(DiagnosticsConfig())
            for _ in range(10):
                m.record_gradient(np.array([g]), 1.0, params=np.array([p]))
            return m.get_report().vanishing_detected

        assert run(1e-7, 1.0) == run(1e-3, 1e-4) is True

    @pytest.mark.parametrize("path", ["jac", "fim"])
    def test_sensitivity_directions_in_original_parameter_space(self, path):
        j = np.array([[1.0, 100.0], [0.0, 1.0]])
        an = ParameterSensitivityAnalyzer(DiagnosticsConfig())
        rep = an.analyze(j) if path == "jac" else an.analyze_from_fim(j.T @ j)
        v = rep.eigenvectors[:, -1]
        assert np.linalg.norm(v) == pytest.approx(1.0)
        np.testing.assert_allclose(np.abs(v), [0.99995, 0.0099985], atol=1e-4)
        assert np.linalg.norm(j @ v) < 0.02

    def test_reports_name_the_equilibrated_coordinates(self):
        from nlsq.diagnostics import ParameterSensitivityReport

        assert "equilibrated" in IdentifiabilityReport.__doc__
        assert "equilibrated" in ParameterSensitivityReport.__doc__

    def test_plugin_issue_missing_fields_is_dropped(self):
        from types import SimpleNamespace

        bad = SimpleNamespace(
            severity=IssueSeverity.WARNING, category="bad", code="P1", message="x"
        )
        PluginRegistry.register(_Plugin("ns", issues=[bad]))
        with pytest.warns(UserWarning, match="invalid issue"):
            results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        rep = create_health_report(plugin_results=results)
        rep.summary()
        json.dumps(rep.to_dict(), allow_nan=False)

    def test_jax_array_plugin_data_is_json_safe(self):
        res = PluginResult(
            plugin_name="x", data={"a": jnp.array([1.0, jnp.nan]), "s": jnp.float32(2)}
        )
        rep = create_health_report(
            plugin_results={"x": res}, config=DiagnosticsConfig(verbose=False)
        )
        json.dumps(rep.to_dict(), allow_nan=False)


# ------------------------------------------------------------ round 3 (Codex)
class TestRound3:
    def test_resolved_1e8_null_component_still_correlated(self):
        j = np.array([[1, 1, 0], [0, 1e-8, 1], [0, 0, 0.0]])
        rep = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze(j)
        pairs = {p[:2] for p in rep.highly_correlated_pairs}
        assert {(0, 1), (0, 2), (1, 2)} <= pairs

    def test_float32_noise_is_not_a_perfect_correlation(self):
        rng = np.random.default_rng(4)
        for _ in range(2):  # the 2nd draw triggered the false +-1 with float64 eps
            u = rng.normal(size=10).astype(np.float32)
            v = rng.normal(size=10).astype(np.float32)
            rep = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze(
                np.stack([u, u, v], axis=1)
            )
            pairs = {p[:2] for p in rep.highly_correlated_pairs}
            assert pairs == {(0, 1)}

    @staticmethod
    def _grad001(grads, params, cost=1.0):
        m = GradientMonitor(DiagnosticsConfig())
        for g, p in zip(grads, params, strict=True):
            m.record_gradient(np.array(g), cost, params=np.array(p))
        return m.get_report().vanishing_detected

    def test_grad001_dropped_test_uses_scaled_history(self):
        a = self._grad001([[1.0, 1e-8]] + [[1e-8, 1e-8]] * 10, [[1e8, 1e8]] * 11)
        b = self._grad001([[1.0, 1.0]] + [[1e-8, 1.0]] * 10, [[1e8, 1.0]] * 11)
        assert a == b

    def test_grad001_zero_parameters_do_not_fake_vanishing(self):
        assert self._grad001([[1.0]] * 10, [[0.0]] * 10) is False

    @pytest.mark.parametrize("tiny", [1e-160, 1e-320])
    def test_back_mapped_directions_survive_extreme_scales(self, tiny):
        rep = ParameterSensitivityAnalyzer(DiagnosticsConfig()).analyze(
            np.diag([1.0, tiny])
        )
        v = rep.eigenvectors
        assert np.all(np.isfinite(v))
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1.0)

    def test_bogus_severity_name_is_dropped(self):
        from types import SimpleNamespace

        good = ModelHealthIssue(
            category=IssueCategory.GRADIENT,
            severity=IssueSeverity.WARNING,
            code="P-1",
            message="m",
            affected_parameters=None,
            details={},
            recommendation="r",
        )
        bad = SimpleNamespace(
            severity=SimpleNamespace(name="BOGUS"),
            category=IssueCategory.GRADIENT,
            code="P-2",
            message="m",
            affected_parameters=None,
            details={},
            recommendation="r",
        )
        PluginRegistry.register(_Plugin("sev", issues=[good, bad]))
        with pytest.warns(UserWarning, match="invalid issue"):
            results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        assert [i.code for i in results["sev"].issues] == ["P-1"]

    def test_non_sequence_issues_marks_plugin_unavailable(self):
        PluginRegistry.register(_Plugin("strissues", issues="bad"))
        with pytest.warns(UserWarning, match="issues"):
            results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        assert results["strissues"].available is False
        assert "issues" in results["strissues"].error_message


# ------------------------------------------------------------ round 4 (Codex)
@pytest.fixture
def x64_off():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.serial
class TestRound4X64Off:
    @staticmethod
    def _dup_jacobian():
        u = np.random.default_rng(4).normal(size=(10, 2))
        return np.column_stack([u[:, 0], u[:, 0], u[:, 1]])

    def test_identifiability_rank_uses_host_float64(self, x64_off):
        j = self._dup_jacobian()
        an = IdentifiabilityAnalyzer(DiagnosticsConfig())
        assert an.analyze(j).numerical_rank == 2
        assert an.analyze_from_fim(j.T @ j).numerical_rank == 2

    def test_correlation_classification_uses_host_float64(self, x64_off):
        pairs = {
            p[:2]
            for p in IdentifiabilityAnalyzer(DiagnosticsConfig())
            .analyze(self._dup_jacobian())
            .highly_correlated_pairs
        }
        assert pairs == {(0, 1)}

    def test_sensitivity_null_eigenvalue_is_zero(self, x64_off):
        j = self._dup_jacobian()
        an = ParameterSensitivityAnalyzer(DiagnosticsConfig())
        assert an.analyze(j).eigenvalues[-1] == 0.0
        assert an.analyze_from_fim(j.T @ j).eigenvalues[-1] == 0.0


class TestRound4PluginNormalization:
    def test_accepted_issue_is_normalized_to_enums_and_warns(self):
        from types import SimpleNamespace

        issue = SimpleNamespace(
            severity=SimpleNamespace(name="CRITICAL"),
            category=SimpleNamespace(name="GRADIENT"),
            code="P-1",
            message="m",
            affected_parameters=[0],
            details={},
            recommendation="r",
        )
        PluginRegistry.register(_Plugin("ns", issues=[issue]))
        results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        got = results["ns"].issues[0]
        assert got.severity is IssueSeverity.CRITICAL
        assert got.category is IssueCategory.GRADIENT
        assert got.affected_parameters == (0,)
        with pytest.warns(UserWarning):
            rep = create_health_report(
                plugin_results=results,
                config=DiagnosticsConfig(emit_warnings=True, verbose=False),
            )
        assert rep.status == HealthStatus.CRITICAL


# ------------------------------------------------------- round 5 (Claude, float32)
def _corr_pair_jacobian(m, c, seed=0):
    u, w = np.random.default_rng(seed).normal(size=(2, m))
    return np.column_stack([u, c * u + np.sqrt(1 - c * c) * w])


class TestRound5TallFloat32:
    @pytest.mark.parametrize("m,c", [(10**5, 0.9999), (10**6, 0.99)])
    def test_tall_float32_jacobian_is_not_rank_deficient(self, m, c):
        j = _corr_pair_jacobian(m, c).astype(np.float32)
        an = IdentifiabilityAnalyzer(DiagnosticsConfig())
        rep = an.analyze(j)
        assert rep.numerical_rank == 2
        assert not any(i.code == "IDENT-001" for i in rep.issues)
        assert rep.correlation_matrix[0, 1] == pytest.approx(-c, abs=5e-3)

    def test_float32_and_float64_jacobian_agree(self):
        j = _corr_pair_jacobian(10**5, 0.99)
        an = IdentifiabilityAnalyzer(DiagnosticsConfig())
        a, b = an.analyze(j), an.analyze(j.astype(np.float32))
        assert a.numerical_rank == b.numerical_rank == 2
        assert b.condition_number == pytest.approx(a.condition_number, rel=1e-3)

    def test_qr_path_matches_full_svd(self):
        from nlsq.diagnostics.identifiability import _host_svd

        j = np.random.default_rng(3).normal(size=(500, 4)) * [1, 10, 1e-3, 5]
        s, v = _host_svd(j)
        _, s_ref, vh_ref = np.linalg.svd(j, full_matrices=False)
        np.testing.assert_allclose(s, s_ref, rtol=1e-12)
        np.testing.assert_allclose(np.abs(v.T @ vh_ref.T), np.eye(4), atol=1e-8)


@pytest.mark.serial
@pytest.mark.slow
def test_curve_fit_x64_off_tall_fit_is_not_rank_deficient(x64_off):
    x = np.linspace(5, 6, 10**6)
    y = 1.0 + 2.0 * x + np.random.default_rng(0).normal(0, 0.1, x.size)

    def model(x, p, q):
        return p + q * x

    r = curve_fit(model, x, y, p0=[0.5, 1.0], compute_diagnostics=True)
    assert r.diagnostics.identifiability.numerical_rank == 2
    assert r.diagnostics.status != HealthStatus.CRITICAL


# ------------------------------------------------------- round 6 (Codex)
class TestRound6AffectedParameters:
    @staticmethod
    def _issue(affected):
        from types import SimpleNamespace

        return SimpleNamespace(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.GRADIENT,
            code="P-1",
            message="m",
            affected_parameters=affected,
            details={},
            recommendation="r",
        )

    @pytest.mark.parametrize(
        "bad", [[object()], ["0"], [0.0], (1.5,), [True], [0, None], [np.float64(1.0)]]
    )
    def test_non_index_affected_parameters_are_dropped(self, bad):
        PluginRegistry.register(_Plugin("ap", issues=[self._issue(bad)]))
        with pytest.warns(UserWarning, match="invalid issue"):
            results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        assert results["ap"].issues == []

    def test_numpy_integers_are_normalized_and_json_safe(self):
        PluginRegistry.register(
            _Plugin("ap", issues=[self._issue([np.int64(0), np.int32(2), 1])])
        )
        results = run_plugins(np.eye(2), np.ones(2), np.ones(2))
        got = results["ap"].issues[0].affected_parameters
        assert got == (0, 2, 1)
        assert all(type(i) is int for i in got)
        rep = create_health_report(
            plugin_results=results, config=DiagnosticsConfig(verbose=False)
        )
        json.dumps(rep.to_dict(), allow_nan=False)


# ------------------------------------------------------- round 7 (Claude, float32 FIM)
class TestRound7Float32FimValidation:
    @staticmethod
    def _validate(fim):
        from nlsq.diagnostics.types import _validate_fim

        return _validate_fim(fim)

    def test_rank_deficient_float32_fims_are_accepted(self):
        rng = np.random.default_rng(0)
        bad = 0
        for _ in range(300):
            j = rng.normal(size=(30, 4)).astype(np.float32)
            j[:, 3] = j[:, 0] + j[:, 1]  # exactly rank deficient
            if self._validate(j.T @ j) is not None:
                bad += 1
        assert bad == 0

    def test_full_rank_float32_fims_are_accepted(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            j = rng.normal(size=(30, 4)).astype(np.float32)
            assert self._validate(j.T @ j) is None

    def test_indefinite_float32_fims_are_still_rejected(self):
        rng = np.random.default_rng(2)
        for _ in range(300):
            q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
            f = (q @ np.diag([3.0, 2.0, 1.0, -0.5]) @ q.T).astype(np.float32)
            f = (f + f.T) / 2
            assert self._validate(f) == "FIM must be positive semi-definite"

    def test_float64_tolerance_unchanged(self):
        f = np.diag([1.0, 1.0, -1e-9])  # beyond float64 round-off: rejected
        assert self._validate(f) == "FIM must be positive semi-definite"
        assert self._validate(np.diag([1.0, 1.0, -1e-17])) is None


# ------------------------------------------------------- round 8 (complex input)
class TestRound8ComplexInput:
    @staticmethod
    def _j():
        rng = np.random.default_rng(0)
        return rng.normal(size=(20, 3)) + 1j * rng.normal(size=(20, 3))

    def test_identifiability_rejects_complex_jacobian(self):
        rep = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze(self._j())
        assert rep.available is False
        assert "complex" in rep.error_message.lower()

    def test_sensitivity_rejects_complex_jacobian(self):
        rep = ParameterSensitivityAnalyzer(DiagnosticsConfig()).analyze(self._j())
        assert rep.available is False
        assert "complex" in rep.error_message.lower()

    def test_both_reject_complex_fim(self):
        j = self._j()
        fim = j.conj().T @ j
        for rep in (
            IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze_from_fim(fim),
            ParameterSensitivityAnalyzer(DiagnosticsConfig()).analyze_from_fim(fim),
        ):
            assert rep.available is False
            assert "complex" in rep.error_message.lower()


# ------------------------------------------------------- round 9 (float16 FIM)
class TestRound9Float16Fim:
    def test_rounded_rank_deficient_float16_fim_is_accepted(self):
        j = np.array([[1.0, 0.022882882882882885]])
        fim = (j.T @ j).astype(np.float16)
        assert np.linalg.eigvalsh(fim.astype(np.float64))[0] < 0  # the premise
        for rep in (
            IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze_from_fim(fim),
            ParameterSensitivityAnalyzer(DiagnosticsConfig()).analyze_from_fim(fim),
        ):
            assert rep.available is True

    def test_indefinite_float16_fim_is_still_rejected(self):
        fim = np.diag([3.0, 2.0, 1.0, -0.5]).astype(np.float16)
        for rep in (
            IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze_from_fim(fim),
            ParameterSensitivityAnalyzer(DiagnosticsConfig()).analyze_from_fim(fim),
        ):
            assert rep.available is False
            assert "positive semi-definite" in rep.error_message


def test_longdouble_fim_tolerance_matches_float64_eigensolve():
    from nlsq.diagnostics.types import _validate_fim

    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        pytest.skip("longdouble is not wider than float64 on this platform")
    # Eigenvalues are computed in float64, so a -5e-17 entry is round-off there.
    fim = np.diag(np.array([1.0, 1.0, -5e-17], dtype=np.longdouble))
    assert _validate_fim(fim) is None
    assert _validate_fim(np.diag(np.array([1.0, 1.0, -1e-9], dtype=np.longdouble)))


# ------------------------------------------------------- round 10 (eigh for FIMs)
class TestRound10FimUsesEigenvalues:
    @staticmethod
    def _fim():
        j = np.array([[1.0, 0.022882882882882885]])
        return (j.T @ j).astype(np.float16)

    def test_negative_rounding_eigenvalue_is_zero_not_positive(self):
        fim = self._fim()
        ident = IdentifiabilityAnalyzer(DiagnosticsConfig()).analyze_from_fim(fim)
        assert ident.numerical_rank == 1
        sens = ParameterSensitivityAnalyzer(DiagnosticsConfig()).analyze_from_fim(fim)
        assert sens.eigenvalues[0] > 0
        assert sens.eigenvalues[1] == 0.0

    def test_sensitivity_fim_matches_jacobian_path(self):
        rng = np.random.default_rng(5)
        j = rng.normal(size=(40, 4)) * [1, 10, 0.1, 3]
        an = ParameterSensitivityAnalyzer(DiagnosticsConfig())
        a, b = an.analyze(j), an.analyze_from_fim(j.T @ j)
        np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, rtol=1e-8)
        np.testing.assert_allclose(
            np.abs(a.eigenvectors.T @ b.eigenvectors).max(axis=0), 1.0, atol=1e-6
        )


# ------------------------------------------------- round 11 (float32 FIM storage eps)
class TestRound11Float32FimStorageEps:
    @staticmethod
    def _cases(n_cases=60):
        rng = np.random.default_rng(3)
        for _ in range(n_cases):
            c64 = rng.standard_normal((400, 5)) * 10.0 ** rng.uniform(-2, 2, 5)
            c64[:, -1] = c64[:, 0] - 0.5 * c64[:, 1]  # exact dependency
            yield c64

    def test_float32_fim_rank_and_correlation_match_truth(self):
        an = IdentifiabilityAnalyzer(DiagnosticsConfig())
        for c64 in self._cases():
            truth = an.analyze(c64).correlation_matrix
            c = c64.astype(np.float32)
            rep = an.analyze_from_fim(c.T @ c)
            assert rep.numerical_rank == 4
            assert np.abs(rep.correlation_matrix - truth).max() < 1e-4

    def test_float32_jacobian_equals_float64_upcast_result(self):
        an = IdentifiabilityAnalyzer(DiagnosticsConfig())
        for c64 in self._cases(20):
            c = c64.astype(np.float32)
            a, b = an.analyze(c), an.analyze(c.astype(np.float64))
            assert a.numerical_rank == b.numerical_rank
            assert a.health_status == b.health_status

    def test_sensitivity_zeroes_float32_null_eigenvalue(self):
        an = ParameterSensitivityAnalyzer(DiagnosticsConfig())
        for c64 in self._cases(30):
            c = c64.astype(np.float32)
            assert an.analyze_from_fim(c.T @ c).eigenvalues[-1] == 0.0


# --------------------------------------------- round 12 (Jacobians are analyzed as stored)
class TestRound12JacobianAsStored:
    @staticmethod
    def _j(dtype):
        return np.array([[1.0, 1.0], [0.0, 1e-7]], dtype=dtype)

    def test_identifiability_float32_jacobian_equals_float64_result(self):
        an = IdentifiabilityAnalyzer(DiagnosticsConfig())
        r32, r64 = an.analyze(self._j(np.float32)), an.analyze(self._j(np.float64))
        assert r32.numerical_rank == r64.numerical_rank == 2
        assert not any(i.code == "IDENT-001" for i in r32.issues)

    def test_sensitivity_float32_jacobian_equals_float64_result(self):
        an = ParameterSensitivityAnalyzer(DiagnosticsConfig())
        e32 = an.analyze(self._j(np.float32)).eigenvalues
        e64 = an.analyze(self._j(np.float64)).eigenvalues
        assert e32[-1] > 0
        np.testing.assert_allclose(e32, e64, rtol=1e-6)
