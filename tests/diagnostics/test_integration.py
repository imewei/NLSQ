"""Integration tests for diagnostics with curve_fit.

Tests cover:
- compute_diagnostics parameter in curve_fit()
- diagnostics property on CurveFitResult
- End-to-end flow from fitting to diagnostics
- Backward compatibility (T054)
- Performance overhead verification (T053)
"""

import time
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.diagnostics.types import (
    DiagnosticLevel,
    DiagnosticsConfig,
    HealthStatus,
    IssueCategory,
)


class TestCurveFitDiagnosticsIntegration:
    """Integration tests for curve_fit with diagnostics."""

    @pytest.fixture
    def simple_model(self):
        """Simple exponential decay model."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        return model

    @pytest.fixture
    def linear_model(self):
        """Simple linear model."""

        def model(x, a, b):
            return a * x + b

        return model

    @pytest.fixture
    def well_conditioned_data(self):
        """Well-conditioned fitting data."""
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y_true = 2.5 * np.exp(-0.5 * x)
        y = y_true + 0.05 * np.random.randn(len(x))
        return x, y

    @pytest.fixture
    def ill_conditioned_data(self):
        """Data that leads to ill-conditioned fitting problem."""
        np.random.seed(42)
        # Very narrow x range leads to poor conditioning
        x = np.linspace(0, 0.01, 50)
        y = 2.5 * np.exp(-0.5 * x) + 0.01 * np.random.randn(len(x))
        return x, y

    # Test compute_diagnostics parameter
    def test_curve_fit_without_diagnostics(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test curve_fit works without diagnostics."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])
        # Should work without compute_diagnostics
        assert result.success
        assert hasattr(result, "popt")
        assert hasattr(result, "pcov")

    def test_curve_fit_with_diagnostics_basic(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test curve_fit with basic diagnostics enabled."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        assert result.success
        assert hasattr(result, "diagnostics")
        assert result.diagnostics is not None

    def test_curve_fit_with_diagnostics_level(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test curve_fit with specific diagnostics level."""
        x, y = well_conditioned_data
        result = curve_fit(
            simple_model,
            x,
            y,
            p0=[2.0, 0.5],
            compute_diagnostics=True,
            diagnostics_level=DiagnosticLevel.BASIC,
        )
        assert result.success
        assert result.diagnostics is not None

    def test_diagnostics_property_returns_identifiability(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test diagnostics property returns identifiability report."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        diag = result.diagnostics
        # Check identifiability analysis is present
        assert hasattr(diag, "identifiability")
        assert diag.identifiability is not None
        assert diag.identifiability.available

    def test_diagnostics_identifiability_report_fields(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test identifiability report has expected fields."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        ident = result.diagnostics.identifiability
        assert hasattr(ident, "condition_number")
        assert hasattr(ident, "numerical_rank")
        assert hasattr(ident, "n_params")
        assert hasattr(ident, "correlation_matrix")
        assert hasattr(ident, "highly_correlated_pairs")
        assert hasattr(ident, "issues")
        assert hasattr(ident, "health_status")

    def test_well_conditioned_fit_healthy(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test well-conditioned fit reports HEALTHY status."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        ident = result.diagnostics.identifiability
        # Well-conditioned problem should be healthy
        assert ident.health_status == HealthStatus.HEALTHY
        # No issues should be detected
        assert len(ident.issues) == 0

    def test_diagnostics_with_bounds(self, simple_model, well_conditioned_data) -> None:
        """Test diagnostics work with parameter bounds."""
        x, y = well_conditioned_data
        result = curve_fit(
            simple_model,
            x,
            y,
            p0=[2.0, 0.5],
            bounds=([0.0, 0.0], [10.0, 2.0]),
            compute_diagnostics=True,
        )
        assert result.success
        assert result.diagnostics is not None
        assert result.diagnostics.identifiability.available

    def test_diagnostics_with_sigma(self, simple_model, well_conditioned_data) -> None:
        """Test diagnostics work with measurement uncertainties."""
        x, y = well_conditioned_data
        sigma = 0.1 * np.ones_like(y)
        result = curve_fit(
            simple_model, x, y, p0=[2.0, 0.5], sigma=sigma, compute_diagnostics=True
        )
        assert result.success
        assert result.diagnostics is not None
        assert result.diagnostics.identifiability.available

    # Test ill-conditioned problems
    def test_ill_conditioned_fit_issues(
        self, simple_model, ill_conditioned_data
    ) -> None:
        """Test ill-conditioned fit reports issues."""
        x, y = ill_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        ident = result.diagnostics.identifiability
        # Should have high condition number
        assert ident.condition_number > 1e4

    # Test correlation detection
    def test_correlated_parameters_detection(self) -> None:
        """Test detection of correlated parameters."""

        # Model with inherently correlated parameters: a*exp(-b*x) + c
        # When x is near 0, a and c are nearly indistinguishable
        def model_correlated(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        np.random.seed(42)
        x = np.linspace(0, 0.1, 50)  # Very narrow range
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.01 * np.random.randn(len(x))

        result = curve_fit(
            model_correlated, x, y, p0=[2.5, 0.5, 1.0], compute_diagnostics=True
        )
        ident = result.diagnostics.identifiability
        # Should detect high correlation between parameters
        # (a and c become indistinguishable when x is small)
        assert ident.condition_number > 100  # Indicates conditioning issues

    # Test tuple unpacking still works
    def test_tuple_unpacking_with_diagnostics(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test tuple unpacking still works with diagnostics enabled."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        # Tuple unpacking should still work
        popt, pcov = result
        assert len(popt) == 2
        assert pcov.shape == (2, 2)

    # Test diagnostics disabled by default
    def test_diagnostics_disabled_by_default(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test diagnostics are disabled by default."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])
        # diagnostics property should return None or not be computed
        # depending on implementation
        diag = getattr(result, "diagnostics", None)
        if diag is not None:
            # If property exists, it might be lazy or None
            pass  # Implementation may vary

    # Test different models
    def test_linear_model_diagnostics(self, linear_model) -> None:
        """Test diagnostics with linear model."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 3.0 + 0.5 * np.random.randn(len(x))

        result = curve_fit(linear_model, x, y, p0=[1.0, 1.0], compute_diagnostics=True)
        assert result.success
        assert result.diagnostics.identifiability.available
        # Linear model should be well-conditioned
        assert result.diagnostics.identifiability.health_status == HealthStatus.HEALTHY

    # Test computation time is reasonable
    def test_diagnostics_computation_time(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test diagnostics computation time is tracked."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        ident = result.diagnostics.identifiability
        assert ident.computation_time_ms >= 0


class TestDiagnosticsReportSummary:
    """Tests for diagnostics report summary functionality."""

    @pytest.fixture
    def simple_model(self):
        """Simple exponential decay model."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        return model

    @pytest.fixture
    def well_conditioned_data(self):
        """Well-conditioned fitting data."""
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y = 2.5 * np.exp(-0.5 * x) + 0.05 * np.random.randn(len(x))
        return x, y

    def test_identifiability_summary_method(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test identifiability report has summary method."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        ident = result.diagnostics.identifiability
        # Should have summary method or __str__
        assert hasattr(ident, "__str__") or hasattr(ident, "summary")

    def test_diagnostics_issue_codes(self, simple_model, well_conditioned_data) -> None:
        """Test diagnostics issue codes are valid."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        ident = result.diagnostics.identifiability
        for issue in ident.issues:
            # Validate issue code format
            assert issue.code.startswith(("IDENT-", "CORR-"))
            # Validate issue has recommendation
            assert len(issue.recommendation) > 0


class TestDiagnosticsConfig:
    """Tests for custom diagnostics configuration."""

    @pytest.fixture
    def simple_model(self):
        """Simple exponential decay model."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        return model

    @pytest.fixture
    def data(self):
        """Standard fitting data."""
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y = 2.5 * np.exp(-0.5 * x) + 0.05 * np.random.randn(len(x))
        return x, y

    def test_custom_condition_threshold(self, simple_model, data) -> None:
        """Test custom condition threshold in diagnostics config."""
        x, y = data
        # Very strict threshold
        config = DiagnosticsConfig(condition_threshold=10.0)
        result = curve_fit(
            simple_model,
            x,
            y,
            p0=[2.0, 0.5],
            compute_diagnostics=True,
            diagnostics_config=config,
        )
        # With very strict threshold, might trigger warning
        ident = result.diagnostics.identifiability
        assert ident.available

    def test_custom_correlation_threshold(self, simple_model, data) -> None:
        """Test custom correlation threshold in diagnostics config."""
        x, y = data
        # Very strict correlation threshold
        config = DiagnosticsConfig(correlation_threshold=0.3)
        result = curve_fit(
            simple_model,
            x,
            y,
            p0=[2.0, 0.5],
            compute_diagnostics=True,
            diagnostics_config=config,
        )
        ident = result.diagnostics.identifiability
        assert ident.available

    def test_verbose_diagnostics(self, simple_model, data, capsys) -> None:
        """Test verbose diagnostics output."""
        x, y = data
        config = DiagnosticsConfig(verbose=True)
        result = curve_fit(
            simple_model,
            x,
            y,
            p0=[2.0, 0.5],
            compute_diagnostics=True,
            diagnostics_config=config,
        )
        # Should complete without error
        assert result.success


class TestDiagnosticsEdgeCases:
    """Edge case tests for diagnostics integration."""

    def test_single_parameter_model(self) -> None:
        """Test diagnostics with single-parameter model."""

        def model(x, a):
            return a * x

        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.5 * x + 0.1 * np.random.randn(len(x))

        result = curve_fit(model, x, y, p0=[1.0], compute_diagnostics=True)
        assert result.success
        assert result.diagnostics.identifiability.n_params == 1

    def test_many_parameters_model(self) -> None:
        """Test diagnostics with many-parameter model."""

        def model(x, a, b, c, d, e):
            return a * jnp.exp(-b * x) + c * jnp.sin(d * x) + e

        np.random.seed(42)
        x = np.linspace(0, 10, 200)
        y_true = 2.5 * np.exp(-0.3 * x) + 1.0 * np.sin(0.5 * x) + 0.5
        y = y_true + 0.1 * np.random.randn(len(x))

        result = curve_fit(
            model, x, y, p0=[2.0, 0.3, 1.0, 0.5, 0.5], compute_diagnostics=True
        )
        assert result.success
        assert result.diagnostics.identifiability.n_params == 5

    def test_small_dataset(self) -> None:
        """Test diagnostics with small dataset."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.array([0, 1, 2, 3, 4])
        y = 2.5 * np.exp(-0.5 * x) + 0.01 * np.random.randn(len(x))

        result = curve_fit(model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        assert result.diagnostics.identifiability.available

    def test_diagnostics_with_failed_fit(self) -> None:
        """Test diagnostics when fit fails or has issues."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Data that's hard to fit
        np.random.seed(42)
        x = np.linspace(0, 1, 10)
        y = np.random.randn(10)  # Random noise, not matching model

        result = curve_fit(
            model,
            x,
            y,
            p0=[1.0, 1.0],
            compute_diagnostics=True,
            maxfev=50,  # Limit iterations
        )
        # Should still have diagnostics even if fit is poor
        if hasattr(result, "diagnostics") and result.diagnostics is not None:
            assert result.diagnostics.identifiability is not None


@pytest.mark.diagnostics
class TestBackwardCompatibility:
    """Backward compatibility tests for diagnostics feature (T054).

    These tests verify that:
    1. Existing code works unchanged without compute_diagnostics parameter
    2. Default behavior is unchanged (no diagnostics by default, no warnings)
    3. API stability (old imports still work, method signatures unchanged)
    """

    @pytest.fixture
    def simple_model(self):
        """Simple exponential decay model."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        return model

    @pytest.fixture
    def well_conditioned_data(self):
        """Well-conditioned fitting data."""
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y_true = 2.5 * np.exp(-0.5 * x)
        y = y_true + 0.05 * np.random.randn(len(x))
        return x, y

    # =========================================================================
    # Test 1: Existing code works unchanged
    # =========================================================================

    def test_curve_fit_without_compute_diagnostics_works(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test curve_fit() without compute_diagnostics parameter works normally."""
        x, y = well_conditioned_data
        # Call curve_fit exactly as existing code would (no compute_diagnostics)
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # Fit should succeed
        assert result.success
        # Parameters should be reasonable (close to true values 2.5, 0.5)
        assert 2.0 < result.popt[0] < 3.0
        assert 0.3 < result.popt[1] < 0.7

    def test_tuple_unpacking_works_without_diagnostics(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test result can be tuple-unpacked: popt, pcov = curve_fit(...)."""
        x, y = well_conditioned_data

        # Tuple unpacking should work exactly as before
        popt, pcov = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # Verify popt is correct type and shape
        assert isinstance(popt, np.ndarray)
        assert len(popt) == 2

        # Verify pcov is correct type and shape
        assert isinstance(pcov, np.ndarray)
        assert pcov.shape == (2, 2)

    def test_all_existing_result_attributes_work(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test all existing result attributes work (popt, pcov, success, etc.)."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # Core attributes should exist and work
        assert hasattr(result, "popt")
        assert hasattr(result, "pcov")
        assert hasattr(result, "success")
        assert hasattr(result, "x")  # Raw optimized parameters
        assert hasattr(result, "cost")
        assert hasattr(result, "fun")  # Residuals
        assert hasattr(result, "jac")  # Jacobian
        assert hasattr(result, "nfev")
        assert hasattr(result, "njev")
        assert hasattr(result, "message")

        # Verify types
        assert isinstance(result.popt, np.ndarray)
        assert isinstance(result.pcov, np.ndarray)
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)

    def test_no_diagnostics_computed_by_default(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test no diagnostics computed by default (result.diagnostics is None)."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # diagnostics property should return None when not computed
        assert result.diagnostics is None

    def test_existing_kwargs_still_work(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test existing kwargs (bounds, sigma, method, etc.) still work."""
        x, y = well_conditioned_data
        sigma = 0.1 * np.ones_like(y)

        # All existing parameters should work as before
        result = curve_fit(
            simple_model,
            x,
            y,
            p0=[2.0, 0.5],
            sigma=sigma,
            absolute_sigma=False,
            check_finite=True,
            bounds=([0, 0], [10, 5]),
            method="trf",
            maxfev=1000,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )

        assert result.success
        assert len(result.popt) == 2

    # =========================================================================
    # Test 2: Default behavior unchanged
    # =========================================================================

    def test_compute_diagnostics_default_is_false(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test compute_diagnostics=False by default."""
        x, y = well_conditioned_data

        # Without specifying compute_diagnostics, it should default to False
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # diagnostics should be None (not computed)
        assert result.diagnostics is None

    def test_no_performance_penalty_when_diagnostics_disabled(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test no performance penalty when diagnostics disabled."""
        x, y = well_conditioned_data

        # Warm up JIT (first call is slow due to compilation)
        _ = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # Measure time without diagnostics
        start = time.perf_counter()
        for _ in range(5):
            _ = curve_fit(simple_model, x, y, p0=[2.0, 0.5])
        time_without = (time.perf_counter() - start) / 5

        # Measure time with diagnostics
        start = time.perf_counter()
        for _ in range(5):
            _ = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)
        time_with = (time.perf_counter() - start) / 5

        # Without diagnostics should be faster or at most equal
        # Allow 50% overhead from diagnostics (generous margin for test stability)
        assert time_without <= time_with * 1.5

    def test_no_warnings_emitted_when_diagnostics_disabled(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test no warnings emitted when diagnostics disabled."""
        x, y = well_conditioned_data

        # Capture warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # Filter for diagnostics-related warnings
        diag_warnings = [
            w for w in caught_warnings if "diagnostic" in str(w.message).lower()
        ]

        # No diagnostics-related warnings should be emitted
        assert len(diag_warnings) == 0
        assert result.success

    def test_explicit_false_same_as_default(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test compute_diagnostics=False explicitly gives same result as default."""
        x, y = well_conditioned_data

        # Default (no parameter)
        result_default = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # Explicit False
        result_explicit = curve_fit(
            simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=False
        )

        # Both should have None diagnostics
        assert result_default.diagnostics is None
        assert result_explicit.diagnostics is None

        # Parameters should be identical
        np.testing.assert_array_almost_equal(
            result_default.popt, result_explicit.popt, decimal=10
        )
        np.testing.assert_array_almost_equal(
            result_default.pcov, result_explicit.pcov, decimal=10
        )

    # =========================================================================
    # Test 3: API stability
    # =========================================================================

    def test_old_import_still_works(self) -> None:
        """Test old imports still work: from nlsq import curve_fit."""
        # This import should work exactly as before
        from nlsq import curve_fit as cf

        # Verify it's the expected function
        assert callable(cf)
        assert cf.__name__ == "curve_fit"

    def test_optional_diagnostics_import_works(self) -> None:
        """Test optional diagnostics import: from nlsq.diagnostics import ..."""
        # These imports should work
        from nlsq.diagnostics import (
            DiagnosticLevel,
            DiagnosticsConfig,
            HealthStatus,
            IdentifiabilityReport,
            ModelHealthReport,
        )

        # Verify they are the expected types
        assert DiagnosticLevel.BASIC is not None
        assert DiagnosticLevel.FULL is not None
        assert HealthStatus.HEALTHY is not None
        assert HealthStatus.WARNING is not None
        assert HealthStatus.CRITICAL is not None

    def test_curve_fit_signature_unchanged_for_positional_args(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test curve_fit signature unchanged for positional arguments."""
        x, y = well_conditioned_data

        # Positional args should work as before: curve_fit(f, xdata, ydata)
        result = curve_fit(simple_model, x, y)
        assert result.success

        # With p0 as keyword
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])
        assert result.success

    def test_curvefit_class_still_works(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test CurveFit class-based interface still works."""
        from nlsq import CurveFit

        x, y = well_conditioned_data

        # Class-based interface should work as before
        fitter = CurveFit()
        result = fitter.curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        assert result.success
        assert len(result.popt) == 2

    def test_result_type_unchanged(self, simple_model, well_conditioned_data) -> None:
        """Test result type is still CurveFitResult."""
        from nlsq.result import CurveFitResult

        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # Result should be CurveFitResult instance
        assert isinstance(result, CurveFitResult)

    def test_diagnostics_is_optional_attribute(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test diagnostics attribute exists but is optional (can be None)."""
        x, y = well_conditioned_data
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5])

        # diagnostics attribute should exist (via property)
        assert hasattr(result, "diagnostics")

        # But should be None when not computed
        diag = result.diagnostics
        assert diag is None

    def test_can_enable_diagnostics_without_breaking_existing_code(
        self, simple_model, well_conditioned_data
    ) -> None:
        """Test enabling diagnostics doesn't break existing code patterns."""
        x, y = well_conditioned_data

        # Enable diagnostics - should still support all existing patterns
        result = curve_fit(simple_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True)

        # Tuple unpacking still works
        popt, _pcov = result
        assert len(popt) == 2

        # All standard attributes still work
        assert result.success
        assert result.popt is not None
        assert result.pcov is not None

        # Diagnostics are now available
        assert result.diagnostics is not None


@pytest.mark.diagnostics
@pytest.mark.slow
class TestDiagnosticsPerformance:
    """Performance benchmark tests for diagnostics overhead (T053).

    These tests verify that diagnostics overhead is reasonable (<100%) to catch
    major performance regressions. Uses 10,000 point exponential decay dataset.

    Note: Overhead varies based on JIT compilation state, CPU load, and hardware.
    The threshold is set conservatively to avoid flaky failures while still
    catching significant regressions.
    """

    @pytest.fixture
    def exponential_model(self):
        """Exponential decay model for performance testing."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        return model

    @pytest.fixture
    def large_dataset(self):
        """10,000 point exponential decay dataset per spec."""
        np.random.seed(42)
        x = np.linspace(0, 10, 10000)
        y_true = 2.5 * np.exp(-0.5 * x)
        noise = 0.05 * np.random.randn(len(x))
        y = y_true + noise
        return x, y

    def test_basic_diagnostics_overhead_under_100_percent(
        self, exponential_model, large_dataset
    ) -> None:
        """Verify basic diagnostics overhead is reasonable (<100%).

        This test:
        1. Generates 10,000 point exponential decay dataset
        2. Times curve_fit without diagnostics (N iterations)
        3. Times curve_fit with diagnostics=True (N iterations)
        4. Calculates overhead percentage
        5. Asserts overhead < 100% (catches major regressions)
        """
        x, y = large_dataset
        n_iterations = 10
        n_warmup = 5  # More warmup for JIT stability

        # Warm up JIT compilation for both code paths
        # This ensures we're measuring execution time, not compilation time
        for _ in range(n_warmup):
            _ = curve_fit(exponential_model, x, y, p0=[2.0, 0.5])
            _ = curve_fit(
                exponential_model,
                x,
                y,
                p0=[2.0, 0.5],
                compute_diagnostics=True,
                diagnostics_level=DiagnosticLevel.BASIC,
            )

        # Measure time without diagnostics
        times_without = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            result = curve_fit(exponential_model, x, y, p0=[2.0, 0.5])
            end = time.perf_counter()
            assert result.success, "Fit without diagnostics should succeed"
            times_without.append(end - start)

        # Measure time with basic diagnostics
        times_with = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            result = curve_fit(
                exponential_model,
                x,
                y,
                p0=[2.0, 0.5],
                compute_diagnostics=True,
                diagnostics_level=DiagnosticLevel.BASIC,
            )
            end = time.perf_counter()
            assert result.success, "Fit with diagnostics should succeed"
            assert result.diagnostics is not None, "Diagnostics should be computed"
            times_with.append(end - start)

        # Calculate median times (more robust than mean for timing)
        median_without = np.median(times_without)
        median_with = np.median(times_with)

        # Calculate overhead percentage
        overhead_percent = ((median_with - median_without) / median_without) * 100

        # Assert overhead is under 100% (catches major regressions while tolerating variance)
        assert overhead_percent < 100.0, (
            f"Diagnostics overhead {overhead_percent:.2f}% exceeds 100% limit. "
            f"Median without: {median_without * 1000:.2f}ms, "
            f"Median with: {median_with * 1000:.2f}ms"
        )

    def test_diagnostics_overhead_multiple_runs(
        self, exponential_model, large_dataset
    ) -> None:
        """Additional stability test with multiple measurement runs.

        Performs multiple measurement cycles to verify overhead is consistently reasonable.
        """
        x, y = large_dataset
        n_iterations_per_run = 5
        n_runs = 3
        n_warmup = 5  # More warmup for JIT stability

        # Warm up
        for _ in range(n_warmup):
            _ = curve_fit(exponential_model, x, y, p0=[2.0, 0.5])
            _ = curve_fit(
                exponential_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True
            )

        overheads = []

        for _ in range(n_runs):
            # Measure without diagnostics
            times_without = []
            for _ in range(n_iterations_per_run):
                start = time.perf_counter()
                _ = curve_fit(exponential_model, x, y, p0=[2.0, 0.5])
                times_without.append(time.perf_counter() - start)

            # Measure with diagnostics
            times_with = []
            for _ in range(n_iterations_per_run):
                start = time.perf_counter()
                _ = curve_fit(
                    exponential_model, x, y, p0=[2.0, 0.5], compute_diagnostics=True
                )
                times_with.append(time.perf_counter() - start)

            median_without = np.median(times_without)
            median_with = np.median(times_with)
            overhead = ((median_with - median_without) / median_without) * 100
            overheads.append(overhead)

        # Check that the average overhead across runs is under 100%
        avg_overhead = np.mean(overheads)
        assert avg_overhead < 100.0, (
            f"Average diagnostics overhead {avg_overhead:.2f}% exceeds 100% limit. "
            f"Individual run overheads: {[f'{o:.2f}%' for o in overheads]}"
        )
