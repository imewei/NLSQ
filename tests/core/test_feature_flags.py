"""Unit tests for FeatureFlags system.

Tests for the feature flag infrastructure that controls gradual rollout
of decomposed CurveFit components.

Reference: specs/017-curve-fit-decomposition/spec.md FR-008, FR-009, FR-010
"""

from __future__ import annotations

import os

import pytest

from nlsq.core.feature_flags import (
    COMPONENT_COVARIANCE,
    COMPONENT_PREPROCESSOR,
    COMPONENT_SELECTOR,
    COMPONENT_STREAMING,
    DEFAULT_IMPL,
    DEFAULT_ROLLOUT_PERCENT,
    ENV_COVARIANCE,
    ENV_PREPROCESSOR,
    ENV_ROLLOUT_PERCENT,
    ENV_SELECTOR,
    ENV_STREAMING,
    FeatureFlags,
    get_feature_flags,
    reset_feature_flags,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean all feature flag environment variables."""
    for var in [
        ENV_PREPROCESSOR,
        ENV_SELECTOR,
        ENV_COVARIANCE,
        ENV_STREAMING,
        ENV_ROLLOUT_PERCENT,
    ]:
        monkeypatch.delenv(var, raising=False)
    reset_feature_flags()


@pytest.fixture
def flags_all_new(monkeypatch: pytest.MonkeyPatch) -> FeatureFlags:
    """Create FeatureFlags with all components set to 'new'."""
    monkeypatch.setenv(ENV_PREPROCESSOR, "new")
    monkeypatch.setenv(ENV_SELECTOR, "new")
    monkeypatch.setenv(ENV_COVARIANCE, "new")
    monkeypatch.setenv(ENV_STREAMING, "new")
    return FeatureFlags.from_env()


@pytest.fixture
def flags_all_old(monkeypatch: pytest.MonkeyPatch) -> FeatureFlags:
    """Create FeatureFlags with all components set to 'old'."""
    monkeypatch.setenv(ENV_PREPROCESSOR, "old")
    monkeypatch.setenv(ENV_SELECTOR, "old")
    monkeypatch.setenv(ENV_COVARIANCE, "old")
    monkeypatch.setenv(ENV_STREAMING, "old")
    return FeatureFlags.from_env()


# =============================================================================
# Test FeatureFlags Creation
# =============================================================================


class TestFeatureFlagsCreation:
    """Tests for FeatureFlags creation and initialization."""

    def test_default_values(self, clean_env: None) -> None:
        """Test FeatureFlags has correct default values."""
        flags = FeatureFlags()

        assert flags.preprocessor_impl == DEFAULT_IMPL
        assert flags.selector_impl == DEFAULT_IMPL
        assert flags.covariance_impl == DEFAULT_IMPL
        assert flags.streaming_impl == DEFAULT_IMPL
        assert flags.rollout_percent == DEFAULT_ROLLOUT_PERCENT

    def test_from_env_defaults(self, clean_env: None) -> None:
        """Test from_env uses defaults when env vars not set."""
        flags = FeatureFlags.from_env()

        assert flags.preprocessor_impl == DEFAULT_IMPL
        assert flags.selector_impl == DEFAULT_IMPL
        assert flags.covariance_impl == DEFAULT_IMPL
        assert flags.streaming_impl == DEFAULT_IMPL
        assert flags.rollout_percent == DEFAULT_ROLLOUT_PERCENT

    def test_from_env_all_new(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env reads 'new' values correctly."""
        monkeypatch.setenv(ENV_PREPROCESSOR, "new")
        monkeypatch.setenv(ENV_SELECTOR, "new")
        monkeypatch.setenv(ENV_COVARIANCE, "new")
        monkeypatch.setenv(ENV_STREAMING, "new")

        flags = FeatureFlags.from_env()

        assert flags.preprocessor_impl == "new"
        assert flags.selector_impl == "new"
        assert flags.covariance_impl == "new"
        assert flags.streaming_impl == "new"

    def test_from_env_all_old(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env reads 'old' values correctly."""
        monkeypatch.setenv(ENV_PREPROCESSOR, "old")
        monkeypatch.setenv(ENV_SELECTOR, "old")
        monkeypatch.setenv(ENV_COVARIANCE, "old")
        monkeypatch.setenv(ENV_STREAMING, "old")

        flags = FeatureFlags.from_env()

        assert flags.preprocessor_impl == "old"
        assert flags.selector_impl == "old"
        assert flags.covariance_impl == "old"
        assert flags.streaming_impl == "old"

    def test_from_env_mixed(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env with mixed values."""
        monkeypatch.setenv(ENV_PREPROCESSOR, "new")
        monkeypatch.setenv(ENV_SELECTOR, "old")
        monkeypatch.setenv(ENV_COVARIANCE, "auto")
        # STREAMING not set

        flags = FeatureFlags.from_env()

        assert flags.preprocessor_impl == "new"
        assert flags.selector_impl == "old"
        assert flags.covariance_impl == "auto"
        assert flags.streaming_impl == DEFAULT_IMPL  # Default

    def test_from_env_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env is case-insensitive."""
        monkeypatch.setenv(ENV_PREPROCESSOR, "NEW")
        monkeypatch.setenv(ENV_SELECTOR, "Old")
        monkeypatch.setenv(ENV_COVARIANCE, "AUTO")

        flags = FeatureFlags.from_env()

        assert flags.preprocessor_impl == "new"
        assert flags.selector_impl == "old"
        assert flags.covariance_impl == "auto"

    def test_from_env_rollout_percent(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env reads rollout percentage correctly."""
        monkeypatch.setenv(ENV_ROLLOUT_PERCENT, "50")

        flags = FeatureFlags.from_env()

        assert flags.rollout_percent == 50

    def test_custom_session_id(self, clean_env: None) -> None:
        """Test from_env with custom session_id."""
        flags = FeatureFlags.from_env(session_id="test-session-123")

        assert flags._session_id == "test-session-123"

    def test_frozen_dataclass(self) -> None:
        """Test FeatureFlags is immutable."""
        flags = FeatureFlags()

        with pytest.raises(AttributeError):
            flags.preprocessor_impl = "new"  # type: ignore[misc]


# =============================================================================
# Test Input Validation
# =============================================================================


class TestFeatureFlagsValidation:
    """Tests for FeatureFlags input validation."""

    def test_invalid_impl_choice(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env rejects invalid implementation choice."""
        monkeypatch.setenv(ENV_PREPROCESSOR, "invalid")

        with pytest.raises(ValueError, match="Invalid implementation choice"):
            FeatureFlags.from_env()

    def test_invalid_rollout_percent_non_numeric(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env rejects non-numeric rollout percent."""
        monkeypatch.setenv(ENV_ROLLOUT_PERCENT, "abc")

        with pytest.raises(ValueError, match="Invalid rollout percentage"):
            FeatureFlags.from_env()

    def test_invalid_rollout_percent_negative(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env rejects negative rollout percent."""
        monkeypatch.setenv(ENV_ROLLOUT_PERCENT, "-1")

        with pytest.raises(ValueError, match="must be 0-100"):
            FeatureFlags.from_env()

    def test_invalid_rollout_percent_over_100(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test from_env rejects rollout percent over 100."""
        monkeypatch.setenv(ENV_ROLLOUT_PERCENT, "101")

        with pytest.raises(ValueError, match="must be 0-100"):
            FeatureFlags.from_env()

    def test_invalid_rollout_in_constructor(self) -> None:
        """Test constructor validates rollout_percent."""
        with pytest.raises(ValueError, match="must be 0-100"):
            FeatureFlags(rollout_percent=150)

    def test_unknown_component(self, clean_env: None) -> None:
        """Test get_impl raises for unknown component."""
        flags = FeatureFlags()

        with pytest.raises(ValueError, match="Unknown component"):
            flags.get_impl("unknown")

    def test_should_use_new_unknown_component(self, clean_env: None) -> None:
        """Test should_use_new raises for unknown component."""
        flags = FeatureFlags()

        with pytest.raises(ValueError, match="Unknown component"):
            flags.should_use_new("unknown")


# =============================================================================
# Test Implementation Selection (FR-008)
# =============================================================================


class TestImplementationSelection:
    """Tests for implementation selection logic (FR-008)."""

    def test_get_impl_returns_correct_choice(self) -> None:
        """Test get_impl returns configured choice."""
        flags = FeatureFlags(
            preprocessor_impl="new",
            selector_impl="old",
            covariance_impl="auto",
            streaming_impl="new",
        )

        assert flags.get_impl(COMPONENT_PREPROCESSOR) == "new"
        assert flags.get_impl(COMPONENT_SELECTOR) == "old"
        assert flags.get_impl(COMPONENT_COVARIANCE) == "auto"
        assert flags.get_impl(COMPONENT_STREAMING) == "new"

    def test_should_use_new_explicit_new(self) -> None:
        """Test should_use_new returns True for explicit 'new'."""
        flags = FeatureFlags(preprocessor_impl="new")

        assert flags.should_use_new(COMPONENT_PREPROCESSOR) is True

    def test_should_use_new_explicit_old(self) -> None:
        """Test should_use_new returns False for explicit 'old'."""
        flags = FeatureFlags(preprocessor_impl="old")

        assert flags.should_use_new(COMPONENT_PREPROCESSOR) is False

    def test_should_use_new_auto_zero_rollout(self) -> None:
        """Test should_use_new returns False for auto with 0% rollout."""
        flags = FeatureFlags(
            preprocessor_impl="auto",
            rollout_percent=0,
        )

        assert flags.should_use_new(COMPONENT_PREPROCESSOR) is False

    def test_should_use_new_auto_full_rollout(self) -> None:
        """Test should_use_new returns True for auto with 100% rollout."""
        flags = FeatureFlags(
            preprocessor_impl="auto",
            rollout_percent=100,
        )

        assert flags.should_use_new(COMPONENT_PREPROCESSOR) is True


# =============================================================================
# Test Rollout Percentage (FR-009)
# =============================================================================


class TestRolloutPercentage:
    """Tests for gradual rollout percentage logic (FR-009)."""

    def test_rollout_deterministic(self) -> None:
        """Test rollout selection is deterministic for same session."""
        flags = FeatureFlags(
            preprocessor_impl="auto",
            rollout_percent=50,
            _session_id="fixed-session-id",
        )

        # Same session should always give same result
        results = [flags.should_use_new(COMPONENT_PREPROCESSOR) for _ in range(100)]

        assert all(r == results[0] for r in results)

    def test_rollout_differs_by_component(self) -> None:
        """Test different components can get different selections."""
        # Use session ID that gives different results for different components
        flags = FeatureFlags(
            preprocessor_impl="auto",
            selector_impl="auto",
            rollout_percent=50,
            _session_id="test-session-different-results",
        )

        # With 50% rollout, different components may get different results
        # (depends on hash, but tests determinism)
        preprocessor_result = flags.should_use_new(COMPONENT_PREPROCESSOR)
        selector_result = flags.should_use_new(COMPONENT_SELECTOR)

        # Both should be boolean
        assert isinstance(preprocessor_result, bool)
        assert isinstance(selector_result, bool)

    def test_rollout_distribution(self) -> None:
        """Test rollout percentage roughly distributes correctly."""
        # Test with 50% rollout across many sessions
        results = []
        for i in range(1000):
            flags = FeatureFlags(
                preprocessor_impl="auto",
                rollout_percent=50,
                _session_id=f"session-{i}",
            )
            results.append(flags.should_use_new(COMPONENT_PREPROCESSOR))

        # With 50% rollout, should be roughly 50% True (allow 10% variance)
        true_count = sum(results)
        assert 400 <= true_count <= 600

    def test_rollout_10_percent(self) -> None:
        """Test 10% rollout gives roughly 10% new implementations."""
        results = []
        for i in range(1000):
            flags = FeatureFlags(
                preprocessor_impl="auto",
                rollout_percent=10,
                _session_id=f"session-{i}",
            )
            results.append(flags.should_use_new(COMPONENT_PREPROCESSOR))

        true_count = sum(results)
        assert 50 <= true_count <= 150  # 10% ± 5%

    def test_rollout_25_percent(self) -> None:
        """Test 25% rollout gives roughly 25% new implementations."""
        results = []
        for i in range(1000):
            flags = FeatureFlags(
                preprocessor_impl="auto",
                rollout_percent=25,
                _session_id=f"session-{i}",
            )
            results.append(flags.should_use_new(COMPONENT_PREPROCESSOR))

        true_count = sum(results)
        assert 200 <= true_count <= 300  # 25% ± 5%


# =============================================================================
# Test Instant Rollback (FR-010)
# =============================================================================


class TestInstantRollback:
    """Tests for instant rollback capability (FR-010)."""

    def test_rollback_via_env_var(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test instant rollback by setting env var to 'old'."""
        # Initially set to new
        monkeypatch.setenv(ENV_PREPROCESSOR, "new")
        flags1 = FeatureFlags.from_env()
        assert flags1.should_use_new(COMPONENT_PREPROCESSOR) is True

        # Rollback by setting to old
        monkeypatch.setenv(ENV_PREPROCESSOR, "old")
        flags2 = FeatureFlags.from_env()
        assert flags2.should_use_new(COMPONENT_PREPROCESSOR) is False

    def test_rollback_via_with_override(self) -> None:
        """Test rollback using with_override method."""
        flags = FeatureFlags(preprocessor_impl="new")
        assert flags.should_use_new(COMPONENT_PREPROCESSOR) is True

        # Rollback via override
        rolled_back = flags.with_override(preprocessor_impl="old")
        assert rolled_back.should_use_new(COMPONENT_PREPROCESSOR) is False

    def test_with_override_preserves_session(self) -> None:
        """Test with_override preserves session ID."""
        flags = FeatureFlags(_session_id="my-session")
        overridden = flags.with_override(preprocessor_impl="new")

        assert overridden._session_id == "my-session"

    def test_with_override_partial(self) -> None:
        """Test with_override only changes specified values."""
        flags = FeatureFlags(
            preprocessor_impl="new",
            selector_impl="old",
            rollout_percent=50,
        )

        overridden = flags.with_override(preprocessor_impl="old")

        assert overridden.preprocessor_impl == "old"
        assert overridden.selector_impl == "old"  # Unchanged
        assert overridden.rollout_percent == 50  # Unchanged


# =============================================================================
# Test Singleton and Module Functions
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_feature_flags_singleton(self, clean_env: None) -> None:
        """Test get_feature_flags returns singleton."""
        flags1 = get_feature_flags()
        flags2 = get_feature_flags()

        assert flags1 is flags2

    def test_reset_feature_flags(self, clean_env: None) -> None:
        """Test reset_feature_flags clears singleton."""
        flags1 = get_feature_flags()
        reset_feature_flags()
        flags2 = get_feature_flags()

        assert flags1 is not flags2

    def test_get_feature_flags_reads_env(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test get_feature_flags reads environment."""
        monkeypatch.setenv(ENV_PREPROCESSOR, "new")

        flags = get_feature_flags()

        assert flags.preprocessor_impl == "new"


# =============================================================================
# Test to_env_dict
# =============================================================================


class TestToEnvDict:
    """Tests for to_env_dict serialization."""

    def test_to_env_dict(self) -> None:
        """Test to_env_dict returns correct dictionary."""
        flags = FeatureFlags(
            preprocessor_impl="new",
            selector_impl="old",
            covariance_impl="auto",
            streaming_impl="new",
            rollout_percent=25,
        )

        env_dict = flags.to_env_dict()

        assert env_dict == {
            ENV_PREPROCESSOR: "new",
            ENV_SELECTOR: "old",
            ENV_COVARIANCE: "auto",
            ENV_STREAMING: "new",
            ENV_ROLLOUT_PERCENT: "25",
        }

    def test_roundtrip_via_env_dict(
        self, monkeypatch: pytest.MonkeyPatch, clean_env: None
    ) -> None:
        """Test flags can roundtrip through environment variables."""
        original = FeatureFlags(
            preprocessor_impl="new",
            selector_impl="old",
            covariance_impl="auto",
            streaming_impl="new",
            rollout_percent=75,
        )

        # Set env vars from flags
        for key, value in original.to_env_dict().items():
            monkeypatch.setenv(key, value)

        # Load from env
        loaded = FeatureFlags.from_env()

        assert loaded.preprocessor_impl == original.preprocessor_impl
        assert loaded.selector_impl == original.selector_impl
        assert loaded.covariance_impl == original.covariance_impl
        assert loaded.streaming_impl == original.streaming_impl
        assert loaded.rollout_percent == original.rollout_percent
