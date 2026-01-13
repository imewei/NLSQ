"""Feature flags for CurveFit component extraction.

This module provides the FeatureFlags system for controlling the gradual
rollout of decomposed CurveFit components. Each component can be toggled
between old (original minpack.py code) and new (extracted component) implementations.

Environment Variables:
    NLSQ_PREPROCESSOR_IMPL: 'old', 'new', or 'auto' (default: 'auto')
    NLSQ_SELECTOR_IMPL: 'old', 'new', or 'auto' (default: 'auto')
    NLSQ_COVARIANCE_IMPL: 'old', 'new', or 'auto' (default: 'auto')
    NLSQ_STREAMING_IMPL: 'old', 'new', or 'auto' (default: 'auto')
    NLSQ_REFACTOR_ROLLOUT_PERCENT: 0-100 (default: 0)

Usage:
    from nlsq.core.feature_flags import FeatureFlags

    flags = FeatureFlags.from_env()
    if flags.should_use_new("preprocessor"):
        # Use new DataPreprocessor component
        ...
    else:
        # Use original minpack.py code path
        ...

Reference: specs/017-curve-fit-decomposition/spec.md FR-008, FR-009, FR-010
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Literal

# Valid implementation choices
ImplChoice = Literal["old", "new", "auto"]

# Component names
COMPONENT_PREPROCESSOR = "preprocessor"
COMPONENT_SELECTOR = "selector"
COMPONENT_COVARIANCE = "covariance"
COMPONENT_STREAMING = "streaming"

# Environment variable names
ENV_PREPROCESSOR = "NLSQ_PREPROCESSOR_IMPL"
ENV_SELECTOR = "NLSQ_SELECTOR_IMPL"
ENV_COVARIANCE = "NLSQ_COVARIANCE_IMPL"
ENV_STREAMING = "NLSQ_STREAMING_IMPL"
ENV_ROLLOUT_PERCENT = "NLSQ_REFACTOR_ROLLOUT_PERCENT"

# Default values
DEFAULT_IMPL: ImplChoice = "auto"
DEFAULT_ROLLOUT_PERCENT = 0  # Start with 0% for safe rollout


@dataclass(frozen=True, slots=True)
class FeatureFlags:
    """Feature flags for CurveFit component extraction.

    Controls which implementation (old or new) to use for each extracted
    component. Supports gradual rollout via percentage-based selection.

    Attributes:
        preprocessor_impl: Implementation choice for DataPreprocessor
        selector_impl: Implementation choice for OptimizationSelector
        covariance_impl: Implementation choice for CovarianceComputer
        streaming_impl: Implementation choice for StreamingCoordinator
        rollout_percent: Percentage (0-100) of requests using new implementation
            when impl is 'auto'
        _session_id: Internal session ID for deterministic rollout selection
    """

    preprocessor_impl: ImplChoice = DEFAULT_IMPL
    selector_impl: ImplChoice = DEFAULT_IMPL
    covariance_impl: ImplChoice = DEFAULT_IMPL
    streaming_impl: ImplChoice = DEFAULT_IMPL
    rollout_percent: int = DEFAULT_ROLLOUT_PERCENT
    _session_id: str = field(default_factory=lambda: os.urandom(8).hex())

    def __post_init__(self) -> None:
        """Validate rollout_percent range."""
        if not 0 <= self.rollout_percent <= 100:
            msg = f"rollout_percent must be 0-100, got {self.rollout_percent}"
            raise ValueError(msg)

    @classmethod
    def from_env(cls, session_id: str | None = None) -> FeatureFlags:
        """Create FeatureFlags from environment variables.

        Args:
            session_id: Optional session ID for deterministic rollout.
                If not provided, a random ID is generated.

        Returns:
            FeatureFlags instance with values from environment.

        Raises:
            ValueError: If environment variable has invalid value.
        """
        preprocessor = cls._parse_impl_choice(
            os.environ.get(ENV_PREPROCESSOR, DEFAULT_IMPL)
        )
        selector = cls._parse_impl_choice(os.environ.get(ENV_SELECTOR, DEFAULT_IMPL))
        covariance = cls._parse_impl_choice(
            os.environ.get(ENV_COVARIANCE, DEFAULT_IMPL)
        )
        streaming = cls._parse_impl_choice(os.environ.get(ENV_STREAMING, DEFAULT_IMPL))
        rollout = cls._parse_rollout_percent(
            os.environ.get(ENV_ROLLOUT_PERCENT, str(DEFAULT_ROLLOUT_PERCENT))
        )

        return cls(
            preprocessor_impl=preprocessor,
            selector_impl=selector,
            covariance_impl=covariance,
            streaming_impl=streaming,
            rollout_percent=rollout,
            _session_id=session_id or os.urandom(8).hex(),
        )

    @staticmethod
    def _parse_impl_choice(value: str) -> ImplChoice:
        """Parse implementation choice from string.

        Args:
            value: String value to parse (case-insensitive)

        Returns:
            Validated ImplChoice

        Raises:
            ValueError: If value is not 'old', 'new', or 'auto'
        """
        normalized = value.lower().strip()
        if normalized not in ("old", "new", "auto"):
            msg = f"Invalid implementation choice '{value}', must be 'old', 'new', or 'auto'"
            raise ValueError(msg)
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _parse_rollout_percent(value: str) -> int:
        """Parse rollout percentage from string.

        Args:
            value: String value to parse

        Returns:
            Integer percentage 0-100

        Raises:
            ValueError: If value is not a valid integer 0-100
        """
        try:
            percent = int(value)
        except ValueError:
            msg = f"Invalid rollout percentage '{value}', must be integer 0-100"
            raise ValueError(msg) from None

        if not 0 <= percent <= 100:
            msg = f"Rollout percentage must be 0-100, got {percent}"
            raise ValueError(msg)
        return percent

    def get_impl(self, component: str) -> ImplChoice:
        """Get implementation choice for a component.

        Args:
            component: Component name ('preprocessor', 'selector',
                'covariance', 'streaming')

        Returns:
            Implementation choice for the component

        Raises:
            ValueError: If component name is unknown
        """
        impl_map = {
            COMPONENT_PREPROCESSOR: self.preprocessor_impl,
            COMPONENT_SELECTOR: self.selector_impl,
            COMPONENT_COVARIANCE: self.covariance_impl,
            COMPONENT_STREAMING: self.streaming_impl,
        }
        if component not in impl_map:
            msg = f"Unknown component '{component}', valid: {list(impl_map.keys())}"
            raise ValueError(msg)
        return impl_map[component]

    def should_use_new(self, component: str) -> bool:
        """Determine if new implementation should be used.

        For 'old' or 'new' choices, returns the explicit choice.
        For 'auto', uses rollout_percent with deterministic selection
        based on session_id and component name.

        Args:
            component: Component name ('preprocessor', 'selector',
                'covariance', 'streaming')

        Returns:
            True if new implementation should be used, False for old

        Raises:
            ValueError: If component name is unknown
        """
        impl = self.get_impl(component)

        if impl == "new":
            return True
        if impl == "old":
            return False

        # 'auto' mode: use rollout percentage with deterministic selection
        return self._is_in_rollout(component)

    def _is_in_rollout(self, component: str) -> bool:
        """Determine if session is in rollout for component.

        Uses deterministic hashing based on session_id and component
        to ensure consistent behavior within a session.

        Args:
            component: Component name

        Returns:
            True if session falls within rollout percentage
        """
        if self.rollout_percent == 0:
            return False
        if self.rollout_percent == 100:
            return True

        # Create deterministic hash from session_id and component
        hash_input = f"{self._session_id}:{component}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        # Use first 4 bytes as integer, modulo 100
        hash_value = int.from_bytes(hash_bytes[:4], "big") % 100

        return hash_value < self.rollout_percent

    def with_override(
        self,
        *,
        preprocessor_impl: ImplChoice | None = None,
        selector_impl: ImplChoice | None = None,
        covariance_impl: ImplChoice | None = None,
        streaming_impl: ImplChoice | None = None,
        rollout_percent: int | None = None,
    ) -> FeatureFlags:
        """Create new FeatureFlags with overridden values.

        Args:
            preprocessor_impl: Override for preprocessor implementation
            selector_impl: Override for selector implementation
            covariance_impl: Override for covariance implementation
            streaming_impl: Override for streaming implementation
            rollout_percent: Override for rollout percentage

        Returns:
            New FeatureFlags instance with overridden values
        """
        return FeatureFlags(
            preprocessor_impl=preprocessor_impl or self.preprocessor_impl,
            selector_impl=selector_impl or self.selector_impl,
            covariance_impl=covariance_impl or self.covariance_impl,
            streaming_impl=streaming_impl or self.streaming_impl,
            rollout_percent=rollout_percent
            if rollout_percent is not None
            else self.rollout_percent,
            _session_id=self._session_id,
        )

    def to_env_dict(self) -> dict[str, str]:
        """Convert flags to environment variable dictionary.

        Returns:
            Dictionary mapping env var names to values
        """
        return {
            ENV_PREPROCESSOR: self.preprocessor_impl,
            ENV_SELECTOR: self.selector_impl,
            ENV_COVARIANCE: self.covariance_impl,
            ENV_STREAMING: self.streaming_impl,
            ENV_ROLLOUT_PERCENT: str(self.rollout_percent),
        }


# Module-level singleton for convenience
_default_flags: FeatureFlags | None = None


def get_feature_flags() -> FeatureFlags:
    """Get or create default FeatureFlags instance.

    Creates a singleton instance on first call using from_env().
    Subsequent calls return the same instance.

    Returns:
        FeatureFlags instance
    """
    global _default_flags  # noqa: PLW0603
    if _default_flags is None:
        _default_flags = FeatureFlags.from_env()
    return _default_flags


def reset_feature_flags() -> None:
    """Reset the default FeatureFlags singleton.

    Useful for testing to force re-reading environment variables.
    """
    global _default_flags  # noqa: PLW0603
    _default_flags = None
