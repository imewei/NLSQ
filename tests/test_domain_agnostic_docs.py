"""Tests for domain-agnostic documentation.

These tests verify that domain-specific references (XPCS, scattering, etc.)
have been removed from core documentation in favor of generic descriptions.
"""

import subprocess
from pathlib import Path

import pytest


class TestDomainAgnosticDocumentation:
    """Test that domain-specific references are removed from documentation."""

    @pytest.fixture
    def nlsq_root(self) -> Path:
        """Get the NLSQ package root directory."""
        return Path(__file__).parent.parent / "nlsq"

    def test_no_xpcs_in_stability_guard(self, nlsq_root: Path) -> None:
        """Test that stability/guard.py has no XPCS references."""
        guard_path = nlsq_root / "stability" / "guard.py"
        assert guard_path.exists(), f"File not found: {guard_path}"

        result = subprocess.run(
            ["grep", "-n", "-i", "xpcs", str(guard_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        # grep returns 0 if matches found, 1 if no matches
        assert result.returncode == 1, (
            f"Found XPCS references in stability/guard.py:\n{result.stdout}"
        )

    def test_no_xpcs_in_hybrid_config(self, nlsq_root: Path) -> None:
        """Test that streaming/hybrid_config.py has no XPCS references."""
        hybrid_config_path = nlsq_root / "streaming" / "hybrid_config.py"
        assert hybrid_config_path.exists(), f"File not found: {hybrid_config_path}"

        result = subprocess.run(
            ["grep", "-n", "-i", "xpcs", str(hybrid_config_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        # grep returns 0 if matches found, 1 if no matches
        assert result.returncode == 1, (
            f"Found XPCS references in streaming/hybrid_config.py:\n{result.stdout}"
        )

    def test_no_domain_specific_technique_names_in_core_docs(
        self, nlsq_root: Path
    ) -> None:
        """Test that core documentation has no domain-specific technique names.

        This checks that files in stability/ and streaming/ do not contain
        references to specific scientific techniques like XPCS, SAXS, scattering,
        etc. in their docstrings.
        """
        # Files to check
        files_to_check = [
            nlsq_root / "stability" / "guard.py",
            nlsq_root / "stability" / "svd_fallback.py",
            nlsq_root / "streaming" / "hybrid_config.py",
        ]

        # Domain-specific terms to check for (case-insensitive)
        domain_terms = ["xpcs", "saxs", "scattering vector"]

        violations = []
        for file_path in files_to_check:
            if not file_path.exists():
                continue

            for term in domain_terms:
                result = subprocess.run(
                    ["grep", "-n", "-i", term, str(file_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    violations.append(
                        f"{file_path.name}: '{term}' found:\n{result.stdout}"
                    )

        assert not violations, (
            "Found domain-specific terms in core documentation:\n"
            + "\n".join(violations)
        )
