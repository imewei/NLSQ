"""Tests for test file structure after parameter sensitivity rename.

This test module verifies that:
- The renamed test file exists at the correct location
- The old test file does not exist (or is a redirect)
- No SLOPPY- issue codes appear in main test assertions (except deprecation tests)

Task Group 6.1: Test file structure verification.
"""

import os
from pathlib import Path


class TestTestFileStructure:
    """Tests verifying test file structure after rename."""

    def test_parameter_sensitivity_test_file_exists(self) -> None:
        """Test tests/diagnostics/test_parameter_sensitivity.py exists."""
        tests_dir = Path(__file__).parent
        test_file = tests_dir / "diagnostics" / "test_parameter_sensitivity.py"
        assert test_file.exists(), (
            f"Expected test file at {test_file} but it does not exist"
        )

    def test_sloppy_model_test_file_does_not_exist(self) -> None:
        """Test tests/diagnostics/test_sloppy_model.py does not exist.

        The old test file should have been renamed to test_parameter_sensitivity.py.
        """
        tests_dir = Path(__file__).parent
        old_test_file = tests_dir / "diagnostics" / "test_sloppy_model.py"
        assert not old_test_file.exists(), (
            f"Old test file {old_test_file} should not exist after rename"
        )

    def test_no_sloppy_issue_codes_in_main_test_assertions(self) -> None:
        """Test no SLOPPY- issue codes appear in main test assertions.

        The issue codes should have been changed from SLOPPY-* to SENS-*.
        This test checks that the main test file uses the new codes.

        Note: Deprecation tests are exempt from this check as they verify
        backwards compatibility.
        """
        tests_dir = Path(__file__).parent
        test_file = tests_dir / "diagnostics" / "test_parameter_sensitivity.py"

        if not test_file.exists():
            # File doesn't exist yet - test passes trivially
            return

        content = test_file.read_text()

        # Find all SLOPPY- references that are in assertions (not deprecation tests)
        lines = content.split("\n")
        violations = []

        in_deprecation_test = False
        for line_num, line in enumerate(lines, 1):
            # Track if we're in a deprecation test class/method
            stripped = line.strip()
            if (
                "class TestDeprecation" in stripped
                or "def test_deprecat" in stripped.lower()
            ):
                in_deprecation_test = True
            elif stripped.startswith("class Test") or (
                stripped.startswith("def test_") and "deprecat" not in stripped.lower()
            ):
                in_deprecation_test = False

            # Skip deprecation tests and comments
            if in_deprecation_test:
                continue
            if stripped.startswith("#"):
                continue
            if "docstring" in stripped.lower() or '"""' in stripped:
                continue

            # Check for SLOPPY- in assertions
            if "SLOPPY-" in line and ("assert" in line or "== " in line):
                violations.append(f"Line {line_num}: {line.strip()}")

        assert len(violations) == 0, (
            "Found SLOPPY- issue codes in main test assertions (should be SENS-):\n"
            + "\n".join(violations)
        )

    def test_renamed_test_file_uses_new_imports(self) -> None:
        """Test the renamed test file imports from parameter_sensitivity module."""
        tests_dir = Path(__file__).parent
        test_file = tests_dir / "diagnostics" / "test_parameter_sensitivity.py"

        if not test_file.exists():
            # File doesn't exist yet - test passes trivially
            return

        content = test_file.read_text()

        # Check that new imports are present
        assert "ParameterSensitivityAnalyzer" in content, (
            "Test file should import ParameterSensitivityAnalyzer"
        )
        assert "ParameterSensitivityReport" in content, (
            "Test file should import ParameterSensitivityReport"
        )

        # Check that it imports from the new module path
        # Could be from nlsq.diagnostics.parameter_sensitivity or nlsq.diagnostics
        has_correct_import = (
            "from nlsq.diagnostics.parameter_sensitivity import" in content
            or "from nlsq.diagnostics import" in content
        )
        assert has_correct_import, (
            "Test file should import from nlsq.diagnostics.parameter_sensitivity "
            "or nlsq.diagnostics"
        )
