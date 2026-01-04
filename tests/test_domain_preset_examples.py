"""Tests for domain preset example files.

This module tests the domain preset example files created as part of the
domain-specific content removal spec. These examples show users how to
recreate domain-specific presets using the with_overrides() pattern.

Tests verify:
- Example files exist and are syntactically valid Python
- Example files demonstrate the with_overrides() pattern correctly
- Example files can be imported without errors
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

# Path to the workflow_system example directory (domain presets were relocated here)
EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples" / "scripts" / "08_workflow_system"
)

# File name mapping for reorganized domain preset examples
XPCS_FILE = "11_xpcs_presets.py"
SAXS_FILE = "10_saxs_presets.py"
KINETICS_FILE = "09_kinetics_presets.py"
CUSTOM_GUIDE_FILE = "08_custom_presets.py"


class TestDomainPresetExamplesExist:
    """Test that domain preset example files exist and are valid Python."""

    def test_xpcs_preset_file_exists(self):
        """Test that XPCS preset file exists."""
        xpcs_path = EXAMPLES_DIR / XPCS_FILE
        assert xpcs_path.exists(), f"Expected {xpcs_path} to exist"

    def test_xpcs_preset_is_valid_python(self):
        """Test that XPCS preset file is syntactically valid Python."""
        xpcs_path = EXAMPLES_DIR / XPCS_FILE
        with open(xpcs_path) as f:
            source = f.read()

        # This will raise SyntaxError if invalid
        ast.parse(source)

    def test_saxs_preset_file_exists(self):
        """Test that SAXS preset file exists."""
        saxs_path = EXAMPLES_DIR / SAXS_FILE
        assert saxs_path.exists(), f"Expected {saxs_path} to exist"

    def test_saxs_preset_is_valid_python(self):
        """Test that SAXS preset file is syntactically valid Python."""
        saxs_path = EXAMPLES_DIR / SAXS_FILE
        with open(saxs_path) as f:
            source = f.read()

        ast.parse(source)

    def test_kinetics_preset_file_exists(self):
        """Test that kinetics preset file exists."""
        kinetics_path = EXAMPLES_DIR / KINETICS_FILE
        assert kinetics_path.exists(), f"Expected {kinetics_path} to exist"

    def test_kinetics_preset_is_valid_python(self):
        """Test that kinetics preset file is syntactically valid Python."""
        kinetics_path = EXAMPLES_DIR / KINETICS_FILE
        with open(kinetics_path) as f:
            source = f.read()

        ast.parse(source)

    def test_custom_preset_guide_file_exists(self):
        """Test that custom preset guide file exists."""
        guide_path = EXAMPLES_DIR / CUSTOM_GUIDE_FILE
        assert guide_path.exists(), f"Expected {guide_path} to exist"

    def test_custom_preset_guide_is_valid_python(self):
        """Test that custom preset guide file is syntactically valid Python."""
        guide_path = EXAMPLES_DIR / CUSTOM_GUIDE_FILE
        with open(guide_path) as f:
            source = f.read()

        ast.parse(source)


class TestCustomPresetGuideContent:
    """Test that custom preset guide demonstrates the with_overrides() pattern."""

    def test_custom_preset_guide_contains_with_overrides(self):
        """Test that custom preset guide demonstrates with_overrides() pattern."""
        guide_path = EXAMPLES_DIR / CUSTOM_GUIDE_FILE
        with open(guide_path) as f:
            source = f.read()

        # The guide should demonstrate the with_overrides() pattern
        assert "with_overrides" in source, (
            f"{CUSTOM_GUIDE_FILE} should demonstrate the with_overrides() pattern"
        )

    def test_custom_preset_guide_contains_from_preset(self):
        """Test that custom preset guide uses WorkflowConfig.from_preset()."""
        guide_path = EXAMPLES_DIR / CUSTOM_GUIDE_FILE
        with open(guide_path) as f:
            source = f.read()

        assert "from_preset" in source, (
            f"{CUSTOM_GUIDE_FILE} should use WorkflowConfig.from_preset()"
        )

    def test_custom_preset_guide_imports_workflow_config(self):
        """Test that custom preset guide imports WorkflowConfig."""
        guide_path = EXAMPLES_DIR / CUSTOM_GUIDE_FILE
        with open(guide_path) as f:
            source = f.read()

        assert "WorkflowConfig" in source, (
            f"{CUSTOM_GUIDE_FILE} should import WorkflowConfig"
        )


class TestExampleScriptsImportSuccessfully:
    """Test that example scripts can be imported without errors."""

    def _import_module_from_path(self, module_name: str, file_path: Path):
        """Helper to import a module from a file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def test_xpcs_preset_imports_successfully(self):
        """Test that XPCS preset imports without errors."""
        xpcs_path = EXAMPLES_DIR / XPCS_FILE
        module = self._import_module_from_path("test_xpcs_preset", xpcs_path)

        # Verify key function exists
        assert hasattr(module, "create_xpcs_preset"), (
            f"{XPCS_FILE} should define create_xpcs_preset()"
        )

        # Verify function works
        config = module.create_xpcs_preset()
        assert config is not None
        assert hasattr(config, "gtol")
        assert hasattr(config, "enable_multistart")

    def test_saxs_preset_imports_successfully(self):
        """Test that SAXS preset imports without errors."""
        saxs_path = EXAMPLES_DIR / SAXS_FILE
        module = self._import_module_from_path("test_saxs_preset", saxs_path)

        # Verify key function exists
        assert hasattr(module, "create_saxs_preset"), (
            f"{SAXS_FILE} should define create_saxs_preset()"
        )

        # Verify function works
        config = module.create_saxs_preset()
        assert config is not None
        assert config.gtol == 1e-9  # SAXS preset uses tighter tolerances

    def test_kinetics_preset_imports_successfully(self):
        """Test that kinetics preset imports without errors."""
        kinetics_path = EXAMPLES_DIR / KINETICS_FILE
        module = self._import_module_from_path("test_kinetics_preset", kinetics_path)

        # Verify key function exists
        assert hasattr(module, "create_kinetics_preset"), (
            f"{KINETICS_FILE} should define create_kinetics_preset()"
        )

        # Verify function works
        config = module.create_kinetics_preset()
        assert config is not None
        assert config.n_starts == 20  # Kinetics preset uses n_starts=20

    def test_custom_preset_guide_imports_successfully(self):
        """Test that custom preset guide imports without errors."""
        guide_path = EXAMPLES_DIR / CUSTOM_GUIDE_FILE

        # Just verify it can be imported (parsing and imports work)
        spec = importlib.util.spec_from_file_location("test_custom_guide", guide_path)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        # Execute to verify all imports resolve
        sys.modules["test_custom_guide"] = module
        spec.loader.exec_module(module)

        # Verify main function exists
        assert hasattr(module, "main"), f"{CUSTOM_GUIDE_FILE} should define main()"


class TestDomainPresetPatterns:
    """Test that domain preset examples follow the with_overrides() pattern correctly."""

    def test_xpcs_preset_uses_precision_standard_base(self):
        """Test that XPCS preset builds on precision_standard."""
        xpcs_path = EXAMPLES_DIR / XPCS_FILE
        with open(xpcs_path) as f:
            source = f.read()

        # XPCS should use precision_standard as base
        assert "precision_standard" in source, (
            f"{XPCS_FILE} should use precision_standard as base preset"
        )

    def test_saxs_preset_uses_precision_standard_base(self):
        """Test that SAXS preset builds on precision_standard."""
        saxs_path = EXAMPLES_DIR / SAXS_FILE
        with open(saxs_path) as f:
            source = f.read()

        assert "precision_standard" in source, (
            f"{SAXS_FILE} should use precision_standard as base preset"
        )

    def test_kinetics_preset_uses_precision_standard_base(self):
        """Test that kinetics preset builds on precision_standard."""
        kinetics_path = EXAMPLES_DIR / KINETICS_FILE
        with open(kinetics_path) as f:
            source = f.read()

        assert "precision_standard" in source, (
            f"{KINETICS_FILE} should use precision_standard as base preset"
        )
