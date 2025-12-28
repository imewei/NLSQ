"""Comprehensive tests for notebook transformation utilities.

Tests cover:
- Individual transformer classes (Strategy pattern)
- Pipeline composition (Chain of Responsibility)
- Incremental processing tracker (checksum-based)
- Edge cases and error handling
"""

import json
import sys

# Add scripts to path for imports
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from notebook_utils.cells import (
    create_ipython_display_import_cell,
    create_matplotlib_config_cell,
    find_cell_with_pattern,
    find_first_code_cell_index,
    has_ipython_display_import,
    has_matplotlib_magic,
    uses_display,
)
from notebook_utils.pipeline import TransformationPipeline
from notebook_utils.tracking import ProcessingTracker
from notebook_utils.transformations import (
    IPythonDisplayImportTransformer,
    MatplotlibInlineTransformer,
    PltShowReplacementTransformer,
)
from notebook_utils.transformations.base import NotebookTransformer
from notebook_utils.transformations.plt_show import (
    find_figure_variable,
    replace_plt_show,
)
from notebook_utils.types import NotebookCell

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def empty_notebook_cells() -> list[NotebookCell]:
    """Empty notebook with no cells."""
    return []


@pytest.fixture
def simple_code_cells() -> list[NotebookCell]:
    """Simple notebook with code cells."""
    return [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import numpy as np"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["x = np.array([1, 2, 3])"],
        },
    ]


@pytest.fixture
def notebook_with_matplotlib() -> list[NotebookCell]:
    """Notebook with matplotlib code."""
    return [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import matplotlib.pyplot as plt"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, ax = plt.subplots()\n",
                "ax.plot([1, 2, 3])\n",
                "plt.show()",
            ],
        },
    ]


@pytest.fixture
def notebook_with_magic() -> list[NotebookCell]:
    """Notebook already has %matplotlib inline."""
    return [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%matplotlib inline"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import matplotlib.pyplot as plt"],
        },
    ]


@pytest.fixture
def notebook_with_display() -> list[NotebookCell]:
    """Notebook uses display() function."""
    return [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%matplotlib inline"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["fig = plt.figure()\n", "display(fig)"],
        },
    ]


@pytest.fixture
def notebook_with_display_import() -> list[NotebookCell]:
    """Notebook has display import."""
    return [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%matplotlib inline"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["from IPython.display import display"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["fig = plt.figure()\n", "display(fig)"],
        },
    ]


@pytest.fixture
def temp_notebook_file(tmp_path: Path) -> Path:
    """Create a temporary notebook file."""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import numpy as np"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path = tmp_path / "test_notebook.ipynb"
    with open(notebook_path, "w") as f:
        json.dump(notebook, f)

    return notebook_path


# ============================================================================
# Cell Utility Tests
# ============================================================================


class TestCellUtilities:
    """Test cell manipulation utility functions."""

    def test_has_matplotlib_magic_true(self, notebook_with_magic):
        """Test detection of %matplotlib inline magic."""
        assert has_matplotlib_magic(notebook_with_magic) is True

    def test_has_matplotlib_magic_false(self, simple_code_cells):
        """Test detection when magic not present."""
        assert has_matplotlib_magic(simple_code_cells) is False

    def test_has_matplotlib_magic_empty(self, empty_notebook_cells):
        """Test detection with empty notebook."""
        assert has_matplotlib_magic(empty_notebook_cells) is False

    def test_has_ipython_display_import_true(self, notebook_with_display_import):
        """Test detection of IPython.display import."""
        assert has_ipython_display_import(notebook_with_display_import) is True

    def test_has_ipython_display_import_false(self, simple_code_cells):
        """Test detection when import not present."""
        assert has_ipython_display_import(simple_code_cells) is False

    def test_uses_display_true(self, notebook_with_display):
        """Test detection of display() usage."""
        assert uses_display(notebook_with_display) is True

    def test_uses_display_false(self, simple_code_cells):
        """Test detection when display() not used."""
        assert uses_display(simple_code_cells) is False

    def test_find_first_code_cell_index(self, simple_code_cells):
        """Test finding first code cell."""
        assert find_first_code_cell_index(simple_code_cells) == 0

    def test_find_first_code_cell_with_markdown(self):
        """Test finding first code cell when markdown comes first."""
        cells = [
            {"cell_type": "markdown", "source": ["# Title"]},
            {"cell_type": "code", "source": ["import numpy"]},
        ]
        assert find_first_code_cell_index(cells) == 1

    def test_find_cell_with_pattern_found(self, notebook_with_magic):
        """Test finding cell with pattern."""
        idx = find_cell_with_pattern(notebook_with_magic, "%matplotlib inline")
        assert idx == 0

    def test_find_cell_with_pattern_not_found(self, simple_code_cells):
        """Test finding cell when pattern not present."""
        idx = find_cell_with_pattern(simple_code_cells, "nonexistent")
        assert idx is None

    def test_create_matplotlib_config_cell(self):
        """Test creation of matplotlib config cell."""
        cell = create_matplotlib_config_cell()
        assert cell["cell_type"] == "code"
        assert "%matplotlib inline" in "".join(cell["source"])

    def test_create_ipython_display_import_cell(self):
        """Test creation of display import cell."""
        cell = create_ipython_display_import_cell()
        assert cell["cell_type"] == "code"
        assert "from IPython.display import display" in "".join(cell["source"])


# ============================================================================
# MatplotlibInlineTransformer Tests
# ============================================================================


class TestMatplotlibInlineTransformer:
    """Test MatplotlibInlineTransformer class."""

    def test_name(self):
        """Test transformer name."""
        transformer = MatplotlibInlineTransformer()
        assert transformer.name() == "matplotlib_inline"

    def test_description(self):
        """Test transformer description."""
        transformer = MatplotlibInlineTransformer()
        assert "matplotlib inline" in transformer.description().lower()

    def test_should_apply_true(self, simple_code_cells):
        """Test should_apply returns True when magic not present."""
        transformer = MatplotlibInlineTransformer()
        assert transformer.should_apply(simple_code_cells) is True

    def test_should_apply_false(self, notebook_with_magic):
        """Test should_apply returns False when magic already present."""
        transformer = MatplotlibInlineTransformer()
        assert transformer.should_apply(notebook_with_magic) is False

    def test_transform_adds_magic(self, simple_code_cells):
        """Test transformation adds %matplotlib inline."""
        transformer = MatplotlibInlineTransformer()
        result, stats = transformer.transform(simple_code_cells)

        assert len(result) == len(simple_code_cells) + 1
        assert stats["magic_added"] == 1
        assert has_matplotlib_magic(result) is True

    def test_transform_skips_when_present(self, notebook_with_magic):
        """Test transformation skips when magic already present."""
        transformer = MatplotlibInlineTransformer()
        result, stats = transformer.transform(notebook_with_magic)

        assert len(result) == len(notebook_with_magic)
        assert stats["magic_added"] == 0

    def test_transform_immutability(self, simple_code_cells):
        """Test transformation doesn't mutate input."""
        transformer = MatplotlibInlineTransformer()
        original_len = len(simple_code_cells)
        result, _ = transformer.transform(simple_code_cells)

        # Original should be unchanged
        assert len(simple_code_cells) == original_len
        # Result should be new list
        assert result is not simple_code_cells

    def test_transform_inserts_at_first_code_cell(self):
        """Test magic inserted before first code cell."""
        cells = [
            {"cell_type": "markdown", "source": ["# Title"]},
            {
                "cell_type": "code",
                "source": ["import numpy"],
                "execution_count": None,
                "metadata": {},
                "outputs": [],
            },
        ]
        transformer = MatplotlibInlineTransformer()
        result, _ = transformer.transform(cells)

        # Magic should be inserted at index 1 (before first code cell)
        assert result[1]["cell_type"] == "code"
        assert "%matplotlib inline" in "".join(result[1]["source"])


# ============================================================================
# IPythonDisplayImportTransformer Tests
# ============================================================================


class TestIPythonDisplayImportTransformer:
    """Test IPythonDisplayImportTransformer class."""

    def test_name(self):
        """Test transformer name."""
        transformer = IPythonDisplayImportTransformer()
        assert transformer.name() == "ipython_display_import"

    def test_description(self):
        """Test transformer description."""
        transformer = IPythonDisplayImportTransformer()
        assert "display" in transformer.description().lower()

    def test_should_apply_true(self, notebook_with_display):
        """Test should_apply when display used but not imported."""
        transformer = IPythonDisplayImportTransformer()
        assert transformer.should_apply(notebook_with_display) is True

    def test_should_apply_false_no_usage(self, simple_code_cells):
        """Test should_apply when display not used."""
        transformer = IPythonDisplayImportTransformer()
        assert transformer.should_apply(simple_code_cells) is False

    def test_should_apply_false_already_imported(self, notebook_with_display_import):
        """Test should_apply when already imported."""
        transformer = IPythonDisplayImportTransformer()
        assert transformer.should_apply(notebook_with_display_import) is False

    def test_transform_adds_import(self, notebook_with_display):
        """Test transformation adds import."""
        transformer = IPythonDisplayImportTransformer()
        result, stats = transformer.transform(notebook_with_display)

        assert len(result) == len(notebook_with_display) + 1
        assert stats["import_added"] == 1
        assert has_ipython_display_import(result) is True

    def test_transform_skips_when_not_needed(self, simple_code_cells):
        """Test transformation skips when display not used."""
        transformer = IPythonDisplayImportTransformer()
        result, stats = transformer.transform(simple_code_cells)

        assert len(result) == len(simple_code_cells)
        assert stats["import_added"] == 0

    def test_transform_skips_when_already_imported(self, notebook_with_display_import):
        """Test transformation skips when import already present."""
        transformer = IPythonDisplayImportTransformer()
        result, stats = transformer.transform(notebook_with_display_import)

        assert len(result) == len(notebook_with_display_import)
        assert stats["import_added"] == 0

    def test_transform_inserts_after_matplotlib(self, notebook_with_display):
        """Test import inserted after %matplotlib inline."""
        transformer = IPythonDisplayImportTransformer()
        result, _ = transformer.transform(notebook_with_display)

        # Find matplotlib magic
        magic_idx = find_cell_with_pattern(result, "%matplotlib inline")
        # Import should be right after
        assert magic_idx is not None
        import_cell = result[magic_idx + 1]
        assert "from IPython.display import display" in "".join(import_cell["source"])

    def test_transform_immutability(self, notebook_with_display):
        """Test transformation doesn't mutate input."""
        transformer = IPythonDisplayImportTransformer()
        original_len = len(notebook_with_display)
        result, _ = transformer.transform(notebook_with_display)

        assert len(notebook_with_display) == original_len
        assert result is not notebook_with_display


# ============================================================================
# PltShowReplacementTransformer Tests
# ============================================================================


class TestPltShowReplacement:
    """Test plt.show() replacement logic."""

    def test_find_figure_variable_with_assignment(self):
        """Test finding figure variable from assignment."""
        source = [
            "import matplotlib.pyplot as plt\n",
            "fig = plt.figure()\n",
            "plt.plot([1, 2, 3])\n",
            "plt.show()\n",
        ]
        fig_var = find_figure_variable(source, 3)
        assert fig_var == "fig"

    def test_find_figure_variable_with_subplots(self):
        """Test finding figure variable from subplots."""
        source = [
            "import matplotlib.pyplot as plt\n",
            "fig, ax = plt.subplots()\n",
            "ax.plot([1, 2, 3])\n",
            "plt.show()\n",
        ]
        fig_var = find_figure_variable(source, 3)
        assert fig_var == "fig"

    def test_find_figure_variable_fallback(self):
        """Test fallback when no figure variable found."""
        source = [
            "import matplotlib.pyplot as plt\n",
            "plt.plot([1, 2, 3])\n",
            "plt.show()\n",
        ]
        fig_var = find_figure_variable(source, 2)
        assert fig_var == "plt.gcf()"

    def test_replace_plt_show_basic(self):
        """Test basic plt.show() replacement."""
        source = ["plt.show()"]
        modified, count = replace_plt_show(source)

        assert count == 1
        assert "plt.tight_layout()" in "".join(modified)
        assert "display(" in "".join(modified)
        assert "plt.close(" in "".join(modified)

    def test_replace_plt_show_with_indentation(self):
        """Test replacement preserves indentation."""
        source = ["    plt.show()"]
        modified, count = replace_plt_show(source)

        assert count == 1
        # Check all lines have same indentation
        for line in modified:
            if line.strip():
                assert line.startswith("    ")

    def test_replace_plt_show_in_comment(self):
        """Test replacement skips comments."""
        source = ["# plt.show()"]
        modified, count = replace_plt_show(source)

        assert count == 0
        assert modified == source

    def test_replace_plt_show_in_string(self):
        """Test replacement skips string literals."""
        source = ['print("plt.show()")']
        modified, count = replace_plt_show(source)

        assert count == 0
        assert modified == source

    def test_replace_plt_show_multiple(self):
        """Test replacing multiple occurrences."""
        source = [
            "plt.figure()\n",
            "plt.plot([1, 2, 3])\n",
            "plt.show()\n",
            "plt.figure()\n",
            "plt.plot([4, 5, 6])\n",
            "plt.show()\n",
        ]
        modified, count = replace_plt_show(source)

        assert count == 2
        assert "".join(modified).count("display(") == 2
        assert "".join(modified).count("plt.close(") == 2

    def test_replace_plt_show_complex_line(self):
        """Test replacement skips complex cases."""
        source = ["result = plt.show() or something"]
        _modified, count = replace_plt_show(source)

        # Should not replace complex cases
        assert count == 0


class TestPltShowReplacementTransformer:
    """Test PltShowReplacementTransformer class."""

    def test_name(self):
        """Test transformer name."""
        transformer = PltShowReplacementTransformer()
        assert transformer.name() == "plt_show_replacement"

    def test_description(self):
        """Test transformer description."""
        transformer = PltShowReplacementTransformer()
        assert "plt.show()" in transformer.description()

    def test_should_apply(self):
        """Test should_apply always returns True."""
        transformer = PltShowReplacementTransformer()
        assert transformer.should_apply([]) is True

    def test_transform_replaces_plt_show(self, notebook_with_matplotlib):
        """Test transformation replaces plt.show()."""
        transformer = PltShowReplacementTransformer()
        result, stats = transformer.transform(notebook_with_matplotlib)

        assert stats["replacements"] == 1
        assert stats["cells_modified"] == 1

        # Check that display() is now in the notebook
        has_display = any(
            "display(" in "".join(cell.get("source", []))
            for cell in result
            if cell.get("cell_type") == "code"
        )
        assert has_display is True

    def test_transform_handles_string_source(self):
        """Test transformation handles source as string."""
        cells = [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "plt.show()",  # String instead of list
            }
        ]
        transformer = PltShowReplacementTransformer()
        _result, stats = transformer.transform(cells)

        assert stats["replacements"] == 1

    def test_transform_preserves_non_code_cells(self):
        """Test transformation preserves markdown cells."""
        cells = [
            {"cell_type": "markdown", "source": ["# Title"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["plt.show()"],
            },
        ]
        transformer = PltShowReplacementTransformer()
        result, _ = transformer.transform(cells)

        assert result[0]["cell_type"] == "markdown"
        assert result[0]["source"] == ["# Title"]

    def test_transform_immutability(self, notebook_with_matplotlib):
        """Test transformation doesn't mutate input."""
        transformer = PltShowReplacementTransformer()
        original_len = len(notebook_with_matplotlib)
        result, _ = transformer.transform(notebook_with_matplotlib)

        assert len(notebook_with_matplotlib) == original_len
        assert result is not notebook_with_matplotlib


# ============================================================================
# TransformationPipeline Tests
# ============================================================================


class TestTransformationPipeline:
    """Test TransformationPipeline class."""

    def test_pipeline_construction(self):
        """Test pipeline can be constructed with transformers."""
        transformers = [
            MatplotlibInlineTransformer(),
            IPythonDisplayImportTransformer(),
        ]
        pipeline = TransformationPipeline(transformers)
        assert len(pipeline.get_transformers()) == 2

    def test_pipeline_describe(self):
        """Test pipeline description."""
        transformers = [MatplotlibInlineTransformer()]
        pipeline = TransformationPipeline(transformers)
        description = pipeline.describe()

        assert len(description) == 1
        assert "name" in description[0]
        assert "description" in description[0]

    def test_pipeline_add_transformer(self):
        """Test adding transformer to pipeline."""
        pipeline = TransformationPipeline([])
        pipeline.add_transformer(MatplotlibInlineTransformer())

        assert len(pipeline.get_transformers()) == 1

    def test_pipeline_run_single_transformer(self, temp_notebook_file):
        """Test running pipeline with single transformer."""
        pipeline = TransformationPipeline([MatplotlibInlineTransformer()])
        stats = pipeline.run(temp_notebook_file, dry_run=True)

        assert "matplotlib_inline" in stats
        assert stats["matplotlib_inline"]["magic_added"] == 1

    def test_pipeline_run_multiple_transformers(self, temp_notebook_file):
        """Test running pipeline with multiple transformers."""
        transformers = [
            MatplotlibInlineTransformer(),
            IPythonDisplayImportTransformer(),
        ]
        pipeline = TransformationPipeline(transformers)
        stats = pipeline.run(temp_notebook_file, dry_run=True)

        assert "matplotlib_inline" in stats
        assert "ipython_display_import" in stats

    def test_pipeline_dry_run(self, temp_notebook_file):
        """Test dry run doesn't modify file."""
        # Read original content
        with open(temp_notebook_file) as f:
            original = f.read()

        pipeline = TransformationPipeline([MatplotlibInlineTransformer()])
        pipeline.run(temp_notebook_file, dry_run=True)

        # File should be unchanged
        with open(temp_notebook_file) as f:
            after = f.read()

        assert original == after

    def test_pipeline_backup(self, temp_notebook_file):
        """Test backup file creation."""
        pipeline = TransformationPipeline([MatplotlibInlineTransformer()])
        pipeline.run(temp_notebook_file, backup=True)

        backup_file = temp_notebook_file.with_suffix(temp_notebook_file.suffix + ".bak")
        assert backup_file.exists()

    def test_pipeline_skips_when_should_apply_false(self, temp_notebook_file):
        """Test pipeline skips transformer when should_apply returns False."""
        # First run adds magic
        pipeline = TransformationPipeline([MatplotlibInlineTransformer()])
        pipeline.run(temp_notebook_file)

        # Second run should skip
        stats = pipeline.run(temp_notebook_file, dry_run=True)
        assert stats["matplotlib_inline"] == {}

    def test_pipeline_validates_results(self, temp_notebook_file):
        """Test pipeline validates transformation results."""

        class BadTransformer(NotebookTransformer):
            def transform(self, cells):
                return "not a list", {}  # Invalid return type

            def name(self):
                return "bad"

            def description(self):
                return "Bad transformer"

        pipeline = TransformationPipeline([BadTransformer()])

        with pytest.raises(ValueError, match="Result must be a list"):
            pipeline.run(temp_notebook_file, dry_run=True)


# ============================================================================
# ProcessingTracker Tests
# ============================================================================


class TestProcessingTracker:
    """Test ProcessingTracker class."""

    def test_tracker_initialization(self, tmp_path):
        """Test tracker initialization."""
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)

        assert tracker.state_file == state_file
        assert isinstance(tracker.state, dict)

    def test_tracker_needs_processing_new_file(self, tmp_path, temp_notebook_file):
        """Test needs_processing returns True for new file."""
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)

        assert (
            tracker.needs_processing(temp_notebook_file, ["matplotlib_inline"]) is True
        )

    def test_tracker_mark_processed(self, tmp_path, temp_notebook_file):
        """Test marking file as processed."""
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)

        tracker.mark_processed(temp_notebook_file, ["matplotlib_inline"])

        # State file should exist
        assert state_file.exists()

        # State should contain entry (either relative or absolute path)
        try:
            rel_path = str(temp_notebook_file.relative_to(Path.cwd()))
        except ValueError:
            # File not relative to cwd (e.g., temp file), use absolute
            rel_path = str(temp_notebook_file.absolute())

        assert rel_path in tracker.state

    def test_tracker_needs_processing_after_mark(self, tmp_path, temp_notebook_file):
        """Test needs_processing returns False after marking."""
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)

        tracker.mark_processed(temp_notebook_file, ["matplotlib_inline"])
        needs = tracker.needs_processing(temp_notebook_file, ["matplotlib_inline"])

        assert needs is False

    def test_tracker_needs_processing_after_file_change(
        self, tmp_path, temp_notebook_file
    ):
        """Test needs_processing returns True after file changes."""
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)

        # Mark as processed
        tracker.mark_processed(temp_notebook_file, ["matplotlib_inline"])

        # Modify file
        with open(temp_notebook_file) as f:
            notebook = json.load(f)
        notebook["cells"].append(
            {
                "cell_type": "code",
                "source": ["# New cell"],
                "execution_count": None,
                "metadata": {},
                "outputs": [],
            }
        )
        with open(temp_notebook_file, "w") as f:
            json.dump(notebook, f)

        # Should need processing again
        needs = tracker.needs_processing(temp_notebook_file, ["matplotlib_inline"])
        assert needs is True

    def test_tracker_needs_processing_different_transformations(
        self, tmp_path, temp_notebook_file
    ):
        """Test needs_processing returns True for different transformations."""
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)

        tracker.mark_processed(temp_notebook_file, ["matplotlib_inline"])
        needs = tracker.needs_processing(
            temp_notebook_file, ["matplotlib_inline", "ipython_display_import"]
        )

        assert needs is True

    def test_tracker_clear(self, tmp_path, temp_notebook_file):
        """Test clearing tracker state."""
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)

        tracker.mark_processed(temp_notebook_file, ["matplotlib_inline"])
        tracker.clear()

        assert len(tracker.state) == 0
        assert not state_file.exists()

    def test_tracker_get_stats(self, tmp_path, temp_notebook_file):
        """Test getting tracker statistics."""
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)

        tracker.mark_processed(temp_notebook_file, ["matplotlib_inline"])
        stats = tracker.get_stats()

        assert stats["total_tracked"] == 1
        assert stats["state_file"] == str(state_file)
        assert stats["state_file_exists"] is True

    def test_tracker_persistence(self, tmp_path, temp_notebook_file):
        """Test tracker state persists across instances."""
        state_file = tmp_path / ".notebook_transforms.json"

        # First instance
        tracker1 = ProcessingTracker(state_file)
        tracker1.mark_processed(temp_notebook_file, ["matplotlib_inline"])

        # Second instance should load state
        tracker2 = ProcessingTracker(state_file)
        needs = tracker2.needs_processing(temp_notebook_file, ["matplotlib_inline"])

        assert needs is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_on_notebook(self, tmp_path):
        """Test full pipeline on a realistic notebook."""
        # Create notebook with matplotlib code
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Example Notebook"],
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import matplotlib.pyplot as plt"],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "fig, ax = plt.subplots()\n",
                        "ax.plot([1, 2, 3])\n",
                        "display(fig)\n",
                        "plt.show()",
                    ],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        notebook_path = tmp_path / "example.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        # Run full pipeline
        transformers = [
            MatplotlibInlineTransformer(),
            IPythonDisplayImportTransformer(),
            PltShowReplacementTransformer(),
        ]
        pipeline = TransformationPipeline(transformers)
        stats = pipeline.run(notebook_path)

        # Verify all transformations applied
        assert stats["matplotlib_inline"]["magic_added"] == 1
        assert stats["ipython_display_import"]["import_added"] == 1
        assert stats["plt_show_replacement"]["replacements"] == 1

        # Load and verify notebook
        with open(notebook_path) as f:
            result = json.load(f)

        cells = result["cells"]
        sources = "".join(
            "".join(cell.get("source", []))
            for cell in cells
            if cell.get("cell_type") == "code"
        )

        assert "%matplotlib inline" in sources
        assert "from IPython.display import display" in sources
        assert "plt.tight_layout()" in sources
        assert "plt.close(" in sources

    def test_incremental_processing(self, tmp_path):
        """Test incremental processing with tracker."""
        # Create notebook
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import numpy as np"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        notebook_path = tmp_path / "test.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        # Initialize components
        state_file = tmp_path / ".notebook_transforms.json"
        tracker = ProcessingTracker(state_file)
        pipeline = TransformationPipeline([MatplotlibInlineTransformer()])
        transform_names = ["matplotlib_inline"]

        # First run - should process
        assert tracker.needs_processing(notebook_path, transform_names) is True
        stats = pipeline.run(notebook_path)
        tracker.mark_processed(notebook_path, transform_names, stats)

        # Second run - should skip
        assert tracker.needs_processing(notebook_path, transform_names) is False

        # Modify notebook
        with open(notebook_path) as f:
            notebook = json.load(f)
        notebook["cells"].append(
            {
                "cell_type": "code",
                "source": ["x = 1"],
                "execution_count": None,
                "metadata": {},
                "outputs": [],
            }
        )
        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        # Third run - should process again due to changes
        assert tracker.needs_processing(notebook_path, transform_names) is True
