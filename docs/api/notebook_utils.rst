Notebook Utilities API Reference
==================================

The ``notebook_utils`` package provides extensible notebook transformation utilities following the **Strategy** and **Chain of Responsibility** design patterns.

.. currentmodule:: notebook_utils

.. contents:: Table of Contents
   :local:
   :depth: 2

Module Overview
---------------

The package is organized into several modules:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Purpose
   * - :mod:`notebook_utils.types`
     - TypedDict definitions for type safety
   * - :mod:`notebook_utils.cells`
     - Cell manipulation utilities
   * - :mod:`notebook_utils.core`
     - Robust I/O operations with validation
   * - :mod:`notebook_utils.transformations.base`
     - Abstract base class for transformers
   * - :mod:`notebook_utils.transformations.matplotlib`
     - Matplotlib inline configuration
   * - :mod:`notebook_utils.transformations.imports`
     - IPython.display import injection
   * - :mod:`notebook_utils.transformations.plt_show`
     - plt.show() replacement logic
   * - :mod:`notebook_utils.pipeline`
     - Transformation pipeline orchestration
   * - :mod:`notebook_utils.tracking`
     - Incremental processing with checksums

Type Definitions
----------------

.. py:module:: notebook_utils.types

.. py:class:: NotebookCell

   TypedDict defining the structure of a Jupyter notebook cell.

   **Attributes**:

   .. py:attribute:: cell_type
      :type: Literal["code", "markdown", "raw"]

      Type of the cell

   .. py:attribute:: execution_count
      :type: int | None

      Execution count for code cells

   .. py:attribute:: metadata
      :type: dict

      Cell metadata dictionary

   .. py:attribute:: outputs
      :type: list

      List of cell outputs (for code cells)

   .. py:attribute:: source
      :type: str | list[str]

      Cell source code or markdown content

   **Example**:

   .. code-block:: python

      cell: NotebookCell = {
          "cell_type": "code",
          "execution_count": None,
          "metadata": {},
          "outputs": [],
          "source": ["import numpy as np"],
      }

.. py:class:: NotebookStats

   TypedDict for tracking transformation statistics.

   **Attributes**:

   .. py:attribute:: matplotlib_magic_added
      :type: int

      Number of %matplotlib inline magics added

   .. py:attribute:: ipython_display_import_added
      :type: int

      Number of IPython.display imports added

   .. py:attribute:: plt_show_replaced
      :type: int

      Number of plt.show() calls replaced

   .. py:attribute:: cells_modified
      :type: int

      Total number of cells modified

Cell Utilities
--------------

.. py:module:: notebook_utils.cells

.. py:function:: has_matplotlib_magic(cells: list[NotebookCell]) -> bool

   Check if notebook already has ``%matplotlib inline`` magic.

   :param cells: List of notebook cells
   :return: True if magic is present

   **Example**:

   .. code-block:: python

      if not has_matplotlib_magic(cells):
          # Add magic
          pass

.. py:function:: has_ipython_display_import(cells: list[NotebookCell]) -> bool

   Check if notebook imports ``display`` from ``IPython.display``.

   :param cells: List of notebook cells
   :return: True if import is present

.. py:function:: uses_display(cells: list[NotebookCell]) -> bool

   Check if notebook uses the ``display()`` function.

   :param cells: List of notebook cells
   :return: True if display() is called

.. py:function:: find_first_code_cell_index(cells: list[NotebookCell]) -> int

   Find index of first code cell in notebook.

   :param cells: List of notebook cells
   :return: Index of first code cell, or 0 if no code cells

.. py:function:: find_cell_with_pattern(cells: list[NotebookCell], pattern: str) -> int | None

   Find first cell containing a pattern.

   :param cells: List of notebook cells
   :param pattern: String pattern to search for
   :return: Index of first matching cell, or None

   **Example**:

   .. code-block:: python

      idx = find_cell_with_pattern(cells, "%matplotlib inline")
      if idx is not None:
          # Insert after matplotlib magic
          insert_idx = idx + 1

.. py:function:: create_matplotlib_config_cell() -> NotebookCell

   Create a cell with matplotlib inline configuration.

   :return: NotebookCell with %matplotlib inline magic

.. py:function:: create_ipython_display_import_cell() -> NotebookCell

   Create a cell with IPython.display import.

   :return: NotebookCell with display import statement

.. py:function:: cell_contains_pattern(cells: list[NotebookCell], pattern: str, cell_type: str = "code") -> bool

   Check if any cell contains a pattern.

   :param cells: List of notebook cells
   :param pattern: String pattern to search for
   :param cell_type: Type of cells to search (default: "code")
   :return: True if pattern found

Core I/O Operations
--------------------

.. py:module:: notebook_utils.core

Exception Classes
~~~~~~~~~~~~~~~~~

.. py:exception:: NotebookError

   Base exception for notebook operations.

.. py:exception:: NotebookReadError

   Raised when notebook cannot be read.

.. py:exception:: NotebookWriteError

   Raised when notebook cannot be written.

.. py:exception:: NotebookValidationError

   Raised when notebook structure is invalid.

Functions
~~~~~~~~~

.. py:function:: validate_notebook_structure(notebook: dict) -> None

   Validate notebook has required structure.

   :param notebook: Notebook dictionary
   :raises NotebookValidationError: If structure is invalid

   **Validates**:

   - ``nbformat`` field exists
   - ``cells`` field exists and is a list

.. py:function:: read_notebook(path: Path) -> dict | None

   Read and validate notebook from file with comprehensive error handling.

   :param path: Path to notebook file
   :return: Notebook dictionary, or None on error

   **Error Handling**:

   - Logs file not found errors
   - Logs JSON decode errors
   - Logs validation errors
   - Returns None on any error

   **Example**:

   .. code-block:: python

      from pathlib import Path

      notebook = read_notebook(Path("example.ipynb"))
      if notebook is None:
          print("Failed to read notebook")

.. py:function:: write_notebook(path: Path, notebook: dict, backup: bool = False) -> bool

   Write notebook to file with atomic write pattern.

   :param path: Path to notebook file
   :param notebook: Notebook dictionary
   :param backup: Create .bak file before writing
   :return: True if successful, False otherwise

   **Atomic Write Pattern**:

   1. Write to temporary file
   2. Validate write succeeded
   3. Move temporary file to target (atomic operation)

   **Example**:

   .. code-block:: python

      success = write_notebook(
          Path("example.ipynb"), notebook, backup=True  # Creates example.ipynb.bak
      )

Transformation Base Class
--------------------------

.. py:module:: notebook_utils.transformations.base

.. py:class:: NotebookTransformer

   Abstract base class for notebook transformations (Strategy pattern).

   Each transformer should be:

   - **Stateless**: Can be reused across multiple notebooks
   - **Idempotent**: Running twice produces same result as running once
   - **Pure**: Only modifies notebook cells, no side effects

   .. py:method:: transform(cells: list[NotebookCell]) -> tuple[list[NotebookCell], dict[str, int]]
      :abstractmethod:

      Transform notebook cells.

      :param cells: List of notebook cells to transform
      :return: Tuple of (modified_cells, stats_dict)

      **Note**: Should return NEW list, not mutate input cells

   .. py:method:: name() -> str
      :abstractmethod:

      Return unique transformation name.

      :return: Transformation identifier (e.g., "matplotlib_inline")

   .. py:method:: description() -> str
      :abstractmethod:

      Return human-readable transformation description.

      :return: Description of what this transformation does

   .. py:method:: should_apply(cells: list[NotebookCell]) -> bool

      Check if transformation should be applied.

      :param cells: Notebook cells to check
      :return: True if transformation should run

      **Default**: Returns True. Override to skip when not needed.

   .. py:method:: validate_result(original: list[NotebookCell], transformed: list[NotebookCell]) -> bool

      Validate transformation result.

      :param original: Original cells before transformation
      :param transformed: Cells after transformation
      :return: True if valid
      :raises ValueError: If validation fails

      **Default**: Checks result is a list. Override for custom validation.

Transformation Implementations
-------------------------------

MatplotlibInlineTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: notebook_utils.transformations.matplotlib

.. py:class:: MatplotlibInlineTransformer

   Bases: :py:class:`~notebook_utils.transformations.base.NotebookTransformer`

   Adds ``%matplotlib inline`` magic before first code cell.

   **Ensures**: Notebooks have matplotlib inline backend configured for proper display.

   .. py:method:: transform(cells: list[NotebookCell]) -> tuple[list[NotebookCell], dict[str, int]]

      Add %matplotlib inline magic if not present.

      :param cells: Notebook cells
      :return: Tuple of (modified_cells, {"magic_added": count})

   .. py:method:: name() -> str

      :return: "matplotlib_inline"

   .. py:method:: description() -> str

      :return: "Add %matplotlib inline magic for inline plotting"

   .. py:method:: should_apply(cells: list[NotebookCell]) -> bool

      :return: True if magic not already present

   **Example**:

   .. code-block:: python

      transformer = MatplotlibInlineTransformer()
      result, stats = transformer.transform(cells)
      print(f"Added {stats['magic_added']} magic(s)")

IPythonDisplayImportTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: notebook_utils.transformations.imports

.. py:class:: IPythonDisplayImportTransformer

   Bases: :py:class:`~notebook_utils.transformations.base.NotebookTransformer`

   Adds ``from IPython.display import display`` when ``display()`` is used.

   **Prevents**: NameError when notebooks use display() without importing it.

   .. py:method:: transform(cells: list[NotebookCell]) -> tuple[list[NotebookCell], dict[str, int]]

      Add IPython.display import if needed.

      :param cells: Notebook cells
      :return: Tuple of (modified_cells, {"import_added": count})

   .. py:method:: name() -> str

      :return: "ipython_display_import"

   .. py:method:: description() -> str

      :return: "Add IPython.display import when display() is used"

   .. py:method:: should_apply(cells: list[NotebookCell]) -> bool

      :return: True if display() used and import not present

PltShowReplacementTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: notebook_utils.transformations.plt_show

.. py:function:: find_figure_variable(source: list[str], show_line_idx: int) -> str

   Find figure variable name by looking backwards from ``plt.show()``.

   :param source: Source code lines
   :param show_line_idx: Index of line containing plt.show()
   :return: Figure variable name or "plt.gcf()" as fallback

   **Looks for patterns**:

   - ``fig = plt.figure()``
   - ``fig, ax = plt.subplots()``

.. py:function:: replace_plt_show(source: list[str]) -> tuple[list[str], int]

   Replace ``plt.show()`` with display pattern using context-aware logic.

   :param source: Source code lines
   :return: Tuple of (modified_source, num_replacements)

   **Replacement Pattern**:

   .. code-block:: python

      # Before
      plt.show()

      # After
      plt.tight_layout()
      display(fig)
      plt.close(fig)

   **Skips**:

   - Comments (lines starting with #)
   - String literals (inside quotes)
   - Complex statements (not standalone plt.show())

.. py:class:: PltShowReplacementTransformer

   Bases: :py:class:`~notebook_utils.transformations.base.NotebookTransformer`

   Replaces ``plt.show()`` calls with display/close pattern.

   **Improves**:

   1. Layout with ``plt.tight_layout()``
   2. Display with explicit ``display()``
   3. Memory management with ``plt.close()``

   .. py:method:: transform(cells: list[NotebookCell]) -> tuple[list[NotebookCell], dict[str, int]]

      Replace plt.show() in all code cells.

      :param cells: Notebook cells
      :return: Tuple of (modified_cells, {"replacements": count, "cells_modified": count})

   .. py:method:: name() -> str

      :return: "plt_show_replacement"

   .. py:method:: description() -> str

      :return: "Replace plt.show() with display/close pattern"

Pipeline Orchestration
-----------------------

.. py:module:: notebook_utils.pipeline

.. py:class:: TransformationPipeline

   Composes multiple transformations with rollback support (Chain of Responsibility pattern).

   **Provides**:

   - Atomic commit semantics
   - Automatic rollback on errors
   - Validation of all transformations
   - Statistics collection

   .. py:method:: __init__(transformers: list[NotebookTransformer])

      Initialize pipeline with transformers.

      :param transformers: List of transformers to apply in order

   .. py:method:: run(notebook_path: Path, backup: bool = False, dry_run: bool = False) -> dict[str, dict]

      Run all transformations with atomic commit.

      :param notebook_path: Path to notebook file
      :param backup: Create .bak file before writing
      :param dry_run: Don't write changes, just return stats
      :return: Dictionary mapping transformer names to their stats
      :raises Exception: If any transformation fails (with rollback)

      **Example**:

      .. code-block:: python

         from pathlib import Path
         from notebook_utils.pipeline import TransformationPipeline
         from notebook_utils.transformations import (
             MatplotlibInlineTransformer,
             IPythonDisplayImportTransformer,
         )

         pipeline = TransformationPipeline(
             [
                 MatplotlibInlineTransformer(),
                 IPythonDisplayImportTransformer(),
             ]
         )

         stats = pipeline.run(Path("example.ipynb"), backup=True, dry_run=False)

         print(stats)
         # {
         #   "matplotlib_inline": {"magic_added": 1},
         #   "ipython_display_import": {"import_added": 1}
         # }

   .. py:method:: get_transformers() -> list[NotebookTransformer]

      Get list of transformers in pipeline.

      :return: List of transformer instances (copy)

   .. py:method:: add_transformer(transformer: NotebookTransformer) -> None

      Add a transformer to the end of the pipeline.

      :param transformer: Transformer instance to add

   .. py:method:: describe() -> list[dict[str, str]]

      Get description of all transformers in pipeline.

      :return: List of dicts with 'name' and 'description' keys

Incremental Processing
-----------------------

.. py:module:: notebook_utils.tracking

.. py:class:: ProcessingTracker

   Tracks processed notebooks to enable incremental updates.

   **Uses**: SHA-256 checksums to detect changes. Stores state in ``.notebook_transforms.json``.

   .. py:method:: __init__(state_file: Path = None)

      Initialize tracker with state file.

      :param state_file: Path to state file (default: .notebook_transforms.json in current directory)

   .. py:method:: needs_processing(notebook_path: Path, transformations: list[str]) -> bool

      Check if notebook needs processing.

      :param notebook_path: Path to notebook
      :param transformations: List of transformation names to apply
      :return: True if notebook should be processed

      **Returns True if**:

      - Notebook is new (not in state)
      - File content changed (different checksum)
      - Transformation set changed

   .. py:method:: mark_processed(notebook_path: Path, transformations: list[str], stats: dict = None) -> None

      Mark notebook as processed.

      :param notebook_path: Path to notebook
      :param transformations: List of transformation names applied
      :param stats: Optional statistics from processing

      **Updates**:

      - Checksum (SHA-256 of file content)
      - Transformations list (sorted)
      - Last processed timestamp
      - Processing statistics

   .. py:method:: clear() -> None

      Clear all processing state.

      **Deletes**: State file and clears in-memory state.

   .. py:method:: get_stats() -> dict

      Get statistics about tracked notebooks.

      :return: Dictionary with tracking statistics

      **Example**:

      .. code-block:: python

         tracker = ProcessingTracker()
         stats = tracker.get_stats()
         print(stats)
         # {
         #   "total_tracked": 42,
         #   "state_file": "/path/to/.notebook_transforms.json",
         #   "state_file_exists": True
         # }

   **Example Usage**:

   .. code-block:: python

      from pathlib import Path
      from notebook_utils.tracking import ProcessingTracker
      from notebook_utils.pipeline import TransformationPipeline
      from notebook_utils.transformations import MatplotlibInlineTransformer

      # Initialize
      tracker = ProcessingTracker()
      pipeline = TransformationPipeline([MatplotlibInlineTransformer()])
      transform_names = ["matplotlib_inline"]

      # Process only if needed
      notebook_path = Path("example.ipynb")
      if tracker.needs_processing(notebook_path, transform_names):
          stats = pipeline.run(notebook_path)
          tracker.mark_processed(notebook_path, transform_names, stats)
      else:
          print("Notebook already up-to-date")

See Also
--------

- :doc:`../developer/notebook_utilities` - User Guide
- :doc:`../developer/ci_cd/index` - CI/CD Integration
- :ref:`genindex` - General Index
