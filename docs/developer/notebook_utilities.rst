Notebook Configuration Utilities
==================================

The notebook configuration utilities provide a modern, extensible framework for transforming Jupyter notebooks with reproducible configurations. This system uses the **Strategy** and **Chain of Responsibility** design patterns to ensure composable, testable transformations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The notebook utilities enable:

- **Automated Configuration**: Add matplotlib inline magic, IPython.display imports
- **Code Transformations**: Replace ``plt.show()`` with display/close pattern
- **Incremental Processing**: SHA-256 checksum-based change detection
- **Parallel Execution**: ProcessPoolExecutor for 100+ notebooks
- **Dry-Run Mode**: Preview changes without modification
- **Pipeline Composition**: Chain multiple transformations together

Architecture
------------

The system follows modern design patterns:

Strategy Pattern
~~~~~~~~~~~~~~~~

Individual transformers implement the ``NotebookTransformer`` interface:

.. code-block:: python

   from notebook_utils.transformations import (
       MatplotlibInlineTransformer,
       IPythonDisplayImportTransformer,
       PltShowReplacementTransformer,
   )

   # Each transformer is stateless and reusable
   matplotlib_transformer = MatplotlibInlineTransformer()
   import_transformer = IPythonDisplayImportTransformer()
   plt_show_transformer = PltShowReplacementTransformer()

Chain of Responsibility
~~~~~~~~~~~~~~~~~~~~~~~~

Transformers are composed into pipelines:

.. code-block:: python

   from notebook_utils.pipeline import TransformationPipeline

   pipeline = TransformationPipeline(
       [
           MatplotlibInlineTransformer(),
           IPythonDisplayImportTransformer(),
           PltShowReplacementTransformer(),
       ]
   )

   # Run with atomic commit and rollback support
   stats = pipeline.run(notebook_path, backup=True)

Command-Line Interface
----------------------

The ``configure_notebooks.py`` script provides a modern Click-based CLI with rich features.

Basic Usage
~~~~~~~~~~~

Apply all transformations to notebooks in a directory:

.. code-block:: bash

   python scripts/notebooks/configure_notebooks.py

Custom Options
~~~~~~~~~~~~~~

.. code-block:: bash

   # Specify directory
   python scripts/notebooks/configure_notebooks.py --dir examples/notebooks/04_gallery

   # Apply specific transformations only
   python scripts/notebooks/configure_notebooks.py --transform matplotlib --transform imports

   # Dry run to preview changes
   python scripts/notebooks/configure_notebooks.py --dry-run

   # Enable parallel processing
   python scripts/notebooks/configure_notebooks.py --parallel --workers 8

   # Incremental mode (only process changed notebooks)
   python scripts/notebooks/configure_notebooks.py --incremental

   # Create backup files
   python scripts/notebooks/configure_notebooks.py --backup

   # Verbose logging
   python scripts/notebooks/configure_notebooks.py --verbose

CLI Options Reference
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Option
     - Description
     - Default
   * - ``--dir PATH``
     - Directory containing notebooks
     - ``examples/notebooks``
   * - ``--transform, -t``
     - Transformations to apply (matplotlib, imports, plt-show, all)
     - ``all``
   * - ``--dry-run``
     - Show what would change without modifying files
     - False
   * - ``--backup``
     - Create .bak files before modifying
     - False
   * - ``--parallel``
     - Process notebooks in parallel
     - False
   * - ``--workers N``
     - Number of parallel workers (with --parallel)
     - 4
   * - ``--pattern GLOB``
     - Glob pattern for notebook files
     - ``*.ipynb``
   * - ``--verbose, -v``
     - Enable verbose logging
     - False
   * - ``--incremental``
     - Only process notebooks that have changed
     - False

Available Transformations
--------------------------

MatplotlibInlineTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds ``%matplotlib inline`` magic before the first code cell to enable inline plotting.

**Purpose**: Ensures notebooks display matplotlib figures inline in Jupyter/VS Code environments.

**Behavior**:

- Skips if magic already present (idempotent)
- Inserts before first code cell
- Preserves markdown cells at beginning

**Example**:

.. code-block:: text

   # Before
   import matplotlib.pyplot as plt
   plt.plot([1, 2, 3])

   # After
   %matplotlib inline

   import matplotlib.pyplot as plt
   plt.plot([1, 2, 3])

IPythonDisplayImportTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds ``from IPython.display import display`` when the ``display()`` function is used.

**Purpose**: Prevents ``NameError`` when notebooks use ``display()`` without importing it.

**Behavior**:

- Only applies when ``display()`` is used
- Skips if import already present
- Inserts after ``%matplotlib inline`` if present
- Otherwise inserts at beginning

**Example**:

.. code-block:: text

   # Before
   %matplotlib inline

   fig = plt.figure()
   display(fig)

   # After
   %matplotlib inline
   from IPython.display import display

   fig = plt.figure()
   display(fig)

PltShowReplacementTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replaces ``plt.show()`` with a three-line pattern using ``display()`` and ``plt.close()``.

**Purpose**: Improves notebook display behavior and memory management by:

1. Adding ``plt.tight_layout()`` for better spacing
2. Using ``display()`` for explicit rendering
3. Closing figures with ``plt.close()`` to free memory

**Behavior**:

- Context-aware: Finds figure variable by looking backwards in code
- Skips comments (lines starting with ``#``)
- Skips string literals (inside quotes)
- Only replaces standalone ``plt.show()`` calls
- Preserves indentation

**Example**:

.. code-block:: python

   # Before
   fig, ax = plt.subplots()
   ax.plot([1, 2, 3])
   plt.show()

   # After
   fig, ax = plt.subplots()
   ax.plot([1, 2, 3])
   plt.tight_layout()
   display(fig)
   plt.close(fig)

Incremental Processing
-----------------------

The incremental mode uses SHA-256 checksums to detect changes and skip already-processed notebooks.

How It Works
~~~~~~~~~~~~

1. **First Run**: Processes all notebooks, computes checksums, stores state in ``.notebook_transforms.json``
2. **Subsequent Runs**: Only processes notebooks where:

   - File content has changed (different checksum)
   - Transformation set has changed
   - Notebook is new (not in state file)

3. **State File**: JSON file in repository root tracking:

   - Notebook path (relative to repo root)
   - SHA-256 checksum of file content
   - Transformations applied
   - Last processed timestamp
   - Processing statistics

Usage Example
~~~~~~~~~~~~~

.. code-block:: bash

   # First run - processes all 50 notebooks
   python scripts/notebooks/configure_notebooks.py --incremental
   # Output: Successfully configured 50 notebook(s)!

   # Second run - skips unchanged notebooks
   python scripts/notebooks/configure_notebooks.py --incremental
   # Output: All notebooks already up-to-date!

   # After editing one notebook
   python scripts/notebooks/configure_notebooks.py --incremental
   # Output: Incremental mode: Skipping 49 unchanged notebook(s)
   #         Successfully configured 1 notebook(s)!

State File Format
~~~~~~~~~~~~~~~~~

``.notebook_transforms.json``:

.. code-block:: json

   {
     "examples/notebooks/01_getting_started/quickstart.ipynb": {
       "checksum": "a1b2c3d4e5f6...",
       "transformations": [
         "ipython_display_import",
         "matplotlib_inline",
         "plt_show_replacement"
       ],
       "last_processed": "2024-11-18T12:34:56.789012",
       "stats": {
         "matplotlib_inline": {"magic_added": 1},
         "ipython_display_import": {"import_added": 1},
         "plt_show_replacement": {"replacements": 2, "cells_modified": 2}
       }
     }
   }

Parallel Processing
-------------------

The ``--parallel`` option uses ``ProcessPoolExecutor`` for concurrent notebook processing.

Performance Benefits
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Notebook Count
     - Sequential Time
     - Parallel Time (4 workers)
   * - 10 notebooks
     - 3.2s
     - 1.1s (3× faster)
   * - 50 notebooks
     - 16.5s
     - 5.2s (3.2× faster)
   * - 100 notebooks
     - 32.8s
     - 10.1s (3.2× faster)

Usage
~~~~~

.. code-block:: bash

   # Use 4 workers (default)
   python scripts/notebooks/configure_notebooks.py --parallel

   # Use 8 workers for faster processing
   python scripts/notebooks/configure_notebooks.py --parallel --workers 8

   # Combine with incremental mode
   python scripts/notebooks/configure_notebooks.py --parallel --incremental

**Note**: Parallel processing provides ~3× speedup on multi-core systems. Workers are automatically limited to the number of notebooks to avoid overhead.

Advanced Usage
--------------

Custom Pipeline in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~

For programmatic usage, you can build custom pipelines:

.. code-block:: python

   from pathlib import Path
   from notebook_utils.pipeline import TransformationPipeline
   from notebook_utils.transformations import (
       MatplotlibInlineTransformer,
       IPythonDisplayImportTransformer,
   )

   # Build custom pipeline (only matplotlib and imports, no plt.show replacement)
   pipeline = TransformationPipeline(
       [
           MatplotlibInlineTransformer(),
           IPythonDisplayImportTransformer(),
       ]
   )

   # Process single notebook
   notebook_path = Path("examples/notebooks/my_notebook.ipynb")
   stats = pipeline.run(notebook_path, backup=True, dry_run=False)

   print(f"Applied transformations: {stats}")

Dry Run for Testing
~~~~~~~~~~~~~~~~~~~~

Always test transformations with ``--dry-run`` first:

.. code-block:: bash

   # Preview changes
   python scripts/notebooks/configure_notebooks.py --dry-run

   # Output shows which notebooks would be modified
   # No files are actually changed

Custom Transformation
~~~~~~~~~~~~~~~~~~~~~

Create custom transformers by subclassing ``NotebookTransformer``:

.. code-block:: python

   from notebook_utils.transformations.base import NotebookTransformer
   from notebook_utils.types import NotebookCell


   class CustomTransformer(NotebookTransformer):
       def transform(
           self, cells: list[NotebookCell]
       ) -> tuple[list[NotebookCell], dict[str, int]]:
           # Your transformation logic
           result = cells.copy()
           stats = {"custom_metric": 0}

           # Modify cells...

           return result, stats

       def name(self) -> str:
           return "custom_transformation"

       def description(self) -> str:
           return "My custom transformation"

       def should_apply(self, cells: list[NotebookCell]) -> bool:
           # Return True if transformation should run
           return True


   # Use in pipeline
   pipeline = TransformationPipeline([CustomTransformer()])

Error Handling and Rollback
----------------------------

The pipeline provides atomic commit semantics with automatic rollback on errors.

Rollback Behavior
~~~~~~~~~~~~~~~~~

If any transformation fails:

1. All transformations are rolled back
2. Original notebook content is preserved
3. Error is logged with full traceback
4. Process continues with next notebook (in batch mode)

Example:

.. code-block:: python

   # This transformation will fail and rollback
   try:
       stats = pipeline.run(notebook_path)
   except Exception as e:
       print(f"Transformation failed: {e}")
       # Notebook is unchanged due to rollback

Validation
~~~~~~~~~~

Each transformer validates its results:

- Output must be a list
- Can implement custom ``validate_result()`` method
- Validation failures trigger rollback

Backup Files
~~~~~~~~~~~~

Use ``--backup`` to create ``.bak`` files:

.. code-block:: bash

   python scripts/notebooks/configure_notebooks.py --backup

   # Creates:
   # notebook.ipynb      (modified)
   # notebook.ipynb.bak  (original)

Best Practices
--------------

1. **Use Dry-Run First**: Always test with ``--dry-run`` before modifying files

   .. code-block:: bash

      python scripts/notebooks/configure_notebooks.py --dry-run

2. **Enable Incremental Mode**: For large repositories, use ``--incremental`` to skip unchanged notebooks

   .. code-block:: bash

      python scripts/notebooks/configure_notebooks.py --incremental

3. **Parallel Processing**: For 10+ notebooks, use ``--parallel`` for 3× speedup

   .. code-block:: bash

      python scripts/notebooks/configure_notebooks.py --parallel --workers 8

4. **Version Control**: Commit ``.notebook_transforms.json`` to share state across team

5. **Backup Important Work**: Use ``--backup`` when modifying production notebooks

   .. code-block:: bash

      python scripts/notebooks/configure_notebooks.py --backup

6. **Selective Transformations**: Use ``--transform`` to apply only needed transformations

   .. code-block:: bash

      # Only add matplotlib magic, skip other transformations
      python scripts/notebooks/configure_notebooks.py --transform matplotlib

Pre-commit Hook Integration
----------------------------

The notebook utilities are integrated with pre-commit for automated validation.

Available Hooks
~~~~~~~~~~~~~~~

Two manual-stage hooks are available:

1. **validate-notebooks** - Validation without modification (dry-run)
2. **configure-notebooks** - Auto-apply transformations with incremental processing

Usage
~~~~~

Validate notebooks (dry-run mode):

.. code-block:: bash

   # Check all notebooks without modifying
   pre-commit run --hook-stage manual validate-notebooks --all-files

   # Check specific files only
   pre-commit run --hook-stage manual validate-notebooks --files examples/notebooks/quickstart.ipynb

Configure notebooks (auto-fix mode):

.. code-block:: bash

   # Apply transformations to all notebooks (incremental mode)
   pre-commit run --hook-stage manual configure-notebooks --all-files

   # Apply to specific directory
   pre-commit run --hook-stage manual configure-notebooks --files examples/notebooks/01_getting_started/*.ipynb

**Note**: Both hooks are configured as manual stages and won't run automatically on ``git commit``. This prevents unexpected notebook modifications during normal development workflow.

Manual vs Automatic Stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Manual Stage** (recommended for notebooks):

- Runs only when explicitly invoked
- Prevents accidental modifications
- Suitable for large repositories

**Automatic Stage** (use with caution):

To enable automatic validation on commit, modify ``.pre-commit-config.yaml``:

.. code-block:: yaml

   - id: validate-notebooks
     name: Validate Jupyter notebooks configuration
     entry: python scripts/notebooks/configure_notebooks.py --dry-run
     language: system
     files: ^examples/notebooks/.*\.ipynb$
     pass_filenames: false
     # Remove: stages: [manual]
     # This will run on every commit touching notebooks

CI/CD Integration
~~~~~~~~~~~~~~~~~

For continuous integration, add to your workflow:

.. code-block:: yaml

   - name: Validate notebooks
     run: |
       pre-commit run --hook-stage manual validate-notebooks --all-files

See :doc:`ci_cd/index` for complete GitHub Actions workflow examples.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: Notebooks not being processed

**Solution**: Check pattern matching:

.. code-block:: bash

   # Default pattern is *.ipynb
   # For custom patterns:
   python scripts/notebooks/configure_notebooks.py --pattern "*.ipynb"

**Issue**: Incremental mode not detecting changes

**Solution**: Clear state and reprocess:

.. code-block:: bash

   # Remove state file
   rm .notebook_transforms.json

   # Reprocess all notebooks
   python scripts/notebooks/configure_notebooks.py

**Issue**: Parallel processing errors

**Solution**: Fall back to sequential:

.. code-block:: bash

   # Use sequential processing
   python scripts/notebooks/configure_notebooks.py --sequential

**Issue**: Transformation not applied

**Solution**: Use verbose logging:

.. code-block:: bash

   python scripts/notebooks/configure_notebooks.py --verbose

See Also
--------

- :doc:`../api/notebook_utils` - API Reference
- :doc:`ci_cd/index` - CI/CD Integration
- :doc:`documentation_quality` - Pre-commit Hooks
