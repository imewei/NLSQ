Configuration Reference
=======================

NLSQ is configured via environment variables (read once at import time) and
programmatic Python configuration objects. There is no auto-discovered
``nlsq.yaml`` or ``~/.config/nlsq/config.yaml`` project config file, and no
``Config`` class or ``config_context()`` helper.

.. note::

   The NLSQ **CLI** does support YAML *workflow* files (a different feature
   from what this page describes) — see :doc:`/howto/configure_yaml` for the
   ``paths`` / ``data`` / ``model`` / ``fitting`` / ``hybrid_streaming``
   workflow schema used by the CLI runner.

Environment Variables
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Variable
     - Effect
     - Default
   * - ``NLSQ_FORCE_CPU``
     - Force the JAX CPU backend
     - unset
   * - ``NLSQ_DISABLE_X64``
     - Skip enabling 64-bit precision in JAX
     - unset (x64 enabled)
   * - ``NLSQ_DISABLE_PERSISTENT_CACHE``
     - Disable the JAX compilation cache
     - unset (cache enabled)
   * - ``NLSQ_JAX_CACHE_DIR``
     - JAX compilation cache directory
     - ``~/.cache/nlsq/jax_cache``
   * - ``NLSQ_CACHE_MIN_COMPILE_TIME_SECS``
     - Minimum compile time (seconds) before caching
     - ``1``
   * - ``NLSQ_GPU_MEMORY_FRACTION``
     - Fraction of GPU memory XLA may claim (0.0-1.0)
     - unset (grows as needed)
   * - ``NLSQ_MEMORY_LIMIT_GB``
     - Default ``MemoryConfig.memory_limit_gb``
     - ``8.0``
   * - ``NLSQ_CHUNK_SIZE_MB``
     - Default ``MemoryConfig.chunk_size_mb``
     - auto
   * - ``NLSQ_OOM_STRATEGY``
     - ``MemoryConfig.out_of_memory_strategy`` (``fallback``, ``reduce``, or ``error``)
     - ``fallback``
   * - ``NLSQ_SAFETY_FACTOR``
     - ``MemoryConfig.safety_factor``
     - ``0.8``
   * - ``NLSQ_DISABLE_PROGRESS_REPORTING``
     - Disable progress reporting for large operations
     - unset (enabled)
   * - ``NLSQ_DISABLE_AUTO_SOLVER_SELECTION``
     - Disable automatic solver selection for large datasets
     - unset (enabled)
   * - ``NLSQ_JACOBIAN_MODE``
     - Force Jacobian AD mode: ``auto``, ``fwd``, or ``rev``
     - ``auto``
   * - ``NLSQ_SKIP_GPU_CHECK``
     - Suppress the startup GPU availability warning
     - unset
   * - ``NLSQ_DEBUG``
     - Enable debug-level logging
     - unset

**Example:**

.. code-block:: bash

   export NLSQ_MEMORY_LIMIT_GB=16
   export NLSQ_DEBUG=1
   python my_script.py

Programmatic Configuration
----------------------------

Memory settings
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.config import (
       MemoryConfig,
       set_memory_limits,
       get_memory_config,
       memory_context,
   )

   # Set process-wide memory limits
   set_memory_limits(memory_limit_gb=16.0, gpu_memory_fraction=0.8)

   # Inspect the current configuration
   config = get_memory_config()
   print(config.memory_limit_gb)

   # Temporarily override for a block of code
   with memory_context(MemoryConfig(memory_limit_gb=32.0)):
       ...  # runs with the temporary limit

Large-dataset settings
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.config import (
       configure_for_large_datasets,
       LargeDatasetConfig,
       large_dataset_context,
   )

   # One-shot setup for large-dataset workflows
   configure_for_large_datasets(memory_limit_gb=16.0, progress_reporting=True)

   # Or temporarily override solver-selection behavior
   with large_dataset_context(LargeDatasetConfig(enable_automatic_solver_selection=False)):
       ...

Jacobian AD mode
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.config import get_jacobian_mode, set_jacobian_mode

   mode, source = get_jacobian_mode()
   print(f"Using {mode} mode from {source}")

   set_jacobian_mode("rev")  # forces reverse-mode AD for this process

``get_jacobian_mode()`` resolves in this order:

1. ``NLSQ_JACOBIAN_MODE`` environment variable
2. ``jacobian_mode`` key in ``~/.nlsq/config.json`` (JSON, not YAML)
3. ``"auto"`` default

``set_jacobian_mode()`` only sets the environment variable for the current
process; to persist the choice, write it to ``~/.nlsq/config.json`` yourself.

Fit Presets and Workflows
----------------------------

The unified ``fit()`` entry point selects a strategy via ``workflow`` (new,
v0.6.3+) or the deprecated ``preset`` argument:

.. code-block:: python

   from nlsq import fit

   # workflow-based selection
   popt, pcov = fit(model, x, y, workflow="auto", goal="quality")

   # legacy preset argument (deprecated but still supported)
   popt, pcov = fit(model, x, y, preset="robust")

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``preset`` value
     - Behavior
   * - ``fast``
     - Single-start optimization for maximum speed
   * - ``robust``
     - Multi-start with 5 starts
   * - ``global``
     - Thorough global search with 20 starts
   * - ``streaming``
     - Streaming optimization for large datasets with multi-start
   * - ``large``
     - Auto-detect dataset size and use the appropriate strategy

See the ``fit()`` API reference for the current ``workflow``/``goal``
argument set, which supersedes ``preset``.

See Also
----------

- :doc:`/howto/configure_yaml` - CLI workflow YAML file reference
- :doc:`/howto/optimize_performance` - Performance tuning
