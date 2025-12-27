Configuration Reference
=======================

NLSQ can be configured via YAML files, environment variables, and programmatic settings.

YAML Configuration
------------------

Create a ``nlsq.yaml`` file in your project directory:

.. code-block:: yaml

   # nlsq.yaml
   optimization:
     gtol: 1.0e-8
     ftol: 1.0e-8
     xtol: 1.0e-8
     max_nfev: 500

   memory:
     limit_gb: 8.0
     chunk_size: auto

   logging:
     level: INFO
     diagnostics: true

   cache:
     enabled: true
     directory: ~/.cache/nlsq

Configuration File Locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NLSQ searches for configuration in this order:

1. ``./nlsq.yaml`` (current directory)
2. ``~/.config/nlsq/config.yaml`` (user config)
3. Environment variables
4. Built-in defaults

Loading Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import Config

   # Load from default locations
   config = Config.load()

   # Load from specific file
   config = Config.load("my_config.yaml")

   # Access values
   print(config.optimization.gtol)

Configuration Sections
----------------------

optimization
~~~~~~~~~~~~

.. code-block:: yaml

   optimization:
     # Convergence tolerances
     gtol: 1.0e-8      # Gradient tolerance
     ftol: 1.0e-8      # Function tolerance
     xtol: 1.0e-8      # Step tolerance

     # Iteration limits
     max_nfev: 500     # Max function evaluations
     max_iter: 100     # Max iterations

     # Algorithm selection
     method: trf       # trf, dogbox, lm
     tr_solver: svd    # svd, lsmr

memory
~~~~~~

.. code-block:: yaml

   memory:
     # Memory limits
     limit_gb: 8.0         # GPU memory limit
     chunk_size: auto      # Automatic or specific size

     # Memory management
     pool_enabled: true    # Use memory pooling
     pool_ttl: 300         # Pool TTL in seconds

logging
~~~~~~~

.. code-block:: yaml

   logging:
     level: INFO           # DEBUG, INFO, WARNING, ERROR
     diagnostics: true     # Enable convergence diagnostics
     progress: true        # Show progress bars
     verbosity: 1          # 0=quiet, 1=normal, 2=verbose

cache
~~~~~

.. code-block:: yaml

   cache:
     enabled: true
     directory: ~/.cache/nlsq
     max_size_gb: 2.0
     jit_cache: true       # JAX JIT compilation cache

precision
~~~~~~~~~

.. code-block:: yaml

   precision:
     dtype: float64        # float32 or float64
     mixed_precision: false
     jax_enable_x64: true  # Enable 64-bit in JAX

streaming
~~~~~~~~~

.. code-block:: yaml

   streaming:
     chunk_size: 100000
     checkpoint_interval: 50
     checkpoint_dir: ./checkpoints
     resume_on_error: true

Environment Variables
---------------------

All settings can be overridden via environment variables:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Variable
     - Config Equivalent
     - Default
   * - ``NLSQ_GTOL``
     - optimization.gtol
     - 1e-8
   * - ``NLSQ_FTOL``
     - optimization.ftol
     - 1e-8
   * - ``NLSQ_XTOL``
     - optimization.xtol
     - 1e-8
   * - ``NLSQ_MAX_NFEV``
     - optimization.max_nfev
     - 500
   * - ``NLSQ_MEMORY_LIMIT_GB``
     - memory.limit_gb
     - auto
   * - ``NLSQ_LOG_LEVEL``
     - logging.level
     - INFO
   * - ``NLSQ_DEBUG``
     - logging.level
     - (sets DEBUG)
   * - ``NLSQ_DISABLE_CACHE``
     - cache.enabled
     - (sets false)
   * - ``NLSQ_FORCE_CPU``
     - -
     - (forces CPU backend)
   * - ``NLSQ_SKIP_GPU_CHECK``
     - -
     - (skips GPU warning)

**Example:**

.. code-block:: bash

   export NLSQ_GTOL=1e-10
   export NLSQ_DEBUG=1
   python my_script.py

Programmatic Configuration
--------------------------

Override configuration in code:

.. code-block:: python

   from nlsq import curve_fit, Config

   # Method 1: Pass directly to functions
   popt, pcov = curve_fit(model, x, y, gtol=1e-10, max_nfev=1000)

   # Method 2: Use Config object
   config = Config(
       optimization={"gtol": 1e-10, "max_nfev": 1000}, memory={"limit_gb": 16.0}
   )
   popt, pcov = curve_fit(model, x, y, config=config)

   # Method 3: Context manager
   from nlsq import config_context

   with config_context(gtol=1e-10):
       popt, pcov = curve_fit(model, x, y)

Presets
-------

Use presets for common configurations:

.. code-block:: python

   from nlsq import fit

   # Fast fitting (lower precision)
   popt, pcov = fit(model, x, y, preset="fast")

   # High precision
   popt, pcov = fit(model, x, y, preset="precise")

   # Global optimization
   popt, pcov = fit(model, x, y, preset="global")

**Preset definitions:**

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - Preset
     - gtol
     - ftol
     - max_nfev
     - Notes
   * - default
     - 1e-8
     - 1e-8
     - 500
     - Standard
   * - fast
     - 1e-6
     - 1e-6
     - 200
     - Quick, good enough
   * - precise
     - 1e-12
     - 1e-12
     - 2000
     - High accuracy
   * - global
     - 1e-8
     - 1e-8
     - 5000
     - Avoids local minima

Default Values
--------------

Complete list of defaults:

.. code-block:: yaml

   # Full default configuration
   optimization:
     gtol: 1.0e-8
     ftol: 1.0e-8
     xtol: 1.0e-8
     max_nfev: 500
     max_iter: 100
     method: trf
     tr_solver: svd

   memory:
     limit_gb: null      # auto-detect
     chunk_size: auto
     pool_enabled: true
     pool_ttl: 300

   logging:
     level: INFO
     diagnostics: false
     progress: true
     verbosity: 1

   cache:
     enabled: true
     directory: ~/.cache/nlsq
     max_size_gb: 2.0
     jit_cache: true

   precision:
     dtype: float64
     mixed_precision: false
     jax_enable_x64: true

   streaming:
     chunk_size: 100000
     checkpoint_interval: 50
     checkpoint_dir: ./checkpoints
     resume_on_error: false

See Also
--------

- :doc:`/howto/configure_yaml` - Configuration how-to guide
- :doc:`/howto/optimize_performance` - Performance tuning
