nlsq.device module
==================

.. currentmodule:: nlsq.device

.. automodule:: nlsq.device
   :noindex:

Overview
--------

The ``nlsq.device`` module provides GPU detection and warning utilities to help
users realize when GPU acceleration is available but not being used. This module
helps maximize performance by alerting users to available hardware acceleration
opportunities.

Key Features
------------

- **Automatic GPU detection** via nvidia-smi hardware query
- **JAX device inspection** to check current compute backend
- **User-friendly warnings** with actionable installation instructions
- **150-270x speedup recommendations** for GPU-enabled configurations
- **Silent failure handling** to avoid disrupting workflow
- **Environment variable control** for CI/CD and intentional CPU-only usage
- **Minimal overhead** (~6ms total: 5ms nvidia-smi + 1ms JAX query)

Functions
---------

.. autosummary::
   :toctree: generated/

   check_gpu_availability

Usage Examples
--------------

Automatic GPU Check on Import
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPU check runs automatically when importing NLSQ:

.. code-block:: python

    import nlsq  # Automatically checks GPU availability

If an NVIDIA GPU is detected but JAX is using CPU, you'll see:

.. code-block:: text

    ⚠️  GPU ACCELERATION AVAILABLE
    ═══════════════════════════════
    NVIDIA GPU detected: Tesla V100-SXM2-16GB
    JAX is currently using: CPU-only

    Enable 150-270x speedup with GPU acceleration:
      make install-jax-gpu

    Or manually:
      pip uninstall -y jax jaxlib
      pip install "jax[cuda12-local]>=0.6.0"

    See README.md GPU Installation section for details.

    To suppress this warning:
      export NLSQ_SKIP_GPU_CHECK=1

Suppressing GPU Warnings
~~~~~~~~~~~~~~~~~~~~~~~~~

For CI/CD pipelines or intentional CPU-only usage:

.. code-block:: python

    import os

    # Option 1: Set before importing NLSQ
    os.environ["NLSQ_SKIP_GPU_CHECK"] = "1"
    import nlsq  # No GPU warning

Or via shell:

.. code-block:: bash

    # Option 2: Export environment variable
    export NLSQ_SKIP_GPU_CHECK=1
    python your_script.py

    # Option 3: Inline with command
    NLSQ_SKIP_GPU_CHECK=1 python your_script.py

    # Option 4: Add to CI/CD environment variables
    # GitHub Actions example:
    env:
      NLSQ_SKIP_GPU_CHECK: "1"

Accepted values: ``"1"``, ``"true"``, ``"yes"`` (case-insensitive)

Manual GPU Check
~~~~~~~~~~~~~~~~

Call the check function directly:

.. code-block:: python

    from nlsq.device import check_gpu_availability

    # Manually trigger GPU check
    check_gpu_availability()

**Note**: This will respect the ``NLSQ_SKIP_GPU_CHECK`` environment variable.

Verifying GPU Usage
~~~~~~~~~~~~~~~~~~~

Check which devices JAX is actually using:

.. code-block:: python

    import jax

    # List all available devices
    devices = jax.devices()
    print(f"JAX devices: {devices}")

    # Expected with GPU: [cuda(id=0)]
    # Expected CPU-only: [CpuDevice(id=0)]

    # Check if using GPU
    using_gpu = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices)
    print(f"Using GPU: {using_gpu}")

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

``NLSQ_SKIP_GPU_CHECK``
    Controls whether GPU availability check runs on import.

    - **Values**: ``"1"``, ``"true"``, ``"yes"`` (case-insensitive)
    - **Default**: Not set (check runs)
    - **Effect**: Suppresses GPU availability warning
    - **Use cases**: CI/CD pipelines, intentional CPU-only usage, stdout parsing

    .. code-block:: bash

        # Suppress GPU check
        export NLSQ_SKIP_GPU_CHECK=1

        # Or inline
        NLSQ_SKIP_GPU_CHECK=1 python script.py

Performance Characteristics
---------------------------

**Check Overhead**:

- **Total time**: ~6ms per import
- **nvidia-smi query**: ~5ms
- **JAX device query**: ~1ms
- **Print warning**: <1ms (only when GPU available but unused)

**When Check Runs**:

- Automatically on first ``import nlsq``
- Only once per Python session (not on subsequent imports)
- Can be manually triggered with ``check_gpu_availability()``

**Failure Behavior**:

The check silently fails (no error messages) when:

- NVIDIA GPU hardware is not present
- nvidia-smi is not installed
- JAX is not installed yet
- Permission denied errors
- Timeout errors (>5 seconds)
- Any unexpected exceptions

This design ensures the check never disrupts normal workflow.

Best Practices
--------------

1. **CI/CD Pipelines**: Set ``NLSQ_SKIP_GPU_CHECK=1`` to suppress warnings
2. **Production Deployment**: Verify GPU usage with ``jax.devices()``
3. **Development**: Keep warnings enabled to catch misconfiguration
4. **Performance Testing**: Compare CPU vs GPU benchmarks
5. **Documentation**: Note GPU requirements in deployment guides

Common Use Cases
----------------

Testing in CI/CD Without GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppress warnings in continuous integration:

.. code-block:: yaml

    # GitHub Actions example
    jobs:
      test:
        runs-on: ubuntu-latest
        env:
          NLSQ_SKIP_GPU_CHECK: "1"  # Suppress GPU warnings
        steps:
          - name: Run tests
            run: pytest tests/

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

Reduce output clutter in notebooks:

.. code-block:: python

    # At top of notebook
    import os

    os.environ["NLSQ_SKIP_GPU_CHECK"] = "1"
    import nlsq  # No warning printed

Programmatic Output Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When parsing stdout programmatically:

.. code-block:: python

    import os
    import subprocess

    # Suppress GPU warnings in subprocess
    env = os.environ.copy()
    env["NLSQ_SKIP_GPU_CHECK"] = "1"

    result = subprocess.run(
        ["python", "my_nlsq_script.py"],
        env=env,
        capture_output=True,
        text=True,
    )

    # Parse clean output without GPU warnings
    output = result.stdout

Debug GPU Detection Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

Manually check GPU availability:

.. code-block:: python

    import subprocess

    # Check if nvidia-smi works
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"GPU detected: {result.stdout.strip()}")
    else:
        print("No GPU detected or nvidia-smi not available")

    # Check JAX backend
    import jax

    print(f"JAX devices: {jax.devices()}")

Implementation Details
----------------------

**GPU Detection Algorithm**:

1. Check if ``NLSQ_SKIP_GPU_CHECK`` environment variable is set
2. Query nvidia-smi for GPU hardware (5 second timeout)
3. Parse GPU name from output
4. Query JAX for current device backend
5. Compare hardware availability vs JAX usage
6. Print warning only if mismatch detected

**Error Handling**:

All exceptions are silently caught to prevent workflow disruption:

- ``FileNotFoundError``: nvidia-smi not installed
- ``subprocess.TimeoutExpired``: nvidia-smi hung
- ``ImportError``: JAX not installed
- ``Exception``: Any other unexpected error

**GPU Name Sanitization**:

- Limit to 100 characters maximum
- Convert to ASCII (replace non-ASCII with ``?``)
- Prevents display issues with special characters

Security Considerations
-----------------------

**Command Injection**:

The module uses ``subprocess.run()`` with a fixed command list (no shell=True),
preventing command injection attacks.

**Timeout Protection**:

nvidia-smi query has a 5-second timeout to prevent hanging.

**Privilege Escalation**:

No privileged operations or file writes are performed.

See Also
--------

- :doc:`nlsq.config` : Configuration management
- :doc:`../tutorials/01_first_fit` : Getting started with GPU setup
- :doc:`../howto/optimize_performance` : Performance optimization guide
- `JAX Installation Guide <https://jax.readthedocs.io/en/latest/installation.html>`_ : JAX GPU setup
