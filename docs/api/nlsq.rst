nlsq package
=============

Package Overview
----------------

NLSQ (Nonlinear Least Squares) is a JAX-powered library that provides GPU/TPU-accelerated curve fitting
with automatic differentiation. It offers a drop-in replacement for SciPy's ``curve_fit`` function with
significant performance improvements on modern hardware (150-270x speedup on GPUs).

Key Features:

* **GPU/TPU Acceleration**: JAX JIT compilation to XLA for massive speedups
* **Automatic Differentiation**: No manual Jacobian calculations required
* **Large Dataset Support**: Automatic chunking and memory management for 100M+ points
* **Production-Ready**: 1,779 tests, 100% pass rate
* **Drop-in Compatibility**: Minimal code changes from ``scipy.optimize.curve_fit``

Quick Start Example
-------------------

Basic exponential fit::

    import numpy as np
    from nlsq import curve_fit
    import jax.numpy as jnp


    # Define model function
    def exponential(x, a, b):
        return a * jnp.exp(-b * x)


    # Generate data
    x = np.linspace(0, 5, 1000)
    y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.randn(1000)

    # Fit (GPU-accelerated!)
    popt, pcov = curve_fit(exponential, x, y, p0=[2, 1])
    print(f"Fitted parameters: a={popt[0]:.2f}, b={popt[1]:.2f}")

Large dataset with automatic chunking::

    from nlsq import curve_fit_large

    # 50 million points - automatically chunked
    x_large = np.linspace(0, 10, 50_000_000)
    y_large = 2.0 * np.exp(-0.5 * x_large) + 0.3

    popt, pcov = curve_fit_large(
        exponential,
        x_large,
        y_large,
        p0=[2.5, 0.6],
        memory_limit_gb=4.0,
        show_progress=True
    )

See Also
--------

* :doc:`/getting_started/quickstart` - Complete tutorial
* :doc:`/guides/large_datasets` - Large dataset handling guide
* :doc:`/guides/performance_guide` - Performance optimization
* :doc:`/guides/migration_scipy` - Migrating from SciPy

Submodules
----------

.. toctree::
   :maxdepth: 4

   nlsq.minpack
   nlsq.least_squares
   nlsq.trf
   nlsq.loss_functions
   nlsq.common_jax
   nlsq.common_scipy
   nlsq.caching
   nlsq.config
   nlsq.logging
   nlsq.optimizer_base

Module contents
---------------

.. automodule:: nlsq
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: StreamingConfig, MemoryConfig, LargeDatasetConfig, NumericalStabilityGuard, ParameterNormalizer
