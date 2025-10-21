.. NLSQ documentation master file

.. figure:: images/NLSQ_logo.png
   :alt: NLSQ Logo

NLSQ: GPU/TPU-Accelerated Nonlinear Least Squares
==================================================

**Fast, production-ready curve fitting for scientific computing**

NLSQ is a JAX-powered library that brings GPU/TPU acceleration to nonlinear least squares curve fitting.
It provides a drop-in replacement for SciPy's ``curve_fit`` with massive speedups on modern hardware.

.. note::
   **New to NLSQ?** Start with :doc:`getting_started/index` → :doc:`getting_started/quickstart`

   **Migrating from SciPy?** See :doc:`guides/migration_scipy`

Key Features
------------

- **GPU/TPU Acceleration**: 150-270x faster than SciPy on large datasets
- **Drop-in Compatibility**: Minimal code changes from ``scipy.optimize.curve_fit``
- **Automatic Differentiation**: JAX autodiff eliminates manual Jacobian calculations
- **Production-Ready**: 80.90% test coverage, 1235 tests, 100% pass rate
- **Large Dataset Support**: Automatic chunking and memory management
- **Advanced Features**: Automatic fallback, smart bounds, numerical stability

Quick Example
-------------

.. code-block:: python

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

See :doc:`getting_started/quickstart` for a complete tutorial.

Performance
-----------

**GPU Benchmarks** (NVIDIA Tesla V100):

- 1M points, 5 parameters: **0.15s** (NLSQ) vs 40.5s (SciPy) = **270x speedup**
- Excellent scaling: 50x more data → only 1.2x slower

See :doc:`developer/optimization_case_study` for detailed performance analysis.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/index
   guides/migration_scipy
   guides/advanced_features
   guides/performance_guide
   guides/large_datasets
   guides/troubleshooting
   migration/streaming_fault_tolerance

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/modules
   api/large_datasets_api

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   developer/index
   developer/optimization_case_study
   developer/performance_tuning_guide
   developer/pypi_setup
   developer/ci_cd/index
   architecture/index

Resources
---------

- **GitHub**: https://github.com/imewei/nlsq
- **PyPI**: https://pypi.org/project/nlsq/
- **Issues**: https://github.com/imewei/nlsq/issues
- **Original Paper**: `JAXFit on ArXiv <https://doi.org/10.48550/arXiv.2208.12187>`_

Project Status
--------------

**Current Release**: v0.1.5

- Production-ready for scientific computing
- Active development and maintenance
- Comprehensive test suite (1235 tests, 100% pass rate)
- 80.90% code coverage (exceeds 80% target)

Citation
--------

If you use NLSQ in your research, please cite the original JAXFit paper:

   Hofer, L. R., Krstajić, M., & Smith, R. P. (2022). JAXFit: Fast Nonlinear Least Squares Fitting in JAX.
   *arXiv preprint arXiv:2208.12187*. https://doi.org/10.48550/arXiv.2208.12187

Acknowledgments
---------------

NLSQ is an enhanced fork of `JAXFit <https://github.com/Dipolar-Quantum-Gases/JAXFit>`_,
originally developed by Lucas R. Hofer, Milan Krstajić, and Robert P. Smith.

Current maintainer: **Wei Chen** (Argonne National Laboratory)

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
