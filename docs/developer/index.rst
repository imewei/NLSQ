Developer Documentation
=======================

Documentation for NLSQ developers and contributors.

.. toctree::
   :maxdepth: 2

   optimization_case_study
   performance_tuning_guide
   documentation_quality
   pypi_setup
   notebook_utilities
   ci_cd/index
   adr/README

Overview
--------

This section contains technical documentation for developers working on NLSQ:

- Performance optimization case studies
- CI/CD pipeline documentation
- Release and publishing guides
- Development best practices
- Architecture Decision Records (ADRs)

Performance & Optimization
--------------------------

Optimization Case Study
~~~~~~~~~~~~~~~~~~~~~~~

:doc:`optimization_case_study`

Comprehensive analysis of NLSQ's performance optimization journey:

- NumPy<->JAX conversion reduction (8% improvement)
- Profiling methodology and tools
- Decision-making process for deferred optimizations
- Lessons learned and best practices

Performance Tuning Guide
~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`performance_tuning_guide`

Deep technical guide for performance optimization:

- Profiling hot paths
- JIT compilation strategies
- Memory optimization techniques
- GPU/TPU utilization
- Benchmarking methodologies

Architecture Decision Records
-----------------------------

:doc:`adr/README`

Documented architectural decisions for NLSQ:

- :doc:`adr/003-streaming-over-subsampling` - Streaming optimization over subsampling
- :doc:`adr/004-parameter-unpacking-simplification` - Parameter unpacking simplification
- :doc:`adr/005-jax-autodiff-gradients` - JAX autodiff for gradient computation

Notebook Utilities
------------------

:doc:`notebook_utilities`

Modern framework for transforming Jupyter notebooks:

- Automated matplotlib inline configuration
- IPython.display import injection
- plt.show() replacement with display/close pattern
- Incremental processing with checksum tracking

Documentation Quality
---------------------

:doc:`documentation_quality`

Comprehensive guide for maintaining high-quality documentation:

- Zero warnings policy and enforcement
- Automated CI/CD checks
- Pre-commit hooks for local validation
- RST formatting best practices

Release Management
------------------

:doc:`pypi_setup`

Complete guide for publishing NLSQ to PyPI:

- Package preparation
- Version management
- Build and distribution
- Testing releases

CI/CD Pipeline
--------------

See :doc:`ci_cd/index` for CI/CD documentation:

- GitHub Actions workflows (main.yml, release.yml)
- Automated testing and coverage
- Pre-commit hooks
- Quality gates

Contributing
------------

For contribution guidelines, see the main repository:

- `CONTRIBUTING.md <https://github.com/imewei/nlsq/blob/main/CONTRIBUTING.md>`_
- `Code of Conduct <https://github.com/imewei/nlsq/blob/main/CODE_OF_CONDUCT.md>`_
- `Issue Tracker <https://github.com/imewei/nlsq/issues>`_
