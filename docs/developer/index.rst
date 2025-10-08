Developer Documentation
=======================

Documentation for NLSQ developers and contributors.

.. toctree::
   :maxdepth: 2

   optimization_case_study
   performance_tuning_guide
   pypi_setup
   ci_cd/index

Overview
--------

This section contains technical documentation for developers working on NLSQ:

- Performance optimization case studies
- CI/CD pipeline documentation
- Release and publishing guides
- Development best practices

Performance & Optimization
--------------------------

Optimization Case Study
~~~~~~~~~~~~~~~~~~~~~~~

:doc:`optimization_case_study`

Comprehensive analysis of NLSQ's performance optimization journey:

- NumPy↔JAX conversion reduction (8% improvement)
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

Release Management
------------------

PyPI Publishing
~~~~~~~~~~~~~~~

:doc:`pypi_setup`

Complete guide for publishing NLSQ to PyPI:

- Package preparation
- Version management
- Build and distribution
- Testing releases
- Documentation deployment

CI/CD Pipeline
--------------

See :doc:`ci_cd/index` for comprehensive CI/CD documentation:

- GitHub Actions workflows
- CodeQL security scanning
- Automated testing
- Pre-commit hooks
- Quality gates

Contributing
------------

For contribution guidelines, see the main repository:

- `CONTRIBUTING.md <https://github.com/imewei/nlsq/blob/main/CONTRIBUTING.md>`_
- `Code of Conduct <https://github.com/imewei/nlsq/blob/main/CODE_OF_CONDUCT.md>`_
- `Issue Tracker <https://github.com/imewei/nlsq/issues>`_
