CI/CD Documentation
===================

Documentation for NLSQ's continuous integration and deployment pipeline.

.. toctree::
   :maxdepth: 2

   github_actions_guide

Overview
--------

NLSQ uses GitHub Actions for automated testing, code quality checks, and deployments:

- **Test Suite**: Automated testing on multiple Python versions (3279 tests, 100% pass rate)
- **Code Quality**: Pre-commit hooks, ruff, mypy
- **Coverage**: Automated coverage reporting (target: 80%)

Workflow Files
--------------

NLSQ's GitHub Actions workflows:

- ``.github/workflows/main.yml`` - Main test suite and code quality checks
- ``.github/workflows/release.yml`` - Package publishing and release automation

GitHub Actions Guide
--------------------

:doc:`github_actions_guide`

Guide to GitHub Actions workflow syntax and validation.

Local Testing
-------------

Run CI checks locally before pushing:

.. code-block:: bash

   # Run pre-commit hooks
   pre-commit run --all-files

   # Run full test suite
   make test

   # Run with coverage
   make test-cov

   # Check code quality
   make lint

See :doc:`../pypi_setup` for release pipeline documentation.
