CI/CD Documentation
===================

Comprehensive documentation for NLSQ's continuous integration and deployment pipeline.

.. toctree::
   :maxdepth: 2

   improvements_summary
   codeql_workflow_fix
   github_actions_guide
   codeql_quick_reference

Overview
--------

NLSQ uses GitHub Actions for automated testing, code quality checks, and deployments:

- **Test Suite**: Automated testing on multiple Python versions
- **Code Quality**: Pre-commit hooks, ruff, black, mypy
- **Security**: CodeQL scanning for vulnerabilities
- **Coverage**: Automated coverage reporting (target: 80%)
- **Performance**: Regression testing for optimization validation

Quick References
----------------

CodeQL Quick Reference
~~~~~~~~~~~~~~~~~~~~~~

:doc:`codeql_quick_reference`

Fast reference for common CodeQL operations and queries.

GitHub Actions Guide
~~~~~~~~~~~~~~~~~~~~

:doc:`github_actions_guide`

Complete guide to GitHub Actions workflow syntax and schema validation.

Detailed Documentation
----------------------

CI/CD Improvements Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`improvements_summary`

Summary of recent CI/CD enhancements:

- Pre-commit compliance (24/24 hooks passing)
- Test suite health (817 tests, 100% pass rate)
- Code quality improvements
- Coverage tracking

CodeQL Workflow Fix
~~~~~~~~~~~~~~~~~~~

:doc:`codeql_workflow_fix`

Technical documentation for the CodeQL workflow schema fix:

- Problem diagnosis
- Solution implementation
- Schema validation
- Lessons learned

Workflow Files
--------------

NLSQ's GitHub Actions workflows:

- ``.github/workflows/test.yml`` - Main test suite
- ``.github/workflows/codeql.yml`` - Security scanning
- ``.github/workflows/coverage.yml`` - Coverage reporting
- ``.github/workflows/docs.yml`` - Documentation builds

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
