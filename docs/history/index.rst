Project History
===============

Historical development artifacts and archived reports from NLSQ's evolution.

.. toctree::
   :maxdepth: 2

   v0.1.1_sprint/README
   v0.1.1_sprint/sprint1_completion
   v0.1.1_sprint/sprint2_completion
   v0.1.1_sprint/sprint3_completion
   v0.1.1_sprint/sprint4_plan
   v0.1.1_sprint/roadmap
   archived_reports/codebase_analysis

Overview
--------

This section preserves the development history of NLSQ, documenting major feature sprints,
design decisions, and evolution of the codebase.

v0.1.1 Feature Sprint
---------------------

The v0.1.1 release (October 7-8, 2025) introduced 25+ major features through a structured
4-phase sprint process:

Sprint Documentation
~~~~~~~~~~~~~~~~~~~~

- :doc:`v0.1.1_sprint/README` - Complete sprint overview
- :doc:`v0.1.1_sprint/sprint1_completion` - Foundation & test safety net
- :doc:`v0.1.1_sprint/sprint2_completion` - Documentation & examples
- :doc:`v0.1.1_sprint/sprint3_completion` - Advanced features & robustness
- :doc:`v0.1.1_sprint/sprint4_plan` - Polish & release preparation
- :doc:`v0.1.1_sprint/roadmap` - 30-day feature roadmap

Sprint Results
~~~~~~~~~~~~~~

**Duration**: 24 days

**Achievements**:

- **Tests**: 743 → 1,160 tests (+417, 99.0% pass rate)
- **Features**: 25+ new production-ready features
- **Documentation**: 10,000+ lines added
- **Examples**: 11 domain-specific examples
- **Performance**: 8% improvement, zero regressions

**Key Features Added**:

- Enhanced CurveFitResult with plotting and statistics
- Progress monitoring callbacks
- Function library with 10+ pre-built models
- Automatic fallback strategies (60% → 85% success rate)
- Smart parameter bounds with auto-inference
- Performance profiler with visualization

See :doc:`v0.1.1_sprint/README` for complete details.

Archived Reports
----------------

Historical analysis and completion reports:

Codebase Analysis
~~~~~~~~~~~~~~~~~

:doc:`archived_reports/codebase_analysis`

Comprehensive codebase architecture analysis covering:

- Module organization and dependencies
- Design patterns and principles
- Code quality metrics
- Test coverage analysis
- Performance characteristics

See ``history/archived_reports/`` directory for additional historical reports.

Current Status
--------------

**Production Release**: v0.1.1 (October 8, 2025)

- Complete backward compatibility maintained
- 25+ new features (all opt-in)
- Comprehensive documentation
- Production-ready robustness
- 77% test coverage (target: 80%)
- 817 tests passing (100% pass rate)

For current development status, see:

- `CHANGELOG.md <https://github.com/imewei/nlsq/blob/main/CHANGELOG.md>`_
- `GitHub Issues <https://github.com/imewei/nlsq/issues>`_
- `GitHub Releases <https://github.com/imewei/nlsq/releases>`_
