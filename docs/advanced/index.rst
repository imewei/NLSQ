Advanced Usage / Developers
===========================

**When should I read this?** Read this section if you need API-level
customization, want to build workflows programmatically, or are
contributing to NLSQ internals.

If you want workflow-first guidance and configuration-driven usage, start
with :doc:`../user_guide/index`.

Advanced Concepts
-----------------

.. toctree::
   :maxdepth: 2

   ../guides/advanced_customization
   ../architecture/index

API Reference
-------------

See the :doc:`../api/index` for complete API documentation:

- :doc:`../api/modules` - Full module reference
- :doc:`../api/large_datasets_api` - Large dataset API

Custom Pipelines
----------------

- Workflow system API: :doc:`../api/nlsq.workflow`
- Streaming optimizer: :doc:`../api/nlsq.streaming_optimizer`
- Optimizer base classes: :doc:`../api/nlsq.optimizer_base`

Extending the System
--------------------

- Architecture decisions: :doc:`../architecture/adr/README`
- Stability and diagnostics: :doc:`../api/nlsq.stability`, :doc:`../api/nlsq.diagnostics`

Developer Guide
---------------

See :doc:`../developer/index` for developer documentation:

- :doc:`../developer/ci_cd/index` - CI/CD documentation
