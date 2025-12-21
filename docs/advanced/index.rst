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

.. toctree::
   :maxdepth: 2

   ../api/index
   ../api/modules
   ../api/large_datasets_api

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

.. toctree::
   :maxdepth: 2

   ../developer/index
   ../developer/ci_cd/index
