nlsq.streaming.phases
=====================

Streaming optimization phase classes for large-scale curve fitting.

This subpackage contains the extracted phase classes from the adaptive hybrid
streaming optimizer, enabling modular streaming optimization workflows.

.. automodule:: nlsq.streaming.phases
   :members:
   :undoc-members:
   :show-inheritance:

Phase Classes
-------------

WarmupPhase
~~~~~~~~~~~

.. autoclass:: nlsq.streaming.phases.WarmupPhase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nlsq.streaming.phases.WarmupResult
   :members:
   :undoc-members:
   :show-inheritance:

GaussNewtonPhase
~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.streaming.phases.GaussNewtonPhase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nlsq.streaming.phases.GNResult
   :members:
   :undoc-members:
   :show-inheritance:

PhaseOrchestrator
~~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.streaming.phases.PhaseOrchestrator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nlsq.streaming.phases.PhaseOrchestratorResult
   :members:
   :undoc-members:
   :show-inheritance:

CheckpointManager
~~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.streaming.phases.CheckpointManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nlsq.streaming.phases.CheckpointState
   :members:
   :undoc-members:
   :show-inheritance:
