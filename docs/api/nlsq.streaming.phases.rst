nlsq.streaming.phases
=====================

Streaming optimization phase classes for large-scale curve fitting.

This subpackage contains the extracted phase classes from the adaptive hybrid
streaming optimizer, enabling modular streaming optimization workflows.

.. automodule:: nlsq.streaming.phases
   :members:
   :undoc-members:
   :show-inheritance:

Phase Classes Overview
----------------------

WarmupPhase
~~~~~~~~~~~

L-BFGS warmup phase for initial parameter optimization. See
:class:`~nlsq.streaming.phases.WarmupPhase` and
:class:`~nlsq.streaming.phases.WarmupResult`.

GaussNewtonPhase
~~~~~~~~~~~~~~~~

Streaming Gauss-Newton phase for refined optimization. See
:class:`~nlsq.streaming.phases.GaussNewtonPhase` and
:class:`~nlsq.streaming.phases.GNResult`.

PhaseOrchestrator
~~~~~~~~~~~~~~~~~

Coordinates the multi-phase optimization workflow. See
:class:`~nlsq.streaming.phases.PhaseOrchestrator` and
:class:`~nlsq.streaming.phases.PhaseOrchestratorResult`.

CheckpointManager
~~~~~~~~~~~~~~~~~

Manages checkpoint save/restore for crash recovery. See
:class:`~nlsq.streaming.phases.CheckpointManager` and
:class:`~nlsq.streaming.phases.CheckpointState`.
