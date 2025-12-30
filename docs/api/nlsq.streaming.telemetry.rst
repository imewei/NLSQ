nlsq.streaming.telemetry
========================

Telemetry and monitoring for the defense layer strategy.

.. versionadded:: 1.2.0
   Extracted from ``nlsq.streaming.adaptive_hybrid`` for modularity.

This module provides telemetry infrastructure for the 4-layer defense strategy
used in adaptive hybrid streaming optimization. It tracks activation counts,
timing, and effectiveness metrics for each defense layer.

Defense Layers
--------------

The telemetry system monitors four defense layers:

1. **Layer 1 - Warm Start**: Detects when initial parameters are close to optimal
2. **Layer 2 - Adaptive Learning Rate**: Monitors learning rate adjustments
3. **Layer 3 - Cost Guard**: Tracks cost increase rejections
4. **Layer 4 - Step Clipping**: Records step size limiting events

Classes
-------

DefenseLayerTelemetry
~~~~~~~~~~~~~~~~~~~~~

Main telemetry class that collects and reports on defense layer activity.

Module Contents
---------------

.. automodule:: nlsq.streaming.telemetry
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from nlsq.streaming.telemetry import DefenseLayerTelemetry

   # Create telemetry instance
   telemetry = DefenseLayerTelemetry()

   # Record layer activations during optimization
   telemetry.record_layer1_activation(cost_reduction=0.05)
   telemetry.record_layer3_rejection(cost_increase=0.02)

   # Get summary report
   report = telemetry.get_summary()
   print(f"Layer 1 activations: {report['layer1_count']}")
   print(f"Layer 3 rejections: {report['layer3_count']}")

See Also
--------

- :doc:`nlsq.adaptive_hybrid_streaming` - Main hybrid optimizer
- :doc:`nlsq.hybrid_streaming_config` - Configuration options
