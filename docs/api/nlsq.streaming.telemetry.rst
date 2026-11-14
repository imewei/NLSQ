nlsq.streaming.telemetry
========================

Telemetry and monitoring for the defense layer strategy.

This module provides telemetry infrastructure for the 4-layer defense strategy
used in adaptive hybrid streaming optimization. It tracks activation counts,
timing, and effectiveness metrics for each defense layer.

Defense Layers
--------------

The telemetry system monitors four defense layers:

1. **Layer 1 - Warm Start**: Detects when initial parameters are close to optimal
2. **Layer 2 - Adaptive Step Size**: Monitors step size adjustments
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

   # Record layer events during optimization
   telemetry.record_warmup_start()
   telemetry.record_layer1_trigger(relative_loss=1e-4, threshold=1e-3)
   telemetry.record_layer3_trigger(cost_ratio=1.02, tolerance=1.01, iteration=5)

   # Get summary report
   report = telemetry.get_summary()
   print(f"Layer 1 triggers: {report['layer1']['triggers']}")
   print(f"Layer 3 triggers: {report['layer3']['triggers']}")

See Also
--------

- :doc:`nlsq.adaptive_hybrid_streaming` - Main hybrid optimizer
- :doc:`nlsq.hybrid_streaming_config` - Configuration options
