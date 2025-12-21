Results and Outputs
===================

This page explains how to interpret workflow outputs and where to find
artifacts after a run.

What you get from a run
-----------------------

A typical workflow produces:

- Fit parameters and uncertainty estimates
- Diagnostic summaries (convergence, stability, residuals)
- Logs for reproducibility and debugging
- Optional export artifacts (tables, plots, reports)

Where outputs are written
-------------------------

Outputs are written to the configured output directory in your YAML file.
Check the ``paths`` section of your configuration for the exact location.

Interpreting fit results
------------------------

For Python usage, the result object is documented here:

- :doc:`../api/nlsq.result`

That reference explains parameter values, covariance, residuals, and
convergence flags.

Logs and diagnostics
--------------------

- User-visible log output is controlled by the ``logging`` section of the
  configuration.
- For diagnostic structures and programmatic access, see
  :doc:`../api/nlsq.diagnostics`.

Reproducibility tips
--------------------

- Keep the exact YAML file used for the run with the outputs.
- Record hardware and JAX versions for comparisons.
- Prefer deterministic settings when comparing runs.
