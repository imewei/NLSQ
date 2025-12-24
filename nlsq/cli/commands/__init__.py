"""NLSQ CLI command handlers.

This package provides command handler modules for the NLSQ CLI:
- fit: Execute single curve fit from YAML workflow configuration
- batch: Execute parallel batch fitting from multiple YAML files
- info: Display system and environment information

Example Usage
-------------
>>> from nlsq.cli.commands import fit, batch, info
>>> result = fit.run_fit("workflow.yaml")
>>> results = batch.run_batch(["w1.yaml", "w2.yaml"])
>>> info.run_info()
"""

from nlsq.cli.commands import batch, fit, info

__all__ = [
    "batch",
    "fit",
    "info",
]
