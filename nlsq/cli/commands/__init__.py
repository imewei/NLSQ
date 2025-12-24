"""NLSQ CLI command handlers.

This package provides command handler modules for the NLSQ CLI:
- fit: Execute single curve fit from YAML workflow configuration
- batch: Execute parallel batch fitting from multiple YAML files
- info: Display system and environment information
- config: Copy configuration templates to current directory

Example Usage
-------------
>>> from nlsq.cli.commands import fit, batch, info, config
>>> result = fit.run_fit("workflow.yaml")
>>> results = batch.run_batch(["w1.yaml", "w2.yaml"])
>>> info.run_info()
>>> config.run_config()
"""

from nlsq.cli.commands import batch, config, fit, info

__all__ = [
    "batch",
    "config",
    "fit",
    "info",
]
