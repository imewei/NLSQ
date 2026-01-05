# nlsq/compat/__init__.py
"""Compatibility module - deprecated import paths have been removed.

As of NLSQ v0.6.0, all deprecated compatibility shims have been removed.
This module is now empty and will be removed in a future release.

Migration Guide
---------------
If you were using deprecated import paths, please update your code:

Old paths (no longer work):
    from nlsq.compat import get_deprecated_module
    from nlsq.diagnostics import SloppyModelAnalyzer
    from nlsq.diagnostics import SloppyModelReport

New paths:
    from nlsq.diagnostics import ParameterSensitivityAnalyzer
    from nlsq.diagnostics import ParameterSensitivityReport

The following deprecated workflow presets have been removed:
    - 'xpcs' -> use 'precision_standard'
    - 'saxs' -> use 'precision_standard'
    - 'kinetics' -> use 'precision_standard'
    - 'dose_response' -> use 'precision_high'
    - 'imaging' -> use 'streaming_large'
    - 'materials' -> use 'precision_standard'
    - 'binding' -> use 'precision_standard'
    - 'synchrotron' -> use 'streaming_large'

For the complete migration guide, see:
    https://nlsq.readthedocs.io/en/latest/howto/migration-v0.6.0.html
"""

__all__: list[str] = []
