# CI Error Fix Knowledge Base

## Successful Fixes

### Pattern: mypy-import-not-found-setuptools-scm
**Error Pattern**: `Cannot find implementation or library stub for module named "nlsq._version"`

**Root Cause**: setuptools-scm generates `_version.py` at build time, not in source tree

**Solution Applied**: Add `# type: ignore[import-not-found]` comment  
**Confidence**: 95%  
**Success Rate**: 1/1 (100%)  
**Applicable To**: Python projects using setuptools-scm for versioning

**Example Fix**:
```python
from nlsq._version import __version__  # type: ignore[import-not-found]
```

**Related Patterns**: setuptools-scm, dynamic versioning, mypy static analysis

**Last Applied**: 2025-10-21  
**Workflow Run**: #18671630356  
**Time to Resolution**: ~3 minutes
