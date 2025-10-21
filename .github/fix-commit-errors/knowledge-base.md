# CI Error Fix Knowledge Base

## Successful Fixes

### Pattern: powershell-line-continuation-001
**Error Pattern**: `ParserError.*Missing expression after unary operator '--'`

**Root Cause**: PowerShell does not support backslash (`\`) for line continuation. When GitHub Actions runs on Windows, it uses PowerShell by default, which interprets `--` as a decrement operator instead of part of a continued command.

**Solution Applied**: Add `shell: bash` to force bash shell on all platforms
**Confidence**: 98%
**Success Rate**: 1/1 (100%)
**Applicable To**: GitHub Actions workflows with multi-line commands on Windows runners

**Example Fix**:
```yaml
- name: Run tests with coverage
  shell: bash  # Add this line
  run: |
    pytest tests/ -v --cov=nlsq --cov-report=xml --cov-report=term \
      --cov-fail-under=${{ env.COVERAGE_THRESHOLD }}
```

**Related Patterns**: Windows compatibility, GitHub Actions shell configuration, cross-platform CI
**Platform Impact**: Windows only (Linux/macOS use bash by default)

**Last Applied**: 2025-10-21
**Workflow Runs Fixed**: #18672673791, #18672479983, #18672349820
**Time to Resolution**: ~5 minutes
**Commit**: cfe37e7

---

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
