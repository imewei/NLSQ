# Migration Guide: NLSQ v0.6.0

This guide helps you migrate from NLSQ v0.5.x to v0.6.0. Version 0.6.0 removes
deprecated functionality that was announced in previous releases.

## Summary of Breaking Changes

1. **Removed deprecated workflow presets** (domain-specific presets)
2. **Removed SloppyModelAnalyzer and SloppyModelReport aliases**
3. **Removed IssueCategory.SLOPPY enum alias**
4. **Removed compute_svd_adaptive() function**
5. **Emptied nlsq.compat module**

## Detailed Migration Instructions

### 1. Workflow Presets

The following domain-specific presets have been removed:

| Removed Preset   | Replacement          |
|------------------|----------------------|
| `xpcs`           | `precision_standard` |
| `saxs`           | `precision_standard` |
| `kinetics`       | `precision_standard` |
| `dose_response`  | `precision_high`     |
| `imaging`        | `streaming_large`    |
| `materials`      | `precision_standard` |
| `binding`        | `precision_standard` |
| `synchrotron`    | `streaming_large`    |

**Before (v0.5.x):**
```python
from nlsq.core.workflow import WorkflowConfig

# This now raises ValueError
config = WorkflowConfig.from_preset("xpcs")
```

**After (v0.6.0):**
```python
from nlsq.core.workflow import WorkflowConfig

# Use the generic replacement
config = WorkflowConfig.from_preset("precision_standard")
```

### 2. Parameter Sensitivity Analysis

The `SloppyModelAnalyzer` and `SloppyModelReport` aliases have been removed.
Use `ParameterSensitivityAnalyzer` and `ParameterSensitivityReport` instead.

**Before (v0.5.x):**
```python
from nlsq.diagnostics import SloppyModelAnalyzer, SloppyModelReport

analyzer = SloppyModelAnalyzer(config=config)
report: SloppyModelReport = analyzer.analyze(jacobian)
```

**After (v0.6.0):**
```python
from nlsq.diagnostics import ParameterSensitivityAnalyzer, ParameterSensitivityReport

analyzer = ParameterSensitivityAnalyzer(config=config)
report: ParameterSensitivityReport = analyzer.analyze(jacobian)
```

### 3. Issue Category Enum

The `IssueCategory.SLOPPY` alias has been removed. Use `IssueCategory.SENSITIVITY`.

**Before (v0.5.x):**
```python
from nlsq.diagnostics import IssueCategory

if issue.category == IssueCategory.SLOPPY:
    handle_sensitivity_issue(issue)
```

**After (v0.6.0):**
```python
from nlsq.diagnostics import IssueCategory

if issue.category == IssueCategory.SENSITIVITY:
    handle_sensitivity_issue(issue)
```

### 4. SVD Functions

The `compute_svd_adaptive()` function has been removed. Use
`compute_svd_with_fallback()` instead.

**Before (v0.5.x):**
```python
from nlsq.stability.svd_fallback import compute_svd_adaptive

U, s, V = compute_svd_adaptive(matrix, use_randomized=False)
```

**After (v0.6.0):**
```python
from nlsq.stability.svd_fallback import compute_svd_with_fallback

U, s, V = compute_svd_with_fallback(matrix)
```

Note: The `use_randomized` parameter was always ignored in v0.5.x (randomized
SVD was removed in v0.3.5 due to optimization divergence issues).

### 5. Compatibility Module

The `nlsq.compat` module is now empty. If you were importing deprecated
functions from this module, update to the new import paths as documented
in the sections above.

**Before (v0.5.x):**
```python
from nlsq.compat import some_deprecated_function
```

**After (v0.6.0):**
```python
# Import from the new canonical location
from nlsq.some_module import the_new_function
```

## Finding Deprecated Usage in Your Code

Run a search for these patterns in your codebase to identify code that needs
updating:

```bash
# Find deprecated preset usage
grep -rn "from_preset.*xpcs\|saxs\|kinetics\|dose_response\|imaging\|materials\|binding\|synchrotron" .

# Find deprecated class names
grep -rn "SloppyModelAnalyzer\|SloppyModelReport" .

# Find deprecated enum usage
grep -rn "IssueCategory.SLOPPY" .

# Find deprecated SVD function
grep -rn "compute_svd_adaptive" .

# Find compat imports
grep -rn "from nlsq.compat import" .
```

## Getting Help

If you encounter issues during migration:

1. Check the [API documentation](https://nlsq.readthedocs.io/en/latest/api/)
2. Search [GitHub Issues](https://github.com/imewei/NLSQ/issues)
3. Open a new issue with the `migration` label
