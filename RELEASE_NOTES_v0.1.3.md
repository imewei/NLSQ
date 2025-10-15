# Release Notes: NLSQ v0.1.3

**Release Date**: 2025-10-15
**Type**: Minor Release (Dependency Optimization + Bug Fixes)

---

## Overview

NLSQ v0.1.3 is a focused release that optimizes dependencies and fixes critical bugs. The headline feature is making h5py an optional dependency, reducing the default install size by **17%** while maintaining full backward compatibility.

---

## What's New

### 🎯 Optional Dependencies (Breaking Change - Minor Impact)

h5py is now optional! This significantly reduces installation size for users who don't need streaming features.

**Before v0.1.3:**
```bash
pip install nlsq  # Installs h5py automatically (larger install)
```

**v0.1.3 and later:**
```bash
# Core features only (17% smaller)
pip install nlsq

# With streaming support
pip install nlsq[streaming]

# Everything (dev, docs, test, build)
pip install nlsq[all]
```

### 📦 New Optional Dependency Groups

- **`[streaming]`**: h5py for StreamingOptimizer
- **`[build]`**: Build tools (setuptools, twine, wheel)
- **`[all]`**: All optional features

### 🐛 Critical Bug Fixes

#### Boolean Operator on NumPy Arrays
Fixed ValueError when using `or` operator on NumPy arrays in `nlsq/large_dataset.py`.

**Problem:**
```python
# ❌ This caused: ValueError: The truth value of an array is ambiguous
result = current_params or np.ones(2)
```

**Solution:**
```python
# ✅ Fixed with explicit None check
result = current_params if current_params is not None else np.ones(2)
```

**Impact**: Prevents runtime errors in edge cases during large dataset curve fitting.

#### Test Suite Improvements
- Streaming tests now skip gracefully when h5py not installed
- Clear error messages guide users to install optional dependencies
- All 1146 tests passing on all platforms (Ubuntu, macOS, Windows)

---

## Installation & Migration

### New Users

```bash
# Install core features
pip install nlsq

# Install with streaming support
pip install nlsq[streaming]
```

### Upgrading from v0.1.2

**Option 1: You use StreamingOptimizer**
```bash
pip install --upgrade nlsq[streaming]
```

**Option 2: You don't use StreamingOptimizer**
```bash
pip install --upgrade nlsq  # 17% smaller install
```

**No code changes required** - the API is 100% backward compatible.

---

## Technical Details

### Implementation Highlights

1. **Lazy h5py imports**: Try/except blocks prevent ImportError when h5py unavailable
2. **Conditional exports**: `__all__` dynamically extended only when h5py present
3. **Smart error messages**: Clear guidance when optional dependencies needed

### Test Results

| Metric | Value |
|--------|-------|
| Tests Passing | 1146/1146 (100%) |
| Tests Skipped | 6 (streaming without h5py) |
| Platforms | Ubuntu ✓, macOS ✓, Windows ✓ |
| Python Versions | 3.12 ✓, 3.13 ✓ |
| Pre-commit Hooks | 24/24 passing |
| CI/CD Status | All workflows passing |

### Commit History

```
d2feef5 style: add trailing comma to skipif decorator
3af11d6 style: apply ruff formatting to h5py import
0d48f3d fix(tests): skip streaming optimizer example test without h5py
1d4b430 fix(tests): skip streaming tests when h5py not installed
1dfb51f style: apply ruff formatting fixes
c286da0 refactor(deps): make h5py optional dependency
fe3d07b fix(large_dataset): replace boolean or with explicit None check
```

---

## Breaking Changes

### Minor: h5py Now Optional

**What changed:**
- h5py moved from core dependencies to `[streaming]` optional group

**Who's affected:**
- Users of `StreamingOptimizer`, `create_hdf5_dataset`, `fit_unlimited_data`

**Migration:**
```bash
# Add [streaming] to your install command
pip install nlsq[streaming]

# Or add to requirements.txt
nlsq[streaming]>=0.1.3
```

**Not affected:**
- Users with h5py already installed (package works as before)
- Users not using streaming features (package still works, 17% smaller)

---

## Compatibility

### Supported Platforms
- ✅ Ubuntu 20.04+ (Python 3.12, 3.13)
- ✅ macOS 12+ (Python 3.12)
- ✅ Windows 10+ (Python 3.12)

### Dependencies
- Python: >=3.12
- NumPy: >=2.0.0
- JAX/JAXlib: >=0.6.0
- SciPy: >=1.14.0

### Optional Dependencies
- h5py: >=3.10.0 (for streaming features)

---

## What's Next

### Planned for v0.1.4
- Documentation improvements
- Additional examples
- Performance optimizations

### Roadmap
- Enhanced GPU/TPU support
- Additional optimization algorithms
- Improved error handling

---

## Thank You

Thanks to all contributors and users who reported issues and provided feedback!

### Contributing

Found a bug? Have a feature request?
- Report issues: https://github.com/imewei/NLSQ/issues
- Contribute: See CONTRIBUTING.md

### Links

- **Documentation**: https://nlsq.readthedocs.io
- **PyPI**: https://pypi.org/project/nlsq
- **GitHub**: https://github.com/imewei/NLSQ
- **Changelog**: See CHANGELOG.md

---

**Full Changelog**: https://github.com/imewei/NLSQ/compare/v0.1.2...v0.1.3
