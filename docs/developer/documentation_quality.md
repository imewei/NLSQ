# Documentation Quality Assurance

This guide covers the automated checks and tools for maintaining high-quality documentation.

## Zero Warnings Policy

NLSQ enforces a **zero warnings policy** for Sphinx documentation builds. All documentation must build cleanly without errors or warnings.

### Status

- **Current State**: ✅ 0 warnings (as of 2025-10-31)
- **Enforcement**: Enabled in CI/CD via `-W` flag
- **Last Major Fix**: Mixed precision documentation ambiguity resolution

## Automated Checks

### 1. CI/CD Pipeline (GitHub Actions)

The documentation workflow (`.github/workflows/docs.yml`) automatically:

- **Builds Sphinx documentation** with warnings treated as errors
- **Checks for broken links** in documentation
- **Validates docstring quality** with pydocstyle and darglint
- **Tests documentation examples** with doctest
- **Measures documentation coverage** with interrogate

**Key Configuration:**
```yaml
# Build with warnings as errors
make html SPHINXOPTS="-W --keep-going"
```

**Flags:**
- `-W`: Treat warnings as errors (build fails on warnings)
- `--keep-going`: Continue after first warning to show all issues
- `-q`: Quiet mode (only show warnings/errors in pre-commit hook)

**Important Note:** The `-n` (nitpicky mode) flag is **intentionally disabled** for scientific computing libraries. Nitpicky mode enforces strict cross-reference checking on all type hints, which is too strict for codebases that use informal type descriptions (e.g., "callable", "array_like", "optional") in docstrings. This pragmatic approach maintains zero-warning standards for actual documentation errors while allowing flexibility in type annotations common in scientific Python code.

### 2. Pre-commit Hook

A manual pre-commit hook is available for local validation:

```bash
# Run documentation build check
pre-commit run --hook-stage manual docs-build --all-files

# Or for specific files
pre-commit run --hook-stage manual docs-build --files docs/guides/workflow_options.rst docs/guides/advanced_customization.rst
```

**Configuration:** See `.pre-commit-config.yaml`

**Why Manual?** Building full Sphinx docs takes ~2-3 seconds. Making it manual prevents slowing down every commit while still providing easy validation when needed.

## Common Documentation Issues

### 1. RST Table Formatting

**Problem:** Column text exceeds separator width
```rst
============================  # Column separator too short
Scenario
============================
Large datasets (>100K points) # Text overflows
```

**Solution:** Widen column separators
```rst
==============================  # Widened separator
Scenario
==============================
Large datasets (>100K points)   # Now fits
```

**Alternative:** Use grid tables for complex layouts
```rst
+=============================+
| Scenario                    |
+=============================+
| Large datasets (>100K pts)  |
+-----------------------------+
```

### 2. Ambiguous Cross-References

**Problem:** Multiple classes with same name
```python
# nlsq/diagnostics.py
class ConvergenceMonitor: ...


# nlsq/mixed_precision.py
class ConvergenceMonitor: ...  # Same name!


# In docstring (ambiguous)
"""Uses :class:`ConvergenceMonitor` to track progress."""
```

**Solution:** Use fully-qualified names
```python
"""Uses :class:`nlsq.mixed_precision.ConvergenceMonitor` to track progress."""
```

### 3. Broken Internal Links

**Problem:** Linking to non-existent documentation
```rst
See :doc:`../guides/nonexistent_guide`
```

**Solution:** Verify paths and use `-n` flag to catch
```bash
make html SPHINXOPTS="-n"
```

## Best Practices

### Writing Documentation

1. **Use fully-qualified class references** when ambiguity exists
   ```rst
   .. Good
   :class:`nlsq.mixed_precision.ConvergenceMonitor`

   .. Avoid (if ambiguous)
   :class:`ConvergenceMonitor`
   ```

2. **Test tables locally** before committing
   ```bash
   make -C docs html
   ```

3. **Follow NumPy docstring style** consistently
   ```python
   def function(param: int) -> str:
       """Short description.

       Longer description paragraph.

       Parameters
       ----------
       param : int
           Parameter description

       Returns
       -------
       str
           Return value description
       """
   ```

4. **Keep table columns balanced**
   - Use consistent column widths
   - Test with longest expected content
   - Consider grid tables for complex data

### Pre-Commit Workflow

```bash
# 1. Make documentation changes
vim docs/guides/my_guide.rst

# 2. Test locally (fast)
make -C docs html

# 3. Run pre-commit check (thorough)
pre-commit run --hook-stage manual docs-build --all-files

# 4. Commit if passing
git add docs/guides/my_guide.rst
git commit -m "docs: add my_guide"
```

## Troubleshooting

### Build Fails in CI but Passes Locally

**Cause:** Different Sphinx versions or dependencies

**Solution:**
```bash
# Install dev dependencies
pip install -e ".[dev,docs]"

# Test with CI flags
make -C docs html SPHINXOPTS="-W --keep-going -n"
```

### Pre-commit Hook Fails

**Cause:** Working directory issues

**Solution:**
```bash
# Ensure you're in project root
cd /path/to/NLSQ

# Verify docs directory exists
ls docs/conf.py

# Run manually for debugging
cd docs && make html SPHINXOPTS="-W --keep-going -n -q"
```

### Many Warnings After Update

**Cause:** Sphinx or dependency update changed behavior

**Solution:**
1. Review warnings in detail: `make html 2>&1 | grep WARNING`
2. Fix systematically, one type at a time
3. Consider adding to `.github/fix-commit-errors/` knowledge base

## Metrics and Monitoring

### Current Documentation Quality

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Sphinx Warnings | 0 | 0 | ✅ |
| Broken Links | 0 | 0 | ✅ |
| Doc Coverage | >80% | ~70% | 🟡 |
| Docstring Style | 100% | ~95% | 🟡 |

### Historical Context

- **v0.1.5 and earlier**: 1036+ warnings (disabled `-W` flag)
- **v0.1.6 (2025-10-31)**:
  - Fixed RST table formatting issues
  - Resolved ambiguous cross-references
  - Achieved zero warnings
  - Enabled `-W` flag in CI

## References

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [GitHub Actions Workflows](https://docs.github.com/en/actions)

## See Also

- [Development Guidelines (CLAUDE.md)](https://github.com/imewei/NLSQ/blob/main/CLAUDE.md)
- [Contribution Workflow (CONTRIBUTING.md)](https://github.com/imewei/NLSQ/blob/main/CONTRIBUTING.md)
- [Sphinx Configuration](../conf.py)
- [CI/CD Workflow](https://github.com/imewei/NLSQ/blob/main/.github/workflows/docs.yml)
