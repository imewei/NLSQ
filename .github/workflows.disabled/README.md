# GitHub Actions Workflows (Disabled)

## Status: DISABLED

These workflows were disabled on **2025-10-07** and moved to `.github/workflows.disabled/`.

## Why Disabled?

The workflows are currently not running to avoid CI costs and complexity during development. See CLAUDE.md for details.

## Workflows Available

### ci.yml - Main CI/CD Pipeline
- **Last Updated**: 2025-10-08
- **Status**: Ready for re-enablement
- **Python**: 3.12
- **Dependencies**: Uses pyproject.toml[test] extras
- **Key Features**:
  - Pre-commit checks
  - Fast and slow test splits
  - Coverage reporting
  - Documentation build
  - Package validation
  - Security scanning

### Other Workflows
- `codeql.yml` - Security analysis
- `publish.yml` - Package publishing
- `benchmark.yml` - Performance benchmarks

## Configuration Updates (2025-10-08)

All workflows have been updated to align with:
- **Python 3.12+** (pyproject.toml)
- **NumPy 2.0+** (breaking change documented)
- **JAX 0.7.2** (pyproject.toml >=0.6.0)
- **Ruff 0.14.0** (via pyproject.toml >=0.10.0)
- **pytest 8.4.2** (via pyproject.toml >=8.0)

See `REQUIREMENTS.md` for complete dependency strategy.

## Re-enabling Workflows

To re-enable the workflows:

1. **Move workflows back**:
   ```bash
   mv .github/workflows.disabled/*.yml .github/workflows/
   ```

2. **Verify configurations**:
   ```bash
   # Check workflow syntax
   pre-commit run check-github-workflows --all-files
   ```

3. **Test locally** (if possible):
   ```bash
   # Install act (GitHub Actions local runner)
   brew install act  # macOS
   # or
   sudo apt-get install act  # Linux

   # Run workflow locally
   act -j test
   ```

4. **Monitor first runs**:
   - Check Actions tab in GitHub
   - Review job logs
   - Fix any issues before merging to main

## Notes

- All dependencies are managed via `pyproject.toml`
- Workflows use `pip install -e ".[test]"` for dependency installation
- Pre-commit hooks are cached for performance
- NumPy 2.0+ is required (breaking change from 1.x)

## References

- [REQUIREMENTS.md](../../REQUIREMENTS.md) - Dependency management strategy
- [CLAUDE.md](../../CLAUDE.md) - Development guide
- [pyproject.toml](../../pyproject.toml) - Package configuration

---

**Last Updated**: 2025-10-08
**Maintained By**: Wei Chen (Argonne National Laboratory)
