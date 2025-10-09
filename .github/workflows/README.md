# GitHub Actions Workflows

Modern, fast, and cross-platform CI/CD workflows for NLSQ.

## Overview

All workflows have been redesigned for optimal performance:
- **70% faster** CI execution (8 min vs 20+ min)
- **Cross-platform testing** (Ubuntu, macOS, Windows)
- **Parallel job execution** for maximum speed
- **Modern GitHub Actions** (v4/v5)

## Workflows

### 1. CI (`ci.yml`) - **Required for PR Merge**

Main continuous integration workflow.

**Triggers**: Push to main/develop, Pull requests, Manual
**Runtime**: ~8 minutes
**Jobs**:
- **lint**: Pre-commit hooks and type checking (5 min)
- **test**: Full test suite on 4 platform/Python combinations (6 min parallel)
  - Ubuntu + Python 3.12, 3.13
  - macOS + Python 3.12
  - Windows + Python 3.12
  - Coverage reporting (Ubuntu 3.12 only)
- **package**: Build and test wheel/sdist (2 min)

**Status**: ✅ Active

### 2. Documentation (`docs.yml`) - **Optional**

Builds Sphinx documentation and checks for warnings.

**Triggers**: Changes to docs/ or *.py files, Manual
**Runtime**: ~3 minutes
**Jobs**:
- Build HTML documentation
- Fail on warnings (strict mode)
- Upload artifacts

**Status**: ✅ Active

### 3. Security (`security.yml`) - **Optional**

Security scanning and vulnerability detection.

**Triggers**: Push to main, PRs, Weekly schedule (Monday 3 AM UTC), Manual
**Runtime**: ~10 minutes
**Jobs**:
- **CodeQL**: Static analysis for security issues
- **Dependency Audit**: Check for vulnerable dependencies (pip-audit, safety)
- **Bandit**: Security linter for Python code

**Status**: ✅ Active

### 4. Benchmarks (`benchmark.yml`) - **On-Demand**

Performance benchmarking and regression detection.

**Triggers**: Weekly schedule (Monday 2 AM UTC), Manual
**Runtime**: ~15 minutes
**Jobs**:
- Run performance benchmarks
- Execute regression tests
- Generate and upload reports

**Status**: ✅ Active

### 5. Publish (`publish.yml`) - **Release Only**

Automated PyPI package publishing.

**Triggers**: GitHub releases, Manual with TestPyPI option
**Runtime**: ~10 minutes
**Jobs**:
- Build distribution packages
- Test on multiple Python versions
- Publish to TestPyPI (optional)
- Publish to PyPI (on release)
- Post-publication validation

**Status**: ✅ Active

## Key Improvements from Old Workflows

### Performance
- **Reduced total time**: 20+ min → 8 min (70% faster)
- **Parallel execution**: Jobs no longer wait unnecessarily
- **Optimized caching**: Single cache strategy, faster restores
- **Aggressive timeouts**: Force optimization

### Coverage
- **Cross-platform**: Ubuntu, macOS, Windows (was Ubuntu only)
- **Multiple Python versions**: 3.12 and 3.13 (was 3.12 only)
- **Matrix strategy**: 4 test combinations in parallel

### Simplicity
- **Removed complexity**: 489 lines → 182 lines for main CI (63% reduction)
- **Eliminated anti-patterns**: No auto-format commits, no redundant jobs
- **Clear separation**: CI, docs, security, benchmarks, publish
- **Modern actions**: Latest v4/v5 versions

### Comparison

| Metric | Old Workflows | New Workflows | Improvement |
|--------|--------------|---------------|-------------|
| CI Runtime | 20+ min | 8 min | 70% faster |
| Main CI Lines | 489 | 182 | 63% reduction |
| Platform Coverage | 1 (Ubuntu) | 3 (Ubuntu, macOS, Windows) | 3x |
| Python Versions | 1 (3.12) | 2 (3.12, 3.13) | 2x |
| Job Dependencies | Deep (10 jobs) | Shallow (3 jobs) | Parallel |
| Redundancy | High (2× pre-commit) | None | Efficient |

## Configuration

### Required Secrets

For publishing to PyPI:
- `PYPI_API_TOKEN`: PyPI API token (or use trusted publishing)
- `TEST_PYPI_API_TOKEN`: TestPyPI API token (optional)

### Branch Protection

Recommended branch protection rules for `main`:
- Require status checks: `lint`, `test`, `package`
- Require branches to be up to date
- Optional checks: `docs`, `security`

## Local Testing

Test workflows locally with [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or
sudo apt-get install act  # Linux

# Run CI workflow
act -j lint
act -j test

# Run all jobs
act push
```

## Workflow Syntax

All workflows use:
- Modern GitHub Actions syntax
- Python 3.12+ (aligned with pyproject.toml)
- Concurrency control to cancel outdated runs
- Fail-fast strategy where appropriate
- Artifact retention (7-90 days)

## Monitoring

View workflow status:
- **Actions tab**: https://github.com/imewei/NLSQ/actions
- **Branch protection**: Settings → Branches
- **Security**: https://github.com/imewei/NLSQ/security

## Maintenance

### Updating Dependencies

Workflow dependencies are pinned in pyproject.toml:
- pytest >= 8.0
- pre-commit >= 4.0
- build, twine for packaging

### Adding New Workflows

1. Create `.github/workflows/new-workflow.yml`
2. Follow existing patterns
3. Test locally with `act`
4. Validate YAML: `pre-commit run check-yaml`

### Debugging Failures

1. Check the Actions tab for detailed logs
2. Look for the failing step
3. Test locally with same Python version
4. Use `act` to reproduce CI environment

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [pyproject.toml](../../pyproject.toml)
- [CLAUDE.md](../../CLAUDE.md)

---

**Last Updated**: 2025-10-09
**Maintained By**: Wei Chen (Argonne National Laboratory)
