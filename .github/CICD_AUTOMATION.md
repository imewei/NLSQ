# CI/CD Automation Guide

Complete guide to NLSQ's automated CI/CD workflows, from development to production deployment.

## Table of Contents

- [Overview](#overview)
- [Workflow Architecture](#workflow-architecture)
- [Core Workflows](#core-workflows)
- [Automation Features](#automation-features)
- [Monitoring & Observability](#monitoring--observability)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

NLSQ implements enterprise-grade CI/CD automation with:

- ✅ **12 Parallel Test Jobs** across 3 platforms × 2 Python versions × 2 test types
- ✅ **Automated Dependency Updates** with Renovate Bot
- ✅ **Performance Benchmarking** with historical tracking
- ✅ **Advanced Security Scanning** with CodeQL
- ✅ **Semantic Release Automation** with changelog generation
- ✅ **Real-time Status Dashboard** with workflow monitoring

### Automation Matrix

| Feature | Status | Trigger | Frequency |
|---------|--------|---------|-----------|
| CI/CD Pipeline | ✅ Active | Push, PR | On-demand |
| Security Scan | ✅ Active | Push, PR, Schedule | Daily + On-change |
| CodeQL Analysis | ✅ Active | Push, PR, Schedule | Weekly + On-change |
| Performance Benchmarks | ✅ Active | Push, PR, Schedule | Weekly + On-change |
| Dependency Updates | ✅ Active | Schedule | Weekdays 10PM-5AM |
| Status Dashboard | ✅ Active | Schedule, Workflow completion | Every 6 hours |
| Documentation Build | ✅ Active | Push, PR | On-change |
| Release Automation | ✅ Active | Push to main | On-demand |

---

## Workflow Architecture

### 1. CI Pipeline (`ci.yml`)

**Purpose**: Comprehensive testing across all platforms and Python versions

**Stages**:
1. **Change Detection** - Identify code vs docs-only changes
2. **Dependency Validation** - Verify dependency specifications
3. **Code Quality** - Linting, type checking, complexity analysis
4. **Test Matrix** - 6 parallel jobs (ubuntu/macos/windows × py3.12/py3.13)
5. **Build & Package** - Wheel/sdist generation and validation
6. **Integration Tests** - 6 parallel jobs with build artifacts
7. **Test Summary** - Aggregate results from all platforms

**Triggers**: `push`, `pull_request`
**Duration**: ~5-8 minutes (parallel execution)
**Artifacts**: Test results, coverage reports (7-day retention)

**Test Configuration**:
```bash
pytest tests/ -v -n auto \
  --cov=nlsq \
  --cov-report=xml \
  --cov-report=html \
  --cov-report=term \
  --cov-fail-under=80 \
  -m "not slow"
```

---

### 2. Security Workflow (`security.yml`)

**Purpose**: Multi-layered security scanning and vulnerability detection

**Security Layers**:
- **Bandit** - Python security linter
- **pip-audit** - Dependency vulnerability scanner
- **Safety** - Known security vulnerabilities database
- **SAST** - Static application security testing
- **License Compliance** - OSI-approved license verification

**Triggers**: `push`, `pull_request`, `schedule (daily)`
**Duration**: ~3-5 minutes
**Fail Conditions**: Critical vulnerabilities, non-compliant licenses

---

### 3. CodeQL Advanced Security (`codeql.yml`)

**Purpose**: Deep code analysis for security vulnerabilities and code quality

**Features**:
- Security-extended query suite
- Security-and-quality analysis
- Custom query filters
- Path-specific configurations
- SARIF result upload

**Configuration**: `.github/codeql/codeql-config.yml`

**Triggers**: `push`, `pull_request`, `schedule (weekly Monday 2AM)`
**Duration**: ~10-15 minutes
**Results**: GitHub Security tab

**Critical Issue Handling**:
```yaml
# Fails on critical (error-level) issues
# Warns on high (warning-level) issues
# Logs medium (note-level) issues
```

---

### 4. Performance Benchmarks (`performance.yml`)

**Purpose**: Track performance metrics and detect regressions

**Benchmark Types**:
1. **Core Performance** - CPU and GPU backend benchmarks
2. **Performance Comparison** - PR vs baseline comparison
3. **Memory Profiling** - Memory usage analysis

**Features**:
- Historical performance tracking
- Automated regression detection (>150% threshold)
- PR comments on performance changes
- Benchmark artifacts with 30-day retention

**Triggers**: `push (main)`, `pull_request`, `schedule (weekly Sunday)`, `workflow_dispatch`
**Duration**: ~15-20 minutes
**Alerts**: @imewei on >150% regression

**Manual Trigger Options**:
```bash
gh workflow run performance.yml \
  -f dataset_size=large \
  -f compare_with=feature-branch
```

---

### 5. Documentation (`docs.yml`)

**Purpose**: Build and validate Sphinx documentation

**Stages**:
1. **Sphinx Build** - Generate HTML documentation
2. **Warning Validation** - Enforce zero-warning builds
3. **Coverage Analysis** - API reference completeness
4. **Link Checking** - Verify all documentation links
5. **Artifact Upload** - Store built documentation

**Triggers**: `push`, `pull_request`
**Duration**: ~7-8 minutes
**Deployment**: GitHub Pages (on main branch)

---

### 6. Dependency Updates (`renovate.json`)

**Purpose**: Automated dependency updates with intelligent grouping

**Update Strategy**:

| Dependency Type | Auto-merge | Schedule | Priority |
|----------------|-----------|----------|----------|
| Security patches | ✅ Yes | Any time | Highest (10) |
| Dev dependencies | ✅ Yes | Weekdays 10PM-5AM | High |
| Testing tools | ✅ Yes | Grouped | Medium |
| Documentation tools | ✅ Yes | Grouped | Medium |
| Code quality tools | ✅ Yes | Grouped | Medium |
| JAX ecosystem | ❌ Manual | Monday 5AM | Critical review |
| NumPy/SciPy | ❌ Manual | Monthly | Critical review |
| Major updates | ❌ Manual | First of month | Manual review |

**Configuration Highlights**:
```json
{
  "schedule": ["after 10pm every weekday", "before 5am every weekday"],
  "prConcurrentLimit": 5,
  "prHourlyLimit": 2,
  "vulnerabilityAlerts": {
    "automerge": true,
    "schedule": ["at any time"]
  }
}
```

**Package Grouping**:
- JAX ecosystem: jax, jaxlib, flax, optax
- Testing tools: pytest, hypothesis, coverage, pytest-cov
- Docs tools: sphinx, myst-parser, furo, sphinx-autobuild
- Quality tools: ruff, mypy, pre-commit

---

### 7. Status Dashboard (`status-dashboard.yml`)

**Purpose**: Real-time workflow status monitoring

**Features**:
- Automated status badge generation
- Workflow health monitoring
- Coverage & metrics tracking
- Quick links to all resources

**Updates**: Every 6 hours + after workflow completions
**Location**: `.github/WORKFLOW_STATUS.md`

**Monitored Workflows**:
- ✅ CI Pipeline
- ✅ Documentation
- ✅ Security Scan
- ✅ Performance Benchmarks
- ✅ CodeQL Analysis

---

### 8. Release Automation (`release.yml`)

**Purpose**: Semantic versioning and automated releases

**Features**:
- Conventional commit parsing
- Automatic version bumping
- CHANGELOG.md generation
- GitHub release creation
- PyPI package publishing

**Triggers**: `push (main)`, `workflow_dispatch`
**Version Strategy**: Semantic versioning (MAJOR.MINOR.PATCH)

---

## Automation Features

### Cross-Platform Testing

**Test Matrix**: 12 parallel jobs

| Platform | Python 3.12 | Python 3.13 |
|----------|-------------|-------------|
| 🐧 Ubuntu | Unit + Integration | Unit + Integration |
| 🍎 macOS | Unit + Integration | Unit + Integration |
| 🪟 Windows | Unit + Integration | Unit + Integration |

**Execution**: All jobs run in parallel with `fail-fast: false`
**Within-job**: `pytest -n auto` for CPU-core parallelization

---

### Intelligent Change Detection

Skips resource-intensive jobs for docs-only changes:

```yaml
# Detects:
- Code changes: *.py, *.toml, *.txt, *.lock
- Workflow changes: .github/workflows/ci.yml
- Docs-only: Everything else

# Skips on docs-only:
- Code quality checks
- Test matrix
- Integration tests
```

---

### Artifact Management

| Artifact Type | Retention | Size | Use Case |
|--------------|-----------|------|----------|
| Test results | 7 days | ~10MB | Debugging test failures |
| Coverage reports | 7 days | ~50MB | Coverage analysis |
| Build distributions | 7 days | ~5MB | Package verification |
| Benchmark results | 30 days | ~100MB | Performance tracking |
| Memory profiles | 30 days | ~20MB | Memory optimization |
| Sphinx HTML | 30 days | ~200MB | Documentation preview |

---

## Monitoring & Observability

### Status Dashboard

View real-time status: `.github/WORKFLOW_STATUS.md`

**Metrics Tracked**:
- Workflow success/failure rates
- Test coverage percentage
- Code quality grade
- Dependency health
- Security vulnerability count

### Notifications

**Slack Integration** (if configured):
```yaml
# Notify on:
- Build failures (main branch)
- Security vulnerabilities
- Performance regressions >150%
- CodeQL critical issues
```

**Email Notifications**:
- @imewei for critical issues
- Security advisories

---

## Configuration

### Environment Variables

```yaml
# CI Workflow
PYTHON_VERSION_DEFAULT: "3.12"
COVERAGE_THRESHOLD: 80

# Performance Benchmarks
BENCHMARK_MIN_ROUNDS: 5
BENCHMARK_WARMUP: on

# Security
BANDIT_SEVERITY: medium
PIP_AUDIT_IGNORE_VULN: ""  # Comma-separated CVE IDs
```

### Secrets Required

| Secret | Purpose | Required For |
|--------|---------|--------------|
| `GITHUB_TOKEN` | Workflow automation | All workflows |
| `CODECOV_TOKEN` | Coverage upload | CI workflow |
| `PYPI_API_TOKEN` | Package publishing | Release workflow |
| `SLACK_WEBHOOK` | Notifications | Optional |

---

## Troubleshooting

### Common Issues

#### 1. Test Failures on Specific Platform

**Symptom**: Tests pass on ubuntu but fail on windows/macos

**Solution**:
```bash
# Run locally with platform-specific pytest markers
pytest tests/ -m "not (slow or platform_specific)"

# Check platform-specific code paths
grep -r "sys.platform" nlsq/
```

#### 2. Coverage Below Threshold

**Symptom**: `coverage fail-under=80` error

**Solution**:
```bash
# Generate local coverage report
pytest tests/ --cov=nlsq --cov-report=html
open htmlcov/index.html  # View missing coverage
```

#### 3. Renovate PRs Not Auto-merging

**Symptom**: Dependency PRs require manual merge

**Checklist**:
- ✅ Branch protection allows auto-merge
- ✅ Status checks passing
- ✅ PR matches auto-merge criteria
- ✅ No conflicts with base branch

#### 4. Performance Regression Alert

**Symptom**: Benchmark >150% slower than baseline

**Investigation**:
```bash
# Download benchmark artifacts
gh run download <run-id> -n benchmark-results-cpu

# Compare with baseline
python scripts/compare_benchmarks.py \
  baseline-results.json \
  current-results.json
```

#### 5. CodeQL False Positives

**Symptom**: CodeQL reports non-issues

**Solution**:
Edit `.github/codeql/codeql-config.yml`:
```yaml
query-filters:
  - exclude:
      id: py/specific-query-id
```

---

## Workflow Execution Guide

### Manual Workflow Triggers

```bash
# Run specific workflow
gh workflow run <workflow-name>.yml

# Run with inputs
gh workflow run performance.yml \
  -f dataset_size=large

# View workflow status
gh run list --workflow=ci.yml --limit 5

# Watch workflow execution
gh run watch <run-id>

# View logs
gh run view <run-id> --log
```

### Local Development Workflow

```bash
# 1. Install pre-commit hooks
pre-commit install

# 2. Run hooks manually
pre-commit run --all-files

# 3. Run tests locally
pytest tests/ -v -n auto --cov=nlsq

# 4. Build documentation
cd docs && make html

# 5. Run benchmarks
pytest benchmark/ --benchmark-only
```

---

## Best Practices

### Commit Message Format

Follow conventional commits for automated releases:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

**Examples**:
```
feat(core): add GPU acceleration for large datasets
fix(optimizer): resolve convergence issue in BFGS
docs(api): update curve_fit parameters
perf(jit): optimize JIT compilation for models
```

### Pull Request Workflow

1. Create feature branch: `git checkout -b feature/description`
2. Make changes with conventional commits
3. Push and create PR
4. Wait for all CI checks to pass
5. Review performance comparison
6. Merge (auto-merge if configured)

### Dependency Management

**Adding Dependencies**:
```bash
# Add to pyproject.toml with minimum version
dependencies = [
    "new-package>=1.0.0"  # Use >= not ==
]

# Renovate will handle updates
```

**Security Updates**:
- Auto-merged within 24 hours
- Review Renovate PRs for breaking changes
- Check CI status before merging major updates

---

## Performance Optimization

### CI Execution Time

**Target**: <10 minutes for full pipeline

**Current Performance**:
- Change Detection: 5s
- Code Quality: 45s (skipped on docs-only)
- Test Matrix: 3-5min (parallel)
- Build: 47s
- Integration: 2-3min (parallel)
- Total: 5-8min (code changes), 1min (docs-only)

**Optimization Strategies**:
- ✅ Parallel matrix execution (`fail-fast: false`)
- ✅ Pytest parallelization (`-n auto`)
- ✅ Smart change detection (skip unnecessary jobs)
- ✅ Aggressive caching (pip, pytest, pre-commit)
- ✅ Artifact reuse (build once, test multiple times)

---

## Security Compliance

### Vulnerability Management

**Detection**:
- Renovate: Dependency vulnerabilities
- Bandit: Python security issues
- pip-audit: Known CVEs
- CodeQL: Code-level security flaws

**Response Time**:
- Critical: Auto-merge within 24 hours
- High: Manual review within 48 hours
- Medium: Next sprint
- Low: Backlog

### License Compliance

**Allowed Licenses**:
- MIT, Apache-2.0, BSD-3-Clause
- Python Software Foundation
- ISC, Unlicense

**Prohibited**:
- GPL variants (copyleft)
- Proprietary/Commercial
- Unknown/Undeclared

---

## Maintenance

### Weekly Tasks

- ✅ Review Renovate dependency PRs
- ✅ Check status dashboard for anomalies
- ✅ Review security scan results
- ✅ Monitor performance benchmarks

### Monthly Tasks

- ✅ Review and update workflow configurations
- ✅ Audit security scan exclusions
- ✅ Update CodeQL query filters
- ✅ Review artifact retention policies
- ✅ Update documentation

### Quarterly Tasks

- ✅ Comprehensive security audit
- ✅ Performance baseline reset
- ✅ Workflow optimization review
- ✅ Dependency major version updates

---

## Resources

### Documentation
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Renovate Configuration](https://docs.renovatebot.com/configuration-options/)
- [CodeQL Query Reference](https://codeql.github.com/docs/)
- [pytest Documentation](https://docs.pytest.org/)

### Internal Links
- [Workflow Status Dashboard](.github/WORKFLOW_STATUS.md)
- [Security Policy](../SECURITY.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Changelog](../CHANGELOG.md)

### Support
- **Issues**: https://github.com/imewei/NLSQ/issues
- **Discussions**: https://github.com/imewei/NLSQ/discussions
- **Security**: wchen@anl.gov

---

*Last Updated: 2025-01-18*
