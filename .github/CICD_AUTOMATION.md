# CI/CD Automation Guide

Complete guide to NLSQ's optimized CI/CD workflows, from development to production deployment.

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

NLSQ implements enterprise-grade CI/CD automation with an optimized 3-workflow architecture:

- ✅ **12 Parallel Test Jobs** across 3 platforms × 2 Python versions × 2 test types
- ✅ **Consolidated Pipeline** - 62.5% reduction in workflows (8 → 3)
- ✅ **Automated Dependency Updates** with Renovate Bot
- ✅ **Performance Benchmarking** with historical tracking
- ✅ **Advanced Security Scanning** with CodeQL
- ✅ **Semantic Release Automation** with changelog generation
- ✅ **Real-time Status Dashboard** with workflow monitoring

### Architecture Highlights

**Optimization Results**:
- 40-50% faster execution through unified setup and smart job orchestration
- Reduced GitHub Actions minutes consumption
- Simplified maintenance with 3 workflows instead of 8
- Improved developer experience with clearer workflow structure

### Automation Matrix

| Feature | Status | Trigger | Frequency |
|---------|--------|---------|-----------|
| Main CI/CD Pipeline | ✅ Active | Push, PR | On-demand |
| CodeQL Analysis | ✅ Active | Schedule | Weekly (Monday 2AM) |
| Performance Benchmarks | ✅ Active | Schedule | Weekly (Sunday 3AM) |
| Security Scan (Fast) | ✅ Active | Push, PR | On-change |
| Security Scan (Comprehensive) | ✅ Active | Schedule | Daily (1AM) |
| Dependency Updates | ✅ Active | Schedule | Weekdays 10PM-5AM |
| Status Dashboard | ✅ Active | Schedule | Every 6 hours |
| Documentation Build | ✅ Active | Push, PR | On-change |
| Release Automation | ✅ Active | Push to main | On-demand |

---

## Workflow Architecture

NLSQ uses a **consolidated 3-workflow architecture** optimized for performance and maintainability:

1. **`main.yml`** - Unified CI/CD pipeline for all PR and push events
2. **`scheduled.yml`** - Consolidated scheduled tasks (security, performance, monitoring)
3. **`release.yml`** - Semantic versioning and automated releases

### Consolidation Benefits

**Before**: 8 separate workflows with redundant setup steps
**After**: 3 optimized workflows with shared job orchestration

**Improvements**:
- ⚡ 40-50% faster execution
- 💰 Reduced GitHub Actions minutes
- 🔧 Simpler maintenance
- 📊 Better visibility

---

### 1. Main CI/CD Pipeline (`main.yml`)

**Purpose**: Unified pipeline for all PR and push events, replacing 4 previous workflows

**Consolidates**:
- Former `ci.yml` - Testing and coverage
- Former `docs.yml` - Documentation building
- Former `security.yml` - Fast security scanning
- Former `validate-notebooks.yml` - Notebook validation

**Job Execution Waves**:

```
Wave 1 (Parallel):
├─ detect-changes       # Smart change detection
└─ validate-deps        # Dependency validation

Wave 2 (Parallel, Conditional):
├─ quality              # Linting, type checking, complexity
├─ security-fast        # Bandit, pip-audit
└─ validate-notebooks   # Notebook structure & execution

Wave 3 (Parallel):
└─ test-matrix          # 6 jobs: 3 platforms × 2 Python versions

Wave 4:
└─ build                # Package building

Wave 5 (Parallel):
├─ docs                 # Sphinx documentation
└─ integration-matrix   # 6 jobs: 3 platforms × 2 Python versions

Wave 6:
└─ summary              # Final validation gate
```

**Smart Features**:
- **Change Detection**: Skip expensive jobs for docs-only changes
- **Parallel Execution**: 12 test jobs run simultaneously
- **Artifact Reuse**: Build once, test multiple times
- **Conditional Jobs**: Only run what's necessary

**Triggers**: `push`, `pull_request`, `workflow_dispatch`
**Duration**:
- Code changes: ~5-6 minutes (down from 8-10 minutes)
- Docs-only: ~1-2 minutes
**Artifacts**: Test results, coverage reports, build packages, documentation (7-30 day retention)

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

### 2. Scheduled Tasks (`scheduled.yml`)

**Purpose**: Consolidated scheduled workflows for deep analysis and monitoring

**Consolidates**:
- Former `codeql.yml` - Advanced security scanning
- Former `performance.yml` - Performance benchmarking
- Former `status-dashboard.yml` - Workflow monitoring

**Jobs**:

#### CodeQL Analysis
- **Schedule**: Weekly Monday 2AM UTC
- **Duration**: ~10-15 minutes
- **Features**: Security-extended queries, SARIF upload, GitHub Security integration

#### Performance Benchmarks
- **Schedule**: Weekly Sunday 3AM UTC
- **Duration**: ~15-20 minutes
- **Features**: CPU/GPU benchmarks, regression detection (>150%), historical tracking

#### Memory Profiling
- **Schedule**: Weekly Sunday 3AM UTC
- **Duration**: ~5-10 minutes
- **Features**: Memory usage analysis with retention

#### Comprehensive Security Scan
- **Schedule**: Daily 1AM UTC
- **Duration**: ~5-8 minutes
- **Tools**: Bandit, pip-audit, Safety, Semgrep, license compliance

#### Status Dashboard
- **Schedule**: Every 6 hours
- **Duration**: ~2-3 minutes
- **Features**: Workflow status, metrics, auto-commit updates

**Triggers**: `schedule`, `workflow_dispatch` (with task selection)
**Manual Execution**:
```bash
gh workflow run scheduled.yml -f task=codeql
gh workflow run scheduled.yml -f task=performance
gh workflow run scheduled.yml -f task=security
gh workflow run scheduled.yml -f task=dashboard
gh workflow run scheduled.yml -f task=all
```

---

### 3. Release Automation (`release.yml`)

**Purpose**: Semantic versioning and automated releases (unchanged from previous architecture)

**Features**:
- Conventional commit parsing
- Automatic version bumping
- CHANGELOG.md generation
- GitHub release creation
- PyPI package publishing

**Triggers**: `push (main)`, `workflow_dispatch`
**Version Strategy**: Semantic versioning (MAJOR.MINOR.PATCH)
**Duration**: ~5-7 minutes

---

### 4. Dependency Updates (`renovate.json`)

**Purpose**: Automated dependency updates with intelligent grouping (unchanged but optimized for new workflows)

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

**Target**: <6 minutes for full pipeline

**Consolidated Architecture Performance**:

| Workflow | Previous | Optimized | Improvement |
|----------|----------|-----------|-------------|
| Main CI/CD (code changes) | 8-10 min | 5-6 min | 40-50% faster |
| Main CI/CD (docs-only) | 2-3 min | 1-2 min | 33-50% faster |
| Scheduled tasks | 25-30 min | 20-25 min | 20% faster |

**Job Execution Breakdown** (main.yml):
- Wave 1 (Parallel): 30s - detect-changes, validate-deps
- Wave 2 (Parallel): 60-90s - quality, security-fast, validate-notebooks
- Wave 3 (Parallel): 3-4min - test-matrix (6 jobs)
- Wave 4: 45s - build
- Wave 5 (Parallel): 2-3min - docs, integration-matrix (6 jobs)
- Wave 6: 10s - summary
- **Total**: ~5-6 minutes (code changes)

**Optimization Strategies**:
- ✅ **Workflow Consolidation**: Single setup for multiple jobs (8 setups → 1 setup)
- ✅ **Wave-based Execution**: Jobs grouped by dependencies for optimal parallelization
- ✅ **Parallel matrix execution**: `fail-fast: false` for all test jobs
- ✅ **Pytest parallelization**: `-n auto` for CPU-core utilization
- ✅ **Smart change detection**: Skip expensive jobs for docs-only changes
- ✅ **Aggressive caching**: uv cache for dependencies
- ✅ **Artifact reuse**: Build once, test multiple times
- ✅ **Conditional execution**: Only run necessary jobs based on change type

**Resource Efficiency**:
- GitHub Actions minutes saved: ~40% reduction
- Fewer concurrent workflow runs
- Reduced artifact storage through shorter retention
- Optimized scheduled task distribution

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

*Last Updated: 2025-01-19 - Optimized to 3-workflow consolidated architecture*
