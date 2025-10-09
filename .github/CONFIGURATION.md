# GitHub Configuration

This directory contains GitHub-specific configuration files for repository automation, issue management, and CI/CD workflows.

## 📁 Directory Structure

```
.github/
├── workflows/           # GitHub Actions CI/CD workflows
│   ├── ci.yml          # Main CI pipeline (required)
│   ├── docs.yml        # Documentation builds
│   ├── security.yml    # Security scanning
│   ├── benchmark.yml   # Performance testing
│   ├── publish.yml     # PyPI publishing
│   └── README.md       # Workflows documentation
├── ISSUE_TEMPLATE/     # Issue report templates
│   ├── bug_report.yml
│   ├── feature_request.yml
│   ├── performance_issue.yml
│   └── config.yml
├── CODEOWNERS          # Code review assignments
├── dependabot.yml      # Dependency update automation
└── pull_request_template.md  # PR template

## 🔧 Configuration Files

### CODEOWNERS

Automatic code review assignment. All paths are owned by @imewei.

**Critical paths with required review:**
- `/nlsq/` - Core library code
- `/nlsq/trf.py`, `/nlsq/least_squares.py` - Critical algorithms
- `/.github/workflows/` - CI/CD infrastructure
- `/.github/workflows/publish.yml` - Publishing (security-sensitive)

**Documentation:** https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners

### dependabot.yml

Automated dependency updates via Dependabot.

**Update Schedule:**
- GitHub Actions: Weekly (Monday 2 AM UTC)
- Python packages: Weekly (Monday 2 AM UTC)
- Grouped updates: JAX stack, NumPy/SciPy, testing tools, linting tools

**Strategy:**
- Major version updates: Ignored (require manual review)
- Minor/patch updates: Grouped by ecosystem
- Security updates: Applied to all dependencies
- Pre-releases: Ignored for stability

**Documentation:** https://docs.github.com/en/code-security/dependabot/dependabot-version-updates

### pull_request_template.md

Structured PR template with comprehensive checklist.

**Sections:**
- Description and type of change
- Related issues
- Testing (coverage, manual, benchmarks)
- Performance impact
- Breaking changes
- Code quality checklist
- Documentation checklist
- CI/CD checklist
- Reviewer notes

### Issue Templates

Structured templates for bug reports, feature requests, and performance issues.

**Templates:**
1. **Bug Report** - Structured bug reporting with environment capture
2. **Feature Request** - Feature proposals with use cases
3. **Performance Issue** - Benchmark reports with profiling data
4. **Config** - Links to documentation, discussions, and security reporting

**Features:**
- Required fields with validation
- Auto-labeling (`bug`, `enhancement`, `performance`)
- Auto-assignment to @imewei
- Environment information capture (NLSQ version, Python, JAX, platform)

## 🚀 Workflows

See [workflows/README.md](workflows/README.md) for detailed documentation.

**Active Workflows:**
- **ci.yml** (8 min) - Main CI with cross-platform testing
- **docs.yml** (3 min) - Sphinx documentation builds
- **security.yml** (10 min) - CodeQL, dependency audit, Bandit
- **benchmark.yml** (15 min) - Performance testing and regression detection
- **publish.yml** (10 min) - Automated PyPI publishing

## 📊 Quick Stats

| File | Lines | Purpose |
|------|-------|---------|
| `workflows/ci.yml` | 182 | Main CI pipeline |
| `workflows/docs.yml` | 68 | Documentation |
| `workflows/security.yml` | 134 | Security scanning |
| `workflows/benchmark.yml` | 108 | Benchmarks |
| `workflows/publish.yml` | 345 | PyPI publishing |
| `dependabot.yml` | 87 | Dependency automation |
| `CODEOWNERS` | 36 | Code review |
| `pull_request_template.md` | 151 | PR template |
| **Total** | **1,111** | All config |

## 🔒 Security

### Protected Files

These files require explicit code owner approval:
- `/.github/workflows/publish.yml` - PyPI publishing (security-critical)
- `/.github/dependabot.yml` - Dependency automation
- `/nlsq/` - Core library code

### Security Features

- ✅ **CodeQL Analysis** - Weekly + PR scanning
- ✅ **Dependabot** - Automated security updates
- ✅ **Bandit** - Python security linter
- ✅ **pip-audit** - Dependency vulnerability scanning
- ✅ **Secret scanning** - GitHub's built-in scanning (enable in Settings)
- ✅ **CODEOWNERS** - Required reviews for sensitive files

### Enabling Additional Security

1. Go to: Settings → Security & analysis
2. Enable:
   - Dependabot alerts
   - Dependabot security updates
   - Secret scanning
   - Secret scanning push protection

## 📝 Best Practices

### For Contributors

1. **Always create issues before PRs** (except for trivial fixes)
2. **Fill out PR template completely**
3. **Run pre-commit hooks** before pushing (`pre-commit run --all-files`)
4. **Add tests** for all new functionality
5. **Update documentation** when changing behavior

### For Maintainers

1. **Review CODEOWNERS assignments** when structure changes
2. **Monitor Dependabot PRs** weekly
3. **Review security alerts** immediately
4. **Keep workflows updated** with latest actions
5. **Update issue templates** when new patterns emerge

## 🔗 External Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)
- [Issue Forms Syntax](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms)
- [CODEOWNERS Syntax](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)

## 📮 Contact

- **Maintainer**: Wei Chen (@imewei)
- **Email**: wchen@anl.gov
- **Security**: https://github.com/imewei/NLSQ/security/advisories/new

---

**Last Updated**: 2025-10-09
**Configuration Version**: 2.0 (Modern workflows)
