# GitHub Actions - Developer Guide

**Purpose**: Understand GitHub Actions workflow patterns and best practices
**Audience**: Developers working with GitHub Actions workflows
**Last Updated**: 2025-12-23

---

## Overview

NLSQ uses GitHub Actions for CI/CD with two main workflows:

- **main.yml**: Test suite, linting, type checking, coverage
- **release.yml**: Package publishing to PyPI

---

## Workflow Schema Validation

### What is Schema Validation?

GitHub Actions workflows are validated against an official JSON schema. The `check-github-workflows` pre-commit hook catches configuration errors before they reach CI.

### Benefits

1. **Early Error Detection**: Catch issues before push
2. **Fast Feedback**: Seconds vs. minutes
3. **Team Productivity**: No CI blockage from syntax errors

### Validation with Pre-commit

```bash
# Validate all workflow files
pre-commit run check-github-workflows --all-files

# Run all pre-commit checks
pre-commit run --all-files
```

---

## Common Patterns

### Pattern 1: Action Input vs. Config Property

Action inputs are fixed; config properties are flexible:

```yaml
# Action inputs go under 'with:'
- uses: actions/checkout@v4
  with:
    fetch-depth: 0

# Complex configs use inline YAML
- uses: some-action@v1
  with:
    config: |
      setting1: value1
      setting2: value2
```

### Pattern 2: Type Mismatches

```yaml
# WRONG: Array where string expected
- uses: actions/checkout@v4
  with:
    ref: ['main', 'develop']  # ref expects string

# CORRECT: String value
- uses: actions/checkout@v4
  with:
    ref: main
```

### Pattern 3: Matrix Strategies

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.12', '3.13']
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
```

---

## NLSQ Workflow Structure

### main.yml Key Jobs

1. **lint**: Runs ruff and mypy
2. **test**: Runs pytest on multiple Python versions
3. **coverage**: Reports test coverage

### release.yml Key Jobs

1. **build**: Creates distribution packages
2. **publish**: Uploads to PyPI on tag push

---

## Local Testing

```bash
# Validate workflow syntax
pre-commit run check-github-workflows --all-files

# Test workflow locally (requires 'act')
act --dryrun -W .github/workflows/main.yml

# Validate YAML syntax
yamllint .github/workflows/
```

---

## Best Practices

1. **Pin action versions**: Use `@v4` not `@latest`
2. **Test locally first**: Run pre-commit before pushing
3. **Use matrix builds**: Test across Python versions/OS
4. **Cache dependencies**: Speed up builds with caching
5. **Set timeouts**: Prevent runaway jobs

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [SchemaStore](https://github.com/SchemaStore/schemastore) - Official workflow schema
