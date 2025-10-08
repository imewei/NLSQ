# GitHub Actions Schema Validation - Educational Guide

**Purpose**: Understand GitHub Actions schema validation and how to fix common issues
**Audience**: Developers working with GitHub Actions workflows
**Last Updated**: 2025-10-07

---

## Table of Contents

1. [What is Schema Validation?](#what-is-schema-validation)
2. [Why It Matters](#why-it-matters)
3. [Common Patterns](#common-patterns)
4. [Real-World Example: CodeQL Fix](#real-world-example-codeql-fix)
5. [Prevention Strategies](#prevention-strategies)

---

(what-is-schema-validation)=
## What is Schema Validation?

### The Basics

**JSON Schema** is a vocabulary that allows you to validate JSON/YAML structures. GitHub Actions workflows are validated against an official schema to catch errors before execution.

```yaml
# Schema defines what's valid
{
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "number"}
  },
  "required": ["name"]
}

# Valid YAML
name: "Alice"
age: 30

# Invalid YAML (schema violation)
name: 123        # Wrong type
nickname: "Ali"  # Unknown property
```

### GitHub Actions Schema

GitHub maintains an official workflow schema at:
- **Source**: [github-workflow.json](https://github.com/SchemaStore/schemastore/blob/master/src/schemas/json/github-workflow.json)
- **Validator**: `check-github-workflows` pre-commit hook
- **Purpose**: Catch configuration errors early

---

(why-it-matters)=
## Why It Matters

### The Cost of Schema Violations

| Stage | Without Validation | With Validation |
|-------|-------------------|-----------------|
| **Development** | 😊 No friction | ⚠️ Pre-commit warning |
| **Push** | ✅ Success | 🛑 Blocked (with hooks) |
| **CI Execution** | ❌ Workflow fails | ✅ Never reaches CI |
| **Debug Time** | 30-60 minutes | 2-5 minutes |
| **Incident Impact** | CI blocked for team | Individual fix |

**Real Cost**:
- **No validation**: Find errors in CI (slow feedback, blocks team)
- **With validation**: Find errors pre-commit (fast feedback, local fix)

### Benefits of Schema Validation

1. ✅ **Early Error Detection**: Catch issues before push
2. ✅ **Fast Feedback**: Seconds vs. minutes
3. ✅ **Team Productivity**: No CI blockage
4. ✅ **Learning Tool**: Understand correct patterns
5. ✅ **Documentation**: Schema is source of truth

---

(common-patterns)=
## Common Patterns

### Pattern 1: Action Input vs. Config Property

**Problem**: Confusing action-level inputs with config-level properties

```yaml
# ❌ WRONG: paths at action level
- uses: some-action@v1
  with:
    language: python
    paths:              # Not valid action input
      - src

# ✅ CORRECT: paths in config
- uses: some-action@v1
  with:
    language: python
    config: |
      paths:            # Valid config property
        - src
```

**Rule of Thumb**:
- **Action inputs** = Function parameters (fixed set)
- **Config properties** = Configuration file (flexible)

### Pattern 2: Type Mismatches

**Problem**: Providing wrong type for property

```yaml
# ❌ WRONG: Array where string expected
- uses: actions/checkout@v4
  with:
    ref: ['main', 'develop']  # ref expects string

# ✅ CORRECT: String value
- uses: actions/checkout@v4
  with:
    ref: main                 # String value
```

**Common Type Errors**:
- Array instead of string
- String instead of boolean
- Number instead of string
- Object instead of primitive

### Pattern 3: Required Properties Missing

**Problem**: Omitting required properties

```yaml
# ❌ WRONG: Missing required 'uses' or 'run'
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Do something
        # Missing 'uses' or 'run'

# ✅ CORRECT: Includes 'uses' or 'run'
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Build
        run: npm run build
```

### Pattern 4: Invalid Property Names

**Problem**: Using non-existent properties

```yaml
# ❌ WRONG: 'timeout' doesn't exist at job level
jobs:
  build:
    runs-on: ubuntu-latest
    timeout: 30          # Invalid property

# ✅ CORRECT: Use 'timeout-minutes'
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30  # Valid property
```

---

(real-world-example-codeql-fix)=
## Real-World Example: CodeQL Fix

### The Problem

Our CodeQL workflow failed schema validation:

```yaml
# ❌ WRONG Configuration
- name: Initialize CodeQL
  uses: github/codeql-action/init@v3
  with:
    languages: ${{ matrix.language }}
    queries: +security-and-quality    # Redundant
    config: |
      name: "Config"
      queries:
        - uses: security-extended
    paths:                            # ❌ Not valid here
      - 'nlsq'
    paths-ignore:                     # ❌ Not valid here
      - 'tests'
```

**Error Message**:
```
$.jobs.analyze.steps[2].with['paths-ignore']:
  ['tests', 'examples', 'docs'] is not of type 'string'
```

### The Analysis

**Root Cause**:
1. `paths` and `paths-ignore` are **config-level** properties
2. They were placed at **action-level** (inside `with:`)
3. Schema expects these in the `config:` block

**Mental Model**:
```
Action Level (with:)          Config Level (config: |)
├── languages ✅              ├── name ✅
├── queries ⚠️                ├── queries ✅
├── config ✅                 ├── paths ✅
├── paths ❌                  └── paths-ignore ✅
└── paths-ignore ❌
```

### The Solution

```yaml
# ✅ CORRECT Configuration
- name: Initialize CodeQL
  uses: github/codeql-action/init@v3
  with:
    languages: ${{ matrix.language }}
    config: |
      name: "NLSQ CodeQL Config"
      queries:
        - uses: security-extended
        - uses: security-and-quality
      paths:                          # ✅ Inside config
        - nlsq
      paths-ignore:                   # ✅ Inside config
        - tests
        - examples
        - docs
        - benchmark
```

**Key Changes**:
1. Moved `paths` into `config:` block
2. Moved `paths-ignore` into `config:` block
3. Removed redundant `queries:` at action level
4. All configuration now in single logical unit

### The Outcome

**Before**:
- ❌ Pre-commit: 23/24 hooks passing
- ⚠️ Schema validation failing
- 🔍 Risk of CI failures

**After**:
- ✅ Pre-commit: 24/24 hooks passing
- ✅ Schema validation passing
- ✅ CI running successfully

---

(prevention-strategies)=
## Prevention Strategies

### 1. Use Pre-Commit Hooks

**Setup** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/python-jsonschema/check-jsonschema
    rev: 0.27.0
    hooks:
      - id: check-github-workflows
        args: ["--verbose"]
```

**Usage**:
```bash
# Install hooks
$ pre-commit install

# Run manually
$ pre-commit run check-github-workflows --all-files

# Auto-runs on git commit
$ git commit -m "update workflow"
```

### 2. Validate Before Push

**Local Testing**:
```bash
# Method 1: Pre-commit
$ pre-commit run --all-files

# Method 2: act (test workflows locally)
$ act --dryrun -W .github/workflows/

# Method 3: yamllint
$ yamllint .github/workflows/
```

### 3. Use Official Examples

**Good Sources**:
- [GitHub Actions Examples](https://github.com/actions/starter-workflows)
- [Action-specific README](https://github.com/github/codeql-action)
- [GitHub Docs](https://docs.github.com/en/actions)

**Red Flags**:
- StackOverflow answers (may be outdated)
- Blog posts >1 year old
- Unofficial examples

### 4. Read Error Messages Carefully

**Error Structure**:
```
Schema validation errors were encountered.
  [FILE]::$.[JSON_PATH]: [DESCRIPTION]

Example:
  .github/workflows/ci.yml::$.jobs.test.steps[0].with['python-version']:
    [3.8, 3.9, 3.10] is not of type 'string'
```

**Decoding**:
1. **File**: `.github/workflows/ci.yml`
2. **Path**: `jobs.test.steps[0].with['python-version']`
3. **Issue**: Array provided, string expected
4. **Location**: Line number often shown in verbose mode

### 5. Understand Action Documentation

**Reading Action Docs**:
```yaml
# actions/checkout@v4 README shows:
Inputs:
  ref:
    description: 'Branch, tag or SHA to checkout'
    required: false
    type: string        # ← Note the type!

# Correct usage:
- uses: actions/checkout@v4
  with:
    ref: main           # String, not array
```

### 6. Test in Isolation

**When Making Changes**:
1. Change one workflow at a time
2. Test locally with pre-commit
3. Push to feature branch first
4. Verify in CI before merging to main

### 7. Use Workflow Syntax Highlighting

**Editor Setup**:
- **VS Code**: YAML extension + GitHub Actions extension
- **IntelliJ**: GitHub Actions support built-in
- **Vim**: vim-yaml + coc-yaml

**Benefits**:
- Real-time syntax checking
- Auto-completion for properties
- Inline documentation
- Type hints

---

## Common Pitfalls & Solutions

### Pitfall 1: Copying Old Examples

**Problem**: Workflow syntax evolves over time

**Solution**:
- Check action version (e.g., `@v3` vs `@v4`)
- Refer to tagged README for that version
- Use `dependabot` to update actions

### Pitfall 2: Mixing Patterns

**Problem**: Combining incompatible configuration methods

**Solution**:
- Choose one config method and stick to it
- Don't mix inline config with external file
- Don't duplicate settings at multiple levels

### Pitfall 3: Ignoring Warnings

**Problem**: Dismissing schema warnings as "false positives"

**Solution**:
- Schema warnings are rarely wrong
- Investigate root cause
- Fix properly, don't disable validation

### Pitfall 4: Not Testing Locally

**Problem**: Pushing without pre-commit checks

**Solution**:
- Install pre-commit hooks
- Run `pre-commit run --all-files` before push
- Use `--no-verify` only in emergencies

---

## Quick Reference

### Validation Checklist

✅ **Before Commit**:
- [ ] Run pre-commit hooks
- [ ] Check YAML syntax
- [ ] Verify action versions
- [ ] Test key changes locally

✅ **After Push**:
- [ ] Monitor workflow execution
- [ ] Check for deprecation warnings
- [ ] Verify expected behavior
- [ ] Document any workarounds

### Emergency Fixes

**Workflow Failing in CI**:
```bash
# 1. Pull latest
$ git pull origin main

# 2. Check schema locally
$ pre-commit run check-github-workflows --all-files --verbose

# 3. Fix issues
$ vim .github/workflows/problem.yml

# 4. Verify fix
$ pre-commit run check-github-workflows --all-files

# 5. Push fix
$ git add .github/workflows/problem.yml
$ git commit -m "fix: schema validation in workflow"
$ git push origin main
```

### Resources

| Resource | URL | Purpose |
|----------|-----|---------|
| **Workflow Schema** | [SchemaStore](https://github.com/SchemaStore/schemastore) | Official schema |
| **GitHub Docs** | [Actions Docs](https://docs.github.com/en/actions) | Documentation |
| **Marketplace** | [Actions Marketplace](https://github.com/marketplace?type=actions) | Find actions |
| **Starter Workflows** | [Starter Workflows](https://github.com/actions/starter-workflows) | Templates |
| **This Project** | `docs/codeql_workflow_fix.md` | Detailed example |

---

## Learning Exercises

### Exercise 1: Identify the Error

```yaml
# What's wrong with this workflow?
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: "0"    # Error here
```

<details>
<summary>Answer</summary>

**Error**: `fetch-depth` expects a number, not a string.

**Fix**:
```yaml
with:
  fetch-depth: 0    # Number, not string
```
</details>

### Exercise 2: Move to Config

```yaml
# Convert this to use config block
- uses: some-action@v1
  with:
    setting1: value1
    setting2: value2
```

<details>
<summary>Answer</summary>

```yaml
- uses: some-action@v1
  with:
    config: |
      setting1: value1
      setting2: value2
```

Note: Only do this if the action supports config block!
</details>

---

## Summary

**Key Takeaways**:
1. 📚 Schema validation catches errors early
2. 🔍 Understand action inputs vs. config properties
3. ✅ Use pre-commit hooks for local validation
4. 📖 Read official documentation, not blog posts
5. 🧪 Test workflows before pushing to main
6. 🎯 Fix errors properly, don't disable validation

**Remember**: Schema validation is a **tool**, not a **barrier**. It helps you write correct workflows faster.

---

*Document Version: 1.0*
*Last Updated: 2025-10-07*
*Maintainer: Development Team*
