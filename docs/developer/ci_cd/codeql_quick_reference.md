# CodeQL Configuration - Quick Reference Card

**Last Updated**: 2025-10-07
**For**: Developers modifying `.github/workflows/codeql.yml`

---

## ✅ Correct Pattern (Use This)

```yaml
- name: Initialize CodeQL
  uses: github/codeql-action/init@v3
  with:
    languages: ${{ matrix.language }}
    config: |
      name: "Your Config Name"
      queries:
        - uses: security-extended
        - uses: security-and-quality
      paths:                      # ✅ Inside config block
        - src
        - lib
      paths-ignore:               # ✅ Inside config block
        - tests
        - docs
        - examples
```

---

## ❌ Anti-Pattern (Don't Use)

```yaml
- name: Initialize CodeQL
  uses: github/codeql-action/init@v3
  with:
    languages: ${{ matrix.language }}
    queries: +security-and-quality    # ❌ Redundant
    config: |
      ...
    paths:                              # ❌ Wrong level
      - src
    paths-ignore:                       # ❌ Wrong level
      - tests
```

---

## 🎯 Configuration Levels

```
Action Level (with:)        Config Level (config: |)
├─ languages ✅             ├─ name ✅
├─ config ✅                ├─ queries ✅
└─ (other inputs)           ├─ paths ✅
                            ├─ paths-ignore ✅
                            ├─ query-filters ✅
                            └─ (other config)
```

---

## 🧪 Testing Commands

```bash
# Validate workflow schema
pre-commit run check-github-workflows --all-files

# Run all pre-commit checks
pre-commit run --all-files

# Test workflow locally (requires 'act')
act --dryrun -W .github/workflows/codeql.yml

# Validate YAML syntax
yamllint .github/workflows/codeql.yml
```

---

## 🔍 Common Configuration Options

### Query Suites

```yaml
queries:
  - uses: security-extended          # All security queries
  - uses: security-and-quality       # Security + quality
  - uses: ./custom-queries           # Local queries
```

### Path Patterns

```yaml
paths:
  - src                               # Directory
  - "**/*.py"                         # Glob pattern
  - lib/core                          # Nested directory

paths-ignore:
  - tests                             # Exclude tests
  - "**/*_test.py"                    # Exclude test files
  - "**/generated/**"                 # Exclude generated code
```

### Multi-Language Setup

```yaml
config: |
  name: "Multi-language Config"
  queries:
    - uses: security-and-quality
  paths:
    - python: src/backend
    - javascript: src/frontend
  paths-ignore:
    - tests
    - docs
```

---

## 🚨 Troubleshooting

### Error: Schema Validation Failed

**Solution**: Move `paths` and `paths-ignore` into `config:` block

### Error: Paths Not Being Filtered

**Check**:
1. Paths are relative to repo root
2. No typos in directory names
3. Config block has proper indentation

**Debug**: Check initialization logs in GitHub Actions

### Error: Queries Not Running

**Check**:
1. Query format: `- uses: query-name`
2. No redundant `queries:` at action level
3. Query packs are available

---

## 📊 Quick Decision Tree

```
Need to configure paths/queries?
│
├─ Simple setup (1-2 languages)
│  └─ Use inline config ✅
│
└─ Complex setup (3+ languages, many rules)
   └─ Consider separate config file
```

---

## 📚 Documentation Links

- **Full Guide**: `docs/codeql_workflow_fix.md`
- **Educational**: `docs/github_actions_schema_guide.md`
- **GitHub Docs**: https://docs.github.com/en/code-security/code-scanning
- **CodeQL Action**: https://github.com/github/codeql-action

---

## 🔄 Workflow for Changes

1. **Edit** `.github/workflows/codeql.yml`
2. **Test** `pre-commit run --all-files`
3. **Commit** with descriptive message
4. **Push** to remote
5. **Verify** in GitHub Actions UI
6. **Monitor** first workflow run

---

## 💡 Best Practices

✅ **DO**:
- Consolidate config in one place
- Test locally before pushing
- Use version pins (`@v3` not `@latest`)
- Document non-obvious choices
- Keep query lists organized

❌ **DON'T**:
- Mix action-level and config-level settings
- Duplicate query specifications
- Disable schema validation
- Push untested changes to main
- Use `@latest` in production

---

## 🆘 Emergency Contacts

- **CI/CD Issues**: DevOps Team
- **Security Findings**: Security Team
- **Questions**: See `docs/codeql_workflow_fix.md`

---

**TL;DR**: Keep paths and queries inside `config:` block, test with pre-commit, refer to full docs for details.
