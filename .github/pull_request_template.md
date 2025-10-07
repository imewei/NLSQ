## Description

<!-- Provide a clear and concise description of your changes -->

## Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 Documentation update
- [ ] 🎨 Code style/refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Test coverage improvement
- [ ] 🔧 CI/CD or infrastructure change
- [ ] 🔒 Security fix

## Related Issues

<!-- Link to related issues using #issue_number -->

Fixes #
Relates to #

## Changes Made

<!-- List the specific changes made in this PR -->

-
-
-

## Testing

<!-- Describe the tests you ran and their results -->

### Test Coverage

- [ ] Added tests for new functionality
- [ ] All existing tests pass (`make test`)
- [ ] Coverage maintained or improved

### Manual Testing

<!-- Describe any manual testing performed -->

```python
# Example test code or usage
```

**Results**:
-

## Performance Impact

<!-- If applicable, describe any performance implications -->

- [ ] No performance impact
- [ ] Performance improved (include benchmark results)
- [ ] Performance decreased (justify why acceptable)

**Benchmark Results** (if applicable):

```
# Paste benchmark output here
```

## Breaking Changes

<!-- If this is a breaking change, describe the impact and migration path -->

**Impact**:
-

**Migration Guide**:
-

## Checklist

<!-- Mark completed items with an 'x' -->

### Code Quality

- [ ] Code follows project style guidelines (pre-commit hooks pass)
- [ ] Self-review completed
- [ ] Code is well-commented, especially complex logic
- [ ] No unnecessary debug statements or commented code
- [ ] Type hints added/updated where appropriate

### Testing

- [ ] Tests added for new functionality
- [ ] All tests pass locally (`make test`)
- [ ] Coverage threshold met (≥70%)
- [ ] Integration tests pass (if applicable)

### Documentation

- [ ] Updated relevant documentation (docstrings, README, docs/)
- [ ] Updated CLAUDE.md (if architecture changed)
- [ ] Added/updated examples (if new feature)
- [ ] Updated CHANGELOG.md

### CI/CD

- [ ] All CI checks pass
- [ ] No security vulnerabilities introduced (bandit passes)
- [ ] Package builds successfully
- [ ] Pre-commit hooks pass

### Performance

- [ ] Benchmarks run (if performance-critical code)
- [ ] No memory leaks introduced
- [ ] No unnecessary allocations in hot paths

## Additional Context

<!-- Add any other context, screenshots, or information about the PR -->

## Reviewer Notes

<!-- Notes for reviewers about what to focus on -->

**Focus Areas**:
-

**Questions for Reviewers**:
-

---

## For Maintainers

<!-- Maintainer checklist - ignore if you're not a maintainer -->

### Before Merge

- [ ] PR title follows conventional commits format
- [ ] All conversations resolved
- [ ] No merge conflicts
- [ ] Squash/rebase if needed for clean history
- [ ] Version bump appropriate (if applicable)

### After Merge

- [ ] Update project board
- [ ] Close related issues
- [ ] Update release notes (if needed)
- [ ] Backport to release branches (if needed)
