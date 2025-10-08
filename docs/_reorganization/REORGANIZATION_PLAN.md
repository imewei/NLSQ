# NLSQ Documentation Reorganization Plan

**Date**: 2025-10-08
**Status**: Implementation Ready
**Sphinx Version**: Using sphinx_rtd_theme, RST-only configuration

## Executive Summary

Reorganizing `/home/wei/Documents/GitHub/nlsq/docs/` to create a clean, maintainable Sphinx documentation structure that:
- Separates user documentation from developer documentation
- Consolidates duplicate content (migration guides, performance docs)
- Archives historical development logs appropriately
- Maintains backward compatibility with existing builds
- Follows Sphinx best practices for professional documentation

## Current State Analysis

### Directory Structure (Before)
```
docs/
├── Root: 17 files (9 RST, 7 MD, 1 txt)
├── user_guides/: 5 MD files (NOT in Sphinx build - invisible!)
├── tutorials/: 2 RST files (properly integrated)
├── autodoc/: 11 RST files (auto-generated API docs)
├── archive/: 4 MD files (historical reports)
├── development/: 28 MD files in phase1-4 subdirs (historical logs)
├── _static/: Sphinx static files
├── images/: 1 logo file
└── _build/: 24MB build artifacts (gitignored ✓)
```

### Critical Issues Identified

#### 1. **DUPLICATE CONTENT**
- **Migration Guide** (2 versions):
  - `migration_guide.rst` (467 lines, Sept 25) - Internal NLSQ features
  - `user_guides/migration_guide.md` (857 lines, Oct 8) - **NEWER**, comprehensive SciPy→NLSQ
  - **Decision**: Use MD version (newer, more comprehensive)

#### 2. **INVISIBLE DOCUMENTATION**
- `user_guides/` directory (5 MD files, 96KB) NOT in Sphinx build:
  - Reason: `conf.py` line 62 disables myst_parser
  - `source_suffix = {".rst": None}` (line 129)
  - **Impact**: Recent, high-quality docs are not accessible to users!

#### 3. **MISNAMED FILE**
- `user_guides/quick_start.md` is actually a **developer implementation guide**, NOT a user quickstart
- Real user quickstart: `tutorials/quickstart.rst`

#### 4. **ORGANIZATIONAL CHAOS**
- Root directory: Mixed user docs + developer docs (no clear separation)
- 7 MD files at root (all developer-focused): optimization_case_study, performance_tuning_guide, CI/CD docs, CodeQL guides, PyPI setup
- No clear navigation hierarchy

#### 5. **HISTORICAL BLOAT**
- `development/`: 28 files, 11K lines, 400KB of daily summaries
- Many daily logs superseded by sprint completion summaries
- Valuable information buried in verbose logs

### Content Classification

#### User-Facing Documentation (High Priority)
| File | Type | Status | Quality |
|------|------|--------|---------|
| `tutorials/quickstart.rst` | RST | ✓ In build | Excellent |
| `tutorials/large_datasets.rst` | RST | ✓ In build | Excellent |
| `user_guides/migration_guide.md` | MD | ✗ Not in build | Excellent (newer) |
| `user_guides/advanced_features.md` | MD | ✗ Not in build | Excellent |
| `user_guides/performance_optimization.md` | MD | ✗ Not in build | Excellent |
| `user_guides/troubleshooting.md` | MD | ✗ Not in build | Excellent |
| `installation.rst` | RST | ✓ In build | Good |
| `large_dataset_guide.rst` | RST | ✓ In build | Good (overlaps tutorials) |
| `advanced_features.rst` | RST | ✓ In build | Good (older than MD version) |
| `migration_guide.rst` | RST | ✓ In build | Superseded by MD version |

#### Developer Documentation (Medium Priority)
| File | Type | Location | Recent? |
|------|------|----------|---------|
| `optimization_case_study.md` | MD | Root | Oct 6 ✓ |
| `performance_tuning_guide.md` | MD | Root | Oct 6 ✓ |
| `CI_CD_IMPROVEMENTS_SUMMARY.md` | MD | Root | Oct 7 ✓ |
| `codeql_workflow_fix.md` | MD | Root | Oct 7 ✓ |
| `github_actions_schema_guide.md` | MD | Root | Oct 7 ✓ |
| `QUICK_REFERENCE_CODEQL.md` | MD | Root | Oct 7 ✓ |
| `PYPI_SETUP.md` | MD | Root | Sept 25 ✓ |

#### Historical Archives (Low Priority)
| Directory | Files | Size | Action |
|-----------|-------|------|--------|
| `development/phase1/` | 16 files | ~200KB | Consolidate |
| `development/phase2/` | 3 files | ~50KB | Keep summaries only |
| `development/phase3/` | 3 files | ~50KB | Keep summaries only |
| `development/phase4/` | 2 files | ~10KB | Keep |
| `development/planning/` | 3 files | ~40KB | Keep |
| `archive/` | 4 files | 100KB | Keep as-is |

## Proposed Structure (After)

### Target Organization
```
docs/
├── index.rst                          # UPDATED: New structure
├── conf.py                            # UPDATED: Enable myst_parser
├── Makefile, make.bat, requirements.txt
├── _static/
├── images/
│
├── getting_started/                   # NEW SECTION
│   ├── installation.rst               # MOVED from root
│   └── quickstart.rst                 # MOVED from tutorials/
│
├── guides/                            # NEW SECTION (User Guides)
│   ├── migration_scipy.rst            # CONVERTED from user_guides/migration_guide.md
│   ├── advanced_features.rst          # KEPT (root version) OR CONVERTED from user_guides/
│   ├── performance_guide.rst          # CONVERTED from user_guides/performance_optimization.md
│   ├── troubleshooting.rst            # CONVERTED from user_guides/troubleshooting.md
│   └── large_datasets.rst             # MOVED from tutorials/
│
├── api/                               # RENAMED from autodoc/
│   ├── modules.rst                    # MOVED from autodoc/
│   ├── nlsq.*.rst                     # MOVED from autodoc/
│   └── large_datasets_api.rst         # MOVED from api_large_datasets.rst
│
├── developer/                         # NEW SECTION
│   ├── optimization_case_study.md     # MOVED from root (keep MD with myst)
│   ├── performance_tuning.md          # MOVED from root (keep MD with myst)
│   ├── ci_cd/                         # NEW: CI/CD subsection
│   │   ├── improvements_summary.md    # MOVED from CI_CD_IMPROVEMENTS_SUMMARY.md
│   │   ├── codeql_workflow_fix.md     # MOVED from root
│   │   ├── github_actions_guide.md    # MOVED from root
│   │   └── codeql_quick_reference.md  # MOVED from root
│   └── pypi_setup.md                  # MOVED from root (keep MD with myst)
│
├── history/                           # NEW SECTION (Historical)
│   ├── v0.1.1_sprint/                 # CONSOLIDATED from development/
│   │   ├── README.md                  # MOVED from development/README.md
│   │   ├── sprint1_completion.md      # MOVED from phase1/
│   │   ├── sprint2_completion.md      # MOVED from phase2/
│   │   ├── sprint3_completion.md      # MOVED from phase3/
│   │   ├── sprint4_plan.md            # MOVED from phase4/
│   │   └── roadmap.md                 # MOVED from planning/
│   └── archived_reports/              # MOVED from archive/
│       ├── codebase_analysis.md
│       ├── sprint_1_2_completion.md
│       ├── sprint_3_completion.md
│       └── test_generation_phase2.md
│
└── _build/                            # (gitignored)

REMOVED:
├── user_guides/                       # REMOVED (content moved to guides/)
├── tutorials/                         # REMOVED (content moved to getting_started/ + guides/)
├── autodoc/                           # REMOVED (renamed to api/)
├── development/                       # REMOVED (consolidated to history/v0.1.1_sprint/)
├── archive/                           # REMOVED (moved to history/archived_reports/)
├── migration_guide.rst                # REMOVED (superseded by guides/migration_scipy.rst)
├── large_dataset_guide.rst            # REMOVED (merged with guides/large_datasets.rst)
└── development/phase*/DAY*.md         # REMOVED (16 daily logs superseded by sprint summaries)
```

### Navigation Hierarchy (index.rst)

```
NLSQ Documentation
├── Getting Started
│   ├── Installation
│   └── Quickstart Tutorial
│
├── User Guides
│   ├── Migrating from SciPy
│   ├── Advanced Features
│   ├── Performance Optimization
│   ├── Large Datasets
│   └── Troubleshooting
│
├── API Reference
│   ├── Core Modules
│   └── Large Dataset API
│
├── Developer Documentation
│   ├── Optimization Case Study
│   ├── Performance Tuning
│   ├── CI/CD Guides
│   └── PyPI Publishing
│
└── Project History
    ├── v0.1.1 Feature Sprint
    └── Archived Reports
```

## Implementation Strategy

### Phase 1: Setup and Preparation
1. ✓ Create `_reorganization/` temp directory
2. Enable myst_parser in conf.py for MD support
3. Convert critical MD files to RST (user guides)
4. Create new section index.rst files

### Phase 2: Content Migration
1. Move files to new structure
2. Update internal cross-references
3. Consolidate historical development logs
4. Remove daily summaries (keep sprint completions)

### Phase 3: Sphinx Configuration
1. Update index.rst with new structure
2. Update conf.py exclude_patterns
3. Test build: `make html`
4. Fix broken links

### Phase 4: Cleanup
1. Remove old directories
2. Remove duplicate files
3. Verify _build/ is gitignored
4. Final build validation

### Phase 5: Documentation
1. Create migration summary
2. Update CLAUDE.md references
3. Commit changes

## File Actions

### MOVE (No Changes)
- `installation.rst` → `getting_started/installation.rst`
- `tutorials/quickstart.rst` → `getting_started/quickstart.rst`
- `tutorials/large_datasets.rst` → `guides/large_datasets.rst`
- `autodoc/*` → `api/*`
- `api_large_datasets.rst` → `api/large_datasets_api.rst`
- `performance_benchmarks.rst` → `developer/performance_benchmarks.rst`
- `images/` → `images/` (unchanged)
- `_static/` → `_static/` (unchanged)

### CONVERT (MD → RST) - User Guides
- `user_guides/migration_guide.md` → `guides/migration_scipy.rst`
- `user_guides/advanced_features.md` → `guides/advanced_features.rst`
- `user_guides/performance_optimization.md` → `guides/performance_guide.rst`
- `user_guides/troubleshooting.md` → `guides/troubleshooting.rst`

### MOVE (Keep MD) - Developer Docs
- `optimization_case_study.md` → `developer/optimization_case_study.md`
- `performance_tuning_guide.md` → `developer/performance_tuning.md`
- `CI_CD_IMPROVEMENTS_SUMMARY.md` → `developer/ci_cd/improvements_summary.md`
- `codeql_workflow_fix.md` → `developer/ci_cd/codeql_workflow_fix.md`
- `github_actions_schema_guide.md` → `developer/ci_cd/github_actions_guide.md`
- `QUICK_REFERENCE_CODEQL.md` → `developer/ci_cd/codeql_quick_reference.md`
- `PYPI_SETUP.md` → `developer/pypi_setup.md`

### CONSOLIDATE - Historical Logs
Keep only sprint completion summaries:
- `development/phase1/sprint1_completion_summary.md` → `history/v0.1.1_sprint/sprint1_completion.md`
- `development/phase2/sprint2_completion_summary.md` → `history/v0.1.1_sprint/sprint2_completion.md`
- `development/phase3/sprint3_completion_summary.md` → `history/v0.1.1_sprint/sprint3_completion.md`
- `development/phase4/sprint4_plan.md` → `history/v0.1.1_sprint/sprint4_plan.md`
- `development/planning/feature_sprint_roadmap.md` → `history/v0.1.1_sprint/roadmap.md`
- `development/README.md` → `history/v0.1.1_sprint/README.md`

Move archives:
- `archive/*` → `history/archived_reports/*`

### DELETE - Obsolete Content
Daily summaries (superseded by sprint completions):
- `development/phase1/DAY1_*.md` (4 files)
- `development/phase1/DAY2_*.md` (2 files)
- `development/phase1/DAY3_*.md` (4 files)
- `development/phase1/DAY5_COMPLETION_SUMMARY.md`
- `development/phase1/DAYS_*.md` (2 combined summaries)
- `development/phase1/sprint1_day1_summary.md`
Total: 13 daily logs removed from phase1

Duplicate/superseded files:
- `migration_guide.rst` (superseded by newer MD version)
- `large_dataset_guide.rst` (merged with tutorials/large_datasets.rst)
- `advanced_features.rst` (superseded by user_guides/ MD version)
- `user_guides/quick_start.md` (misnamed developer guide, content not user-facing)
- `tutorials.rst` (TOC file, will be recreated)
- `main.rst` (will be merged into index.rst)

Total deletions: ~16 files, ~3000 lines

## Conversion Process

### MD → RST Conversion
Using pandoc for initial conversion, then manual cleanup:

```bash
pandoc -f markdown -t rst input.md -o output.rst
```

Manual adjustments needed:
- Fix code block syntax (``` → `.. code-block::`)
- Fix admonitions (> Note → `.. note::`)
- Fix internal links
- Validate RST syntax

### Cross-Reference Updates
Search and replace patterns:
- `:doc:` links to moved files
- `.. include::` directives
- `.. toctree::` entries
- Relative paths in code examples

## Expected Outcomes

### Benefits
1. **Clear Information Architecture**: Users can find what they need quickly
2. **Visible Documentation**: All 5 user guide MD files now accessible
3. **No Duplicates**: Single source of truth for each topic
4. **Maintainability**: Organized structure easier to update
5. **Professional Appearance**: Follows Sphinx best practices
6. **Historical Preservation**: Sprint history preserved but not cluttering main docs

### Metrics
- **Files reduced**: 61 → ~40 (-35%)
- **Directories**: 8 → 6 (cleaner structure)
- **Duplicate content eliminated**: 3 files
- **User guide accessibility**: 0% → 100% (5 files now in build)
- **Historical log consolidation**: 28 files → 6 summaries

### Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Broken links | Search all .rst files for old paths, update systematically |
| Build failures | Test build after each phase |
| Content loss | Keep backups in `_reorganization/backup/` |
| MD conversion issues | Manual review of all converted files |

## Testing Checklist

- [ ] `make clean`
- [ ] `make html`
- [ ] No Sphinx warnings
- [ ] All pages accessible from index
- [ ] No broken internal links
- [ ] No broken external links
- [ ] API docs generate correctly
- [ ] Images load correctly
- [ ] Code examples render properly
- [ ] Search functionality works

## Timeline

**Total Estimated Time**: 4-6 hours

1. **Setup** (30 min): Create structure, update conf.py
2. **Convert MD to RST** (90 min): 4 user guide files
3. **Move files** (30 min): Systematic file moves
4. **Update index.rst** (45 min): New navigation structure
5. **Fix cross-references** (60 min): Search and replace
6. **Test builds** (30 min): Iterative fixing
7. **Cleanup** (30 min): Remove old files
8. **Documentation** (30 min): Write migration summary

## Success Criteria

✓ Sphinx builds without warnings
✓ All user guides visible in documentation
✓ No duplicate content
✓ Clear separation of user vs developer docs
✓ Historical content preserved but archived
✓ Navigation is intuitive and follows best practices
✓ All internal links work
✓ Build size similar or smaller than before

---

**Next Step**: Begin Phase 1 implementation
