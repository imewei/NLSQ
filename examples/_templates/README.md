# NLSQ Notebook Introduction Templates

This directory contains standardized templates for NLSQ tutorial notebooks.

## Template Files

1. **`00_UNIVERSAL_TEMPLATE.md`** - Base template with all sections
2. **`01_GETTING_STARTED_TEMPLATE.md`** - For beginner tutorials
3. **`02_CORE_TUTORIALS_TEMPLATE.md`** - For intermediate tutorials
4. **`03_GALLERY_TEMPLATE.md`** - For domain-specific examples
5. **`04_ADVANCED_TEMPLATE.md`** - For advanced/expert topics
6. **`05_FEATURE_DEMOS_TEMPLATE.md`** - For feature demonstrations
7. **`06_STREAMING_TEMPLATE.md`** - For production/streaming topics

## Usage Guide

### Step 1: Choose the Right Template

Match your notebook to the appropriate category:

- **Getting Started:** First-time users, basic concepts, hand-holding
- **Core Tutorials:** Intermediate users, specific techniques, best practices
- **Gallery:** Domain-specific applications (biology, chemistry, physics, engineering)
- **Advanced:** Expert users, customization, advanced algorithms
- **Feature Demos:** Specific NLSQ feature deep-dives
- **Streaming:** Production deployment, fault tolerance, monitoring

### Step 2: Customize the Template

Replace all `{placeholders}` with actual content:

- `{Notebook Title}` → Your notebook's title
- `{One-sentence description}` → Brief description
- `{15-30}` → Estimated time in minutes
- `{Level}` → ●○○ Beginner / ●●○ Intermediate / ●●● Advanced
- `{Specific skill X}` → Concrete learning objectives
- etc.

### Step 3: Adapt Sections as Needed

**Keep:** All core sections (What You'll Learn, Learning Path, Why This Matters, etc.)

**Modify:** Adjust depth and tone for your audience

**Remove:** Sections not applicable to your notebook (but keep structure consistent)

**Add:** Category-specific sections as shown in templates

## Template Structure

All templates follow this basic structure:

```markdown
1. Header (Title, Time, Level)
2. Learning Objectives
3. Learning Path / Navigation
4. Prerequisites
5. Motivation (Why This Matters)
6. Quick Start Example
7. Setup (Matplotlib Configuration + Imports) ← NEW
8. Main Content (varies by category)
9. Key Takeaways
10. Common Questions
11. Related Resources
12. Glossary
```

### ⚠️ Critical: Matplotlib Configuration

**ALL notebooks MUST start with this configuration cell:**

```python
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
%matplotlib inline
```

This cell should be **Cell 0** (the very first code cell) in every notebook, before any imports. This ensures proper inline plotting in both VS Code and Jupyter environments.

## Category-Specific Adaptations

### Getting Started
- **Extra sections:** Common First-Time Questions, Interactive Exercises
- **Tone:** Very friendly, patient, encouraging
- **Code:** Heavily commented, step-by-step

### Core Tutorials
- **Extra sections:** Performance Considerations, Best Practices, Common Pitfalls
- **Tone:** Instructional, thorough
- **Code:** Strategically commented, production-ready examples

### Gallery
- **Extra sections:** Domain Background, Parameter Interpretation, References
- **Tone:** Domain-specific terminology
- **Code:** Domain-realistic examples with physical interpretations

### Advanced
- **Extra sections:** API Reference, Theoretical Background, Research Applications
- **Tone:** Technical, concise
- **Code:** Advanced patterns, customization examples

### Feature Demos
- **Extra sections:** Configuration Guide, Performance Impact, Integration Patterns
- **Tone:** Feature-focused, practical
- **Code:** Before/after comparisons, configuration examples

### Streaming
- **Extra sections:** Production Checklist, Monitoring, Fault Scenarios
- **Tone:** Professional, operational
- **Code:** Production-ready with comprehensive error handling

## Icon/Emoji Guide

Use these consistently across all notebooks:

- 📘 Notebook title
- 🎯 Learning objectives
- 🗺️ Learning path/navigation
- 📚 Prerequisites
- 💡 Why this matters / insights / tips
- ⚡ Quick start / fast example
- 📖 Main tutorial content
- 🎓 Key takeaways / what you learned
- ❓ Questions / FAQ
- 🔗 Related resources / links
- ⚠️ Warnings / common pitfalls
- ✓ Success checkpoints / expected outputs
- 🔬 Domain-specific (science)
- 📊 Data/statistics
- ⚙️ Configuration/settings
- 🏭 Production/enterprise
- 🔧 Tools/utilities
- 💻 Code examples
- 📈 Performance/metrics

## Quality Checklist

Before finalizing your notebook, verify:

- [ ] All placeholders replaced with actual content
- [ ] Time estimate is realistic (test it!)
- [ ] Difficulty level matches content complexity
- [ ] Prerequisites are explicitly listed with links
- [ ] Quick Start example works (copy-paste-run)
- [ ] Learning objectives are specific and measurable
- [ ] Navigation links work (prev/next)
- [ ] Code examples are tested and work
- [ ] Expected outputs are shown
- [ ] Glossary terms are defined
- [ ] Related resources are linked
- [ ] No broken links
- [ ] Consistent formatting and style

## Examples

See the pilot implementations:
- Getting Started: `../notebooks/01_getting_started/nlsq_quickstart.ipynb`
- Core Tutorial: `../notebooks/02_core_tutorials/large_dataset_demo.ipynb`
- Gallery: `../notebooks/04_gallery/physics/damped_oscillation.ipynb`

## Questions?

See the [NLSQ Documentation](https://nlsq.readthedocs.io/) or [open an issue](https://github.com/imewei/NLSQ/issues).
