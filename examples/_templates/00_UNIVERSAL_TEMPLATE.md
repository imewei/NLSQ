# Universal NLSQ Notebook Introduction Template

This template should be adapted for each notebook category. See category-specific variants below.

---

# 📘 {Notebook Title}

> {One-sentence description of what this notebook teaches}

⏱️ **{15-30} minutes** | 📊 **Level:** {●○○ Beginner / ●●○ Intermediate / ●●● Advanced}

---

## 🎯 What You'll Learn

By the end of this notebook, you will be able to:

- ✓ {Specific, measurable skill 1}
- ✓ {Specific, measurable skill 2}
- ✓ {Specific, measurable skill 3}

---

## 🗺️ Learning Path

**You are here:** {Category} > {Subcategory} > **{Current Notebook}**

```
{ASCII visualization showing position in learning journey}
Getting Started → Core Tutorials → [You are here] → Advanced
```

**Recommended flow:**
- ← **Previous:** [{Prerequisite notebook}](link)
- → **Next:** [{Logical next step}](link)

**Alternative paths:**
- 🚀 Want to optimize performance? → [{Performance notebook}](link)
- 🔬 Need domain examples? → [{Gallery}](link)

---

## 📚 Before You Begin

**Required knowledge:**
- [ ] {Prerequisite skill/concept 1} ([Learn here](link))
- [ ] {Prerequisite skill/concept 2} ([Learn here](link))

**Required software:**
- NLSQ >= {version} ([Installation guide](../../README.md#installation))
- Python >= 3.12
- {GPU note if applicable}

**First time with NLSQ?** Start here: [NLSQ Quickstart](../01_getting_started/nlsq_quickstart.ipynb)

---

## 💡 Why This Matters

{2-3 sentences explaining real-world motivation and practical importance}

**Common use cases:**
- 🔬 {Domain-specific example 1}
- 📊 {Domain-specific example 2}
- ⚙️ {Domain-specific example 3}

**When to use this approach:**
- ✅ **Good for:** {Scenarios where this applies}
- ❌ **Not ideal for:** {Scenarios} → Try [{Alternative notebook}](link) instead

---

## ⚡ Quick Start (30 seconds)

See NLSQ in action with this minimal example:

```python
{3-5 lines of working code that demonstrates the core concept}
```

**Expected output:**
```
{Show expected result}
```

✓ If you see similar output, you're ready to continue!

---

## 🔧 Setup

**IMPORTANT:** Always include this configuration cell first (before any imports):

```python
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
%matplotlib inline
```

**Then add your imports:**

```python
import numpy as np
import jax.numpy as jnp
from nlsq import curve_fit
# ... other imports
```

---

## 📖 Tutorial Content

{Main content sections follow...}

### Section 1: {Title}

{Content...}

### Section 2: {Title}

{Content...}

---

## 🎓 Key Takeaways

After completing this notebook, remember:

1. **{Key concept 1}:** {Summary in one sentence}
2. **{Key concept 2}:** {Summary in one sentence}
3. **{Key concept 3}:** {Summary in one sentence}

---

## ❓ Common Questions

**Q: {Anticipated question 1}?**
A: {Clear answer with code example if relevant}

**Q: {Anticipated question 2}?**
A: {Clear answer}

**Q: {Anticipated question 3}?**
A: {Clear answer}

💬 [See all FAQs](../../docs/faq.md) | [Ask a question](https://github.com/imewei/NLSQ/discussions)

---

## 🔗 Related Resources

**Next steps:**
- [{Related notebook 1}](link) - {Why it's relevant}
- [{Related notebook 2}](link) - {Why it's relevant}

**Further reading:**
- [API Documentation](https://nlsq.readthedocs.io/)
- [GitHub Repository](https://github.com/imewei/NLSQ)
- {Research paper if applicable}

**Need help?**
- 💬 [Discussions](https://github.com/imewei/NLSQ/discussions)
- 🐛 [Report issues](https://github.com/imewei/NLSQ/issues)

---

## 📚 Glossary

**{Term 1}:** {Definition}
**{Term 2}:** {Definition}
**{Term 3}:** {Definition}

[Complete glossary](../../docs/glossary.md)
