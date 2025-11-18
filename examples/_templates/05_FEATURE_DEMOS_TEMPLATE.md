# Feature Demonstrations Category Template

**Audience:** Users wanting to explore specific NLSQ features
**Tone:** Tutorial-style, feature-focused, practical
**Emphasis:** Feature capabilities, configuration, use cases

---

# 📘 {Feature Name}: {Subtitle}

> {One-sentence description of what this feature does}

⏱️ **{15-25} minutes** | 📊 **Level: ●●○ Intermediate** | 🎯 **Feature Focus**

---

## 🎯 What You'll Learn

After this demonstration, you will understand:
- ✓ What {feature} does and when to use it
- ✓ How to enable and configure {feature}
- ✓ Practical applications of {feature}
- ✓ {Feature} vs not using {feature}

---

## 💡 Feature Overview

**What is {feature}?**
{Clear explanation of the feature in 2-3 sentences}

**Why was it added?**
{Problem it solves or improvement it provides}

**When to use it:**
- ✅ **Enable when:** {Scenario 1}
- ✅ **Enable when:** {Scenario 2}
- ❌ **Not needed when:** {Scenario 3}

---

## 🗺️ Learning Path

**You are here:** Feature Demonstrations > **{Feature}**

**Prerequisites:**
- Basic NLSQ usage: [Quickstart](../01_getting_started/nlsq_quickstart.ipynb)
- {Additional prerequisite if needed}

**Related features:**
- [{Related feature 1}](link)
- [{Related feature 2}](link)

---

## ⚡ Quick Comparison

**Without {feature}:**
```python
{Code example without feature}
```
**Result:** {Outcome}

**With {feature}:**
```python
{Code example with feature enabled}
```
**Result:** {Improved outcome}

**Improvement:** {Quantified benefit}

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

## 📖 Feature Demonstration

### Demo 1: Basic Usage

**Scenario:** {Simple use case}

```python
{Complete working example}
```

**What's happening:**
1. {Step 1 explanation}
2. {Step 2 explanation}
3. {Benefit of using feature}

### Demo 2: Configuration Options

**Available settings:**

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `option1` | {default} | {what it controls} | {scenario} |
| `option2` | {default} | {what it controls} | {scenario} |

**Configuration example:**
```python
{Code showing different configurations}
```

### Demo 3: Advanced Application

**Complex scenario:** {Description}

```python
{Advanced usage example}
```

**Benefits in this scenario:**
- {Benefit 1 with metrics}
- {Benefit 2 with metrics}

### Demo 4: Integration Patterns

**Combining with other features:**
```python
{Example showing feature integration}
```

---

## 🔧 Configuration Guide

**Complete configuration options:**

```python
{
    'feature_param1': {value},  # {Description}
    'feature_param2': {value},  # {Description}
    # ... more options
}
```

**Recommended settings:**
- **For {use case 1}:** `{config}`
- **For {use case 2}:** `{config}`
- **For {use case 3}:** `{config}`

**Environment variables:**
{If applicable, show environment-based configuration}

---

## 📊 Performance Impact

**Benchmarks:**

| Metric | Without Feature | With Feature | Improvement |
|--------|----------------|--------------|-------------|
| {Metric 1} | {value} | {value} | {%} |
| {Metric 2} | {value} | {value} | {%} |

**Trade-offs:**
- **Pros:** {Benefits}
- **Cons:** {Costs or limitations}

---

## 🔑 Key Takeaways

1. **{Feature} is best for:** {Primary use case}
2. **Enable by:** {How to enable}
3. **Key benefit:** {Main advantage}
4. **Watch out for:** {Important consideration}

---

## ⚠️ Common Issues

**Issue 1: {Feature} not working as expected**
- **Check:** {Diagnostic step}
- **Fix:** {Solution}

**Issue 2: {Performance/behavior problem}**
- **Cause:** {Explanation}
- **Solution:** {Fix}

---

## 💡 Best Practices

1. **{Practice 1}:** {Explanation}
2. **{Practice 2}:** {Explanation}
3. **{Practice 3}:** {Explanation}

**Anti-patterns to avoid:**
- ❌ {What not to do}: {Why}
- ❌ {What not to do}: {Why}

---

## 🔗 Related Resources

**Other features:**
- [{Complementary feature}](link) - Works well with this
- [{Alternative feature}](link) - Different approach

**Documentation:**
- [API Reference](https://nlsq.readthedocs.io/en/latest/api.html#{feature})
- [Configuration guide](link)

---

## ❓ Feature FAQs

**Q: Does {feature} work with {other feature/scenario}?**
A: {Answer with examples}

**Q: What's the overhead of enabling {feature}?**
A: {Performance impact explanation}

**Q: Can I use this in production?**
A: {Stability/maturity assessment}

---

## 📚 Glossary

**{Feature-specific term 1}:** {Definition}
**{Feature-specific term 2}:** {Definition}

[Complete glossary](../../docs/glossary.md)
