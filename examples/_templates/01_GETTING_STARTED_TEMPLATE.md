# Getting Started Category Template

**Audience:** Complete beginners, first-time curve fitting users
**Tone:** Friendly, patient, encouraging, non-technical
**Emphasis:** Hand-holding, quick wins, building confidence

---

# 📘 {Notebook Title}

> {One-sentence description in simple, non-technical language}

⏱️ **{5-15} minutes** | 📊 **Level: ●○○ Beginner** | 🎓 **No prior experience needed**

---

## 🎯 What You'll Learn

By the end of this tutorial, you will:
- ✓ {Specific skill 1 in beginner-friendly language}
- ✓ {Specific skill 2 in beginner-friendly language}
- ✓ {Specific skill 3 in beginner-friendly language}

---

## 💡 Why This Matters

{2-3 sentences explaining why curve fitting is useful in simple terms}

**Perfect for:**
- 🔬 {Beginner-friendly use case 1}
- 📊 {Beginner-friendly use case 2}
- 🚀 {Beginner-friendly use case 3}

**This is the right place if:**
- This is your first time fitting curves to data
- You've heard of curve fitting but never tried it
- You want to see results quickly without diving deep

---

## ⚡ Quick Start (30 seconds)

Let's fit your first curve! Run this code:

```python
{Simple, well-commented example with clear variable names}
```

**Expected output:**
```
{Show expected result with explanations}
```

✓ **Success!** If you see parameters close to {expected values}, you've fit your first curve!

💡 **Tip:** {Helpful tip about what just happened}

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

## 📚 Before You Begin

**First time here?** Perfect! This is exactly where you should start.

**What you need to know:**
- [ ] Python basics (variables, functions, lists) - [Quick Python intro](link)
- [ ] How to run Jupyter notebooks - [Jupyter tutorial](link)

**What you don't need to know:**
- ❌ Advanced mathematics
- ❌ GPU programming
- ❌ JAX or NumPy internals

**Software setup:**
- Python 3.12+ installed
- NLSQ package installed: `pip install nlsq` ([Full guide](../../../README.md#installation))

---

## 📖 Step-by-Step Tutorial

### Step 1: {Action}

{Detailed explanation with inline tips}

💡 **What's happening here:** {Conceptual explanation}

```python
{Code with extensive comments}
```

**Expected output:**
```
{Output with explanations}
```

✓ **Checkpoint:** Make sure you see {specific output} before continuing

### Step 2: {Action}

{Continue with hand-holding approach...}

⚠️ **Common mistake:** {Highlight potential pitfall and how to avoid it}

---

## 🎓 What You've Learned

Congratulations! You've just:
1. ✓ {Achievement 1}
2. ✓ {Achievement 2}
3. ✓ {Achievement 3}

---

## 🗺️ What's Next?

**Ready for more?**
→ [Next Tutorial: {Title}](link) - Build on what you learned ({X} min)

**Want to try it yourself?**

**Exercise 1:** {Simple modification to the example}
<details>
<summary>💡 Hint</summary>
{Helpful hint}
</details>

<details>
<summary>✅ Solution</summary>

```python
{Solution code}
```
</details>

**Not quite ready?**
That's okay! Try:
- Re-running the examples above
- Changing the parameter values to see what happens
- [Reading the FAQ](link) for common questions

---

## ❓ Common First-Time Questions

**Q: What if my fitted parameters are different from the true values?**
A: Small differences are normal! Curve fitting finds the best fit to noisy data. If your parameters are within 10-20% of the true values, that's great for a first try.

**Q: I got an error saying "module not found". What do I do?**
A: Make sure you've installed NLSQ: `pip install nlsq`. Then restart your Jupyter notebook.

**Q: Do I need a GPU to use NLSQ?**
A: No! NLSQ works perfectly on CPU for getting started. GPU acceleration helps when you have very large datasets (millions of points).

**Q: What's the difference between NLSQ and Excel's trendline?**
A: NLSQ is much more powerful! It can fit custom equations, handle millions of data points, and gives you uncertainty estimates. But the basic idea is the same.

---

## 🔗 Next Steps

**Continue your learning journey:**
1. [Interactive Tutorial](nlsq_interactive_tutorial.ipynb) - Hands-on practice (30 min)

**Got a specific need?**
- Large dataset? → [Large Dataset Guide](../02_core_tutorials/large_dataset_demo.ipynb)
- Specific field? → [Gallery](../04_gallery/) - Choose your domain

**Resources:**
- [Complete documentation](https://nlsq.readthedocs.io/)
- [Video tutorial](link) - Visual walkthrough
- [Community forum](https://github.com/imewei/NLSQ/discussions) - Ask questions

---

## 📚 Glossary for Beginners

**Curve fitting:** The process of finding a mathematical function that best describes your data
**Parameters:** The numbers in your equation that we're trying to find
**Residuals:** The difference between your data and the fitted curve
**Covariance:** A measure of how uncertain we are about the fitted parameters

[Concepts and explanations](../../../docs/explanation/index.rst)
