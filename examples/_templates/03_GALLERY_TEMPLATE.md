# Gallery (Domain Examples) Category Template

**Audience:** Domain experts (biology, chemistry, physics, engineering)
**Tone:** Domain-specific, practical, applications-focused
**Emphasis:** Real-world relevance, parameter interpretation, literature context

---

# 📘 {Application Name}: {Specific Example}

> {One-sentence description in domain-specific terminology}

⏱️ **{15-25} minutes** | 📊 **Level: ●●○ Intermediate** | 🏷️ **{Domain}** | 🔬 **{Subfield}**

---

## 🔬 Domain Background

**Physical/Chemical/Biological System:**
{Description of the system being modeled}

**Model Equation:**
$$
{Mathematical model in LaTeX}
$$

**Where:**
- ${parameter1}$: {Physical meaning and units}
- ${parameter2}$: {Physical meaning and units}
- ${parameter3}$: {Physical meaning and units}

**Common applications in {domain}:**
- {Application 1 with specific examples}
- {Application 2 with specific examples}
- {Application 3 with specific examples}

---

## 🎯 What You'll Learn

After this tutorial, you will be able to:
- ✓ Fit {domain-specific model} to experimental data
- ✓ Interpret fitted parameters in {domain} context
- ✓ Assess goodness of fit using {domain-standard metrics}
- ✓ Extract physically meaningful insights from fitted parameters

---

## 💡 Why This Model Matters

**In {domain} research:**
{2-3 sentences explaining importance in the field}

**Typical experimental setups:**
- {Setup 1}: {Description}
- {Setup 2}: {Description}

**What you can learn from fitted parameters:**
- ${parameter1}$ tells you about {physical property}
- ${parameter2}$ indicates {behavior/characteristic}
- Ratio ${param1/param2}$ is related to {derived quantity}

---

## 📊 Parameter Interpretation Guide

| Parameter | Physical Meaning | Typical Range | What Affects It |
|-----------|-----------------|---------------|-----------------|
| ${p1}$ | {meaning} | {range} {units} | {factors} |
| ${p2}$ | {meaning} | {range} {units} | {factors} |
| ${p3}$ | {meaning} | {range} {units} | {factors} |

**Derived quantities:**
- **{Quantity 1}** = ${formula}$: {Physical significance}
- **{Quantity 2}** = ${formula}$: {Physical significance}

---

## ⚡ Quick Example

```python
{Complete working example with domain-realistic data}
```

**Expected fitted parameters:**
{Parameter interpretation in domain context}

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

## 📖 Detailed Analysis

### 1. Experimental Data

{Description of the experimental data}

**Data characteristics:**
- Sample size: {N} measurements
- Measurement uncertainty: ±{σ} {units}
- Experimental conditions: {conditions}

### 2. Model Fitting

{Step-by-step fitting with domain-specific explanations}

```python
{Code with domain-relevant comments}
```

### 3. Results Interpretation

**Fitted parameters:**
- ${p1}$ = {value} ± {uncertainty} {units}
  - **Interpretation:** {What this means for the system}
  - **Literature comparison:** {Typical values cite source}

- ${p2}$ = {value} ± {uncertainty} {units}
  - **Interpretation:** {What this means}
  - **Expected range:** {Range from literature}

### 4. Physical Insights

**Derived physical quantities:**
{Calculate and interpret derived quantities}

**System characterization:**
{Classification based on fitted parameters}

### 5. Quality Assessment

**Goodness of fit:**
- χ²/dof = {value} (expect ≈ 1.0)
- R² = {value}
- {Domain-specific metric} = {value}

**Residual analysis:**
{Interpretation of residuals in domain context}

---

## 🔑 Key Insights

1. **{Finding 1}:** {Physical interpretation}
2. **{Finding 2}:** {Comparison to theory/literature}
3. **{Finding 3}:** {Practical implications}

---

## ⚠️ Domain-Specific Considerations

**Experimental artifacts to watch for:**
- **{Artifact 1}:** {How it affects fitting, how to detect}
- **{Artifact 2}:** {How it affects fitting, how to detect}

**Physical constraints:**
- {Parameter} must be {constraint} because {physical reason}
- Use bounds: `bounds=([lower], [upper])` to enforce physical constraints

**Alternative models:**
When {condition}, consider:
- [{Alternative model 1}](link): For {scenario}
- [{Alternative model 2}](link): For {scenario}

---

## 📚 Related Techniques in {Domain}

**Similar analyses:**
- [{Related example 1}](link) - {Relationship}
- [{Related example 2}](link) - {Relationship}

**Extensions:**
- Multi-component fitting for {complex scenario}
- Time-resolved analysis for {dynamic systems}
- Spatial fitting for {imaging data}

---

## 📖 References

**Literature:**
1. {Author et al.} ({Year}). {Title}. {Journal}. [DOI](link)
2. {Author et al.} ({Year}). {Title}. {Journal}. [DOI](link)

**Theoretical background:**
- [Textbook chapter](link) on {topic}
- [Review article](link) on {topic}

**NLSQ resources:**
- [API Documentation](https://nlsq.readthedocs.io/)
- [Performance guide](../../docs/performance_guide.md) for large datasets

---

## ❓ Domain-Specific Questions

**Q: My fitted {parameter} is outside the expected range. What does this mean?**
A: {Possible physical explanations and troubleshooting steps}

**Q: How do I account for {experimental complication}?**
A: {Domain-specific guidance}

**Q: Can I use this model for {related system}?**
A: {Applicability discussion with caveats}

---

## 📚 Glossary

**{Domain term 1}:** {Definition in domain context}
**{Domain term 2}:** {Definition in domain context}
**{Technique abbreviation}:** {Full name and explanation}

[Complete glossary](../../docs/glossary.md)
