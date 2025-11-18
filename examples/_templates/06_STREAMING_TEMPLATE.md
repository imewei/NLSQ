# Streaming & Production Category Template

**Audience:** Engineers deploying NLSQ in production
**Tone:** Professional, reliability-focused, operational
**Emphasis:** Fault tolerance, monitoring, production best practices

---

# 📘 {Production Feature}: {Specific Topic}

> {One-sentence description of production capability}

⏱️ **{20-30} minutes** | 📊 **Level: ●●○ Intermediate** | 🏭 **Production-Ready**

---

## 🎯 What You'll Learn

After this tutorial, you will:
- ✓ Implement {production feature} in production systems
- ✓ Configure fault tolerance and recovery strategies
- ✓ Monitor and diagnose issues in production
- ✓ Follow best practices for {specific scenario}

---

## 💡 Production Context

**Why this matters in production:**
{Explanation of production challenges this addresses}

**Common production scenarios:**
- {Scenario 1}: {Challenge and how this helps}
- {Scenario 2}: {Challenge and how this helps}
- {Scenario 3}: {Challenge and how this helps}

**SLA impact:**
- Availability: {Impact description}
- Reliability: {Impact description}
- Performance: {Impact description}

---

## 🗺️ Prerequisites

**Technical requirements:**
- [ ] Production NLSQ deployment
- [ ] Understanding of [Core Tutorials](../02_core_tutorials/)
- [ ] {Additional infrastructure requirement}

**Operational readiness:**
- [ ] Monitoring infrastructure in place
- [ ] Error handling strategy defined
- [ ] Backup/recovery procedures established

---

## ⚡ Production Example

```python
{Production-grade code example with error handling}
```

**Key production features:**
- ✓ Error handling and recovery
- ✓ Logging and monitoring hooks
- ✓ Resource cleanup
- ✓ Graceful degradation

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

## 📖 Implementation Guide

### 1. Basic Production Setup

{Step-by-step production implementation}

```python
{Production-ready code with comprehensive error handling}
```

**Production considerations:**
- Resource limits and quotas
- Timeout configurations
- Retry strategies
- Circuit breakers

### 2. Fault Tolerance Configuration

**Failure scenarios to handle:**
1. **{Failure type 1}:** {How to detect and recover}
2. **{Failure type 2}:** {How to detect and recover}
3. **{Failure type 3}:** {How to detect and recover}

**Retry configuration:**
```python
{
    'max_retries': 3,
    'retry_delay': 1.0,  # seconds
    'backoff_factor': 2.0,  # Exponential backoff
    'retry_on': [TimeoutError, MemoryError],
}
```

### 3. Checkpoint and Resume

**Why checkpointing:**
{Explanation of long-running job recovery}

**Implementation:**
```python
{Code showing checkpoint/resume pattern}
```

**Best practices:**
- Checkpoint frequency: {Guidance}
- Storage considerations: {Where to store}
- Cleanup strategy: {When to delete}

### 4. Monitoring and Observability

**Key metrics to track:**
- **Performance metrics:**
  - Fit completion time
  - Throughput (fits/second)
  - Memory usage
  - GPU utilization (if applicable)

- **Reliability metrics:**
  - Success rate
  - Retry count
  - Checkpoint frequency
  - Recovery time

**Monitoring integration:**
```python
{Code showing metrics collection}
```

**Alerting thresholds:**
- Error rate > {threshold}%
- Performance degradation > {threshold}%
- Resource usage > {threshold}%

### 5. Production Diagnostics

**Diagnostic information to collect:**
```python
{Code showing diagnostic data collection}
```

**Log structure:**
```python
{
    'timestamp': '2025-01-18T10:30:00Z',
    'job_id': 'fit-12345',
    'status': 'completed|failed|retrying',
    'metrics': {...},
    'diagnostics': {...}
}
```

---

## 🏭 Production Checklist

Before deploying to production, verify:

**Configuration:**
- [ ] Resource limits configured appropriately
- [ ] Timeout values tested under load
- [ ] Retry strategy validated
- [ ] Checkpoint storage configured

**Monitoring:**
- [ ] Metrics collection enabled
- [ ] Alerts configured
- [ ] Dashboards created
- [ ] Log aggregation working

**Reliability:**
- [ ] Error handling tested
- [ ] Recovery procedures documented
- [ ] Fallback strategies implemented
- [ ] Graceful shutdown implemented

**Performance:**
- [ ] Load tested at expected scale
- [ ] Resource usage within limits
- [ ] Performance SLAs validated

**Operations:**
- [ ] Runbooks created
- [ ] On-call procedures defined
- [ ] Escalation paths documented

---

## ⚠️ Production Issues and Solutions

**Issue 1: Out of Memory in Production**
- **Symptoms:** {How it manifests}
- **Root cause:** {Explanation}
- **Immediate mitigation:** {Quick fix}
- **Long-term solution:** {Proper fix}
- **Prevention:** {How to avoid}

**Issue 2: Timeout Errors Under Load**
- **Diagnosis:** {How to identify}
- **Tune:** {Configuration adjustments}
- **Monitor:** {What to track}

**Issue 3: Checkpoint Recovery Failures**
- **Causes:** {Common reasons}
- **Recovery procedure:** {Step-by-step}
- **Validation:** {How to verify}

---

## 📊 Production Metrics

**Key Performance Indicators:**

| Metric | Target | Alert Threshold | Action |
|--------|--------|-----------------|--------|
| Success Rate | >99% | <95% | {Action} |
| P95 Latency | <{X}s | >{Y}s | {Action} |
| Memory Usage | <{X}GB | >{Y}GB | {Action} |

**Capacity planning:**
- Expected load: {Requests/day}
- Peak capacity: {Max concurrent}
- Resource requirements: {CPU/Memory/GPU}

---

## 🔧 Advanced Production Patterns

**Pattern 1: Circuit Breaker**
```python
{Circuit breaker implementation}
```

**Pattern 2: Bulkhead Isolation**
```python
{Resource isolation pattern}
```

**Pattern 3: Graceful Degradation**
```python
{Fallback strategy implementation}
```

---

## 🔑 Key Takeaways

1. **Always implement:** {Critical production feature}
2. **Monitor these metrics:** {Key metrics list}
3. **Have runbooks for:** {Common failure scenarios}
4. **Test under load:** {Load testing guidance}

---

## 🔗 Related Resources

**Production topics:**
- [{Related production feature}](link)
- [{Monitoring guide}](link)
- [{Troubleshooting guide}](../03_advanced/troubleshooting_guide.ipynb)

**Documentation:**
- [Production deployment guide](../../docs/production.md)
- [Monitoring integration](../../docs/monitoring.md)
- [API Reference](https://nlsq.readthedocs.io/)

---

## ❓ Production FAQs

**Q: How do I scale NLSQ horizontally?**
A: {Scaling guidance}

**Q: What's the recommended checkpoint frequency?**
A: {Recommendation with reasoning}

**Q: How do I handle version upgrades in production?**
A: {Upgrade strategy}

---

## 📚 Production Glossary

**Circuit Breaker:** {Definition in production context}
**Checkpoint:** {Definition and purpose}
**Graceful Degradation:** {Definition and implementation}

[Complete glossary](../../docs/glossary.md)
