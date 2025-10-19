# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for NLSQ.

## What is an ADR?

An Architecture Decision Record (ADR) captures an important architectural decision made along with its context and consequences. ADRs help future maintainers understand why certain decisions were made.

## ADR Template

Each ADR follows this structure:

```markdown
# ADR-XXX: Title

**Status**: Accepted | Proposed | Deprecated | Superseded

**Date**: YYYY-MM-DD

**Deciders**: Names or roles

## Context

What is the issue we're facing? What factors are we considering?

## Decision

What decision did we make?

## Consequences

### Positive
- What benefits does this bring?

### Negative
- What are the tradeoffs or downsides?

## References
- Links to related resources
```

## Index of ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [003](003-streaming-over-subsampling.md) | Streaming Optimization Over Subsampling | Accepted | 2025-10-17 |
| [004](004-parameter-unpacking-simplification.md) | Parameter Unpacking Simplification | Accepted | 2025-10-18 |
| [005](005-jax-autodiff-gradients.md) | JAX Autodiff for Gradient Computation | Accepted | 2025-10-18 |

## When to Create an ADR

Create an ADR when making decisions about:

- **Technology choices**: Which libraries, frameworks, or platforms to use
- **Architecture patterns**: How to structure the code or system
- **API design**: How users interact with the library
- **Performance trade-offs**: When optimizing for speed vs. memory vs. accuracy
- **Breaking changes**: When introducing incompatible changes

## References

- [ADR GitHub Organization](https://adr.github.io/)
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
