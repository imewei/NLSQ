# Workspace notes

- NLSQ is a Python/JAX nonlinear least-squares library with CLI and Qt interfaces.
- Repository behavior and conventions are documented in AGENTS.md,
  CONTRIBUTING.md, API docstrings, docs/api, and docs/superpowers designs.
- The user requests repeated parallel review, reproduction, correction, and
  verification until review passes report zero new confirmed findings.
- Local fixes are authorized without intermediate approval. Keep public API
  compatibility and reproduce bugs before changing production code.
- Keep review Standards and Spec axes separate. No originating feature ticket
  was supplied; assess documented API behavior and relevant existing designs.
- Tests use two CPU workers and offscreen Qt. Use focused deterministic tests
  during fixes; freeze source before broad verification. GPU smoke checks may
  use the installed backend without changing toolchain packages.
- repomix-output.xml stays outside commits; it is ignored by the current tree.
- This request authorizes review and fixes, not publication or a new commit.
