# Repeated whole-codebase review and debugging

## Trigger and scope

Run when the user requests another whole-codebase review/debugging loop. Review
all production subsystems and their contracts. On this run, pin review HEAD to
6a6488850d79106076594826bd5ddc7e7b9be328 and the previous audit commit to
11b3259abc1c95b564625bf1b1f1fef9c71911cc. Use the three-dot diff for the
changed-code review and whole-tree inspection for additional runtime defects.
Preserve unrelated work and never publish, push, or commit without instruction.

## Loop

1. Inventory the current tree and assign non-overlapping review domains in
   parallel. Keep Standards and Spec findings isolated; use existing API/design
   contracts as the Spec source when no originating feature specification exists.
2. Record scope coverage and candidate findings. Count a correctness finding only
   after an automated test or harness reproduces its specific wrong behavior.
   Label architectural smells as judgement calls and cite the actual contract.
3. Minimize each failure, then show ranked falsifiable hypotheses. Probe one
   variable at a time. Redact credentials before showing evidence.
4. Write a regression at the actual call boundary, observe red, apply the smallest
   root-cause fix, and observe green. Run nearby tests. Remove temporary probes.
5. Synthesize findings with statuses: confirmed, fixed, refuted, or advisory with
   an explicit disposition. Do not restore obsolete behavior merely to satisfy
   an outdated test. Do not relax numerical accuracy assertions to hide failures.
6. Repeat with changed emphasis or reviewer/domain rotation. Any new confirmed
   finding resets the consecutive-clean-pass count to zero. Re-review every fix.
7. After two complete consecutive review passes find no new actionable issues,
   freeze production and test source. Run the full repository suite with two
   workers, no automatic reruns, offscreen Qt, and incremental failure artifacts.
   If interrupted, reconcile exact node IDs so no current case is omitted.
8. Resolve any verification failures, rerun their original scenarios and affected
   suites, reset the clean count, and resume review. Refresh the AST graph after
   code changes. Run lint, formatting, configured type/security gates, and docs
   checks appropriate to the final change. Record optional skips and warnings.

## Checkpoint and completion brief

There is no approval checkpoint for reversible local review, regression tests,
or fixes already authorized by the user. Ask only for a genuinely missing
decision or a blocker that cannot be resolved from local evidence. Repeated
failed fixes must trigger an architecture review under systematic-debugging.

Finish when the two-clean-pass condition holds, no confirmed findings remain
open, and required checks pass or have an explicit non-code environmental
disposition. Present findings separately under Standards and Spec, plus the
runtime-debugging record, verified counts, limitations, and a report link.
Zero findings means none found by these scoped review passes; it is not a proof
that every possible input is bug-free.
