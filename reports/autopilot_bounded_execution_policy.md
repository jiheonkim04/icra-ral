# Autopilot Bounded Execution Policy

This policy supersedes any earlier instruction to run unbounded end-to-end
research loops in a single Codex execution.

## Execution Limit

Each Codex execution may complete at most one major research milestone.

Examples of one major milestone:

- real candidate-generation smoke,
- research-integrity policy update,
- ActionMap vs TCA-Map tiny training/eval,
- LoRA tiny training/eval,
- rollout diagnostic,
- paper-grade roadmap update.

After completing one milestone, Codex must stop, report the result, and name the
next recommended milestone. It must not immediately start another major
milestone in the same execution.

## Diff Size Stop Gates

Before commit or merge, Codex must compute the changed-file count and line diff.

Stop before commit and report if either threshold is exceeded:

- more than 50 files would change,
- more than 5,000 changed lines would be committed.

Large diffs must not be merged without an explicit summary and justification.

## Runtime Stop Gates

Stop and report if a task runs longer than 2 hours without actual training or
rollout progress.

If no loss, metric, rollout result, or concrete validation result is being
produced, Codex must not keep expanding planners indefinitely. It should produce
one bounded plan, report the next executable step, and stop.

## Required Pre-Merge Report

Before every merge, Codex must report:

- files changed count,
- line diff count,
- whether training happened,
- whether rollout happened,
- whether loss was computed,
- whether the work is only planning/scaffolding,
- validation commands and results,
- concise justification for merging.

This report is required even for small diffs.

## Evidence Labels

Planning/scaffolding is not evidence of method performance. Offline proxy,
smoke, diagnostic, rollout, and paper-grade candidate results must remain
separately labeled.
