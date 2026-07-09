# FCAR Next Decision

Date: 2026-07-09 KST

Final decision: `READY_TO_IMPLEMENT_FCAR_TINY_GATE`

Reason:

- the problem statement is fixed;
- baselines and anchors are fixed;
- metrics and split policy are fixed;
- MoIRA/AAC comparison is clear;
- kill criteria are fixed;
- implementation TODO is specific;
- the method remains only a tiny gate on official SmolVLA-LIBERO predictions, with no VLA retraining.

Known caveat:

- saved per-frame base/LoRA prediction artifacts are missing, so the first implementation step must regenerate and save them from official assets before training the tiny gate.

Exact next prompt:

Implement the FCAR tiny-gate experiment exactly as specified in `reports/fcar_implementation_todo.md`. Do not change baselines, metrics, split policy, or kill criteria after seeing results.
