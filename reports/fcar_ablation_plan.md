# FCAR Ablation Plan

Date: 2026-07-09 KST

Run ablations only after the first FCAR implementation plan is approved. Do not run them in this planning pass.

## Required Ablations

1. No frozen/base expert

- force gate to use LoRA or mixed LoRA-heavy predictions;
- kill role: proves whether base retention is necessary.

2. Instruction-only router

- use instruction/task embedding only;
- kill role: should match MoIRA/task-oracle scale. If it matches FCAR, FCAR novelty is weak.

3. Frame-state-only router

- use current 8D state, phase, action norm/disagreement, but no instruction embedding;
- kill role: tests whether frame signal is the real source of gain.

4. No retention loss

- train only action L2 mixture objective;
- kill role: tests whether retention objective matters.

5. Static mixture / adapter soup

- use fixed alpha grid or static adapter merge;
- kill role: if static merge matches FCAR, learned frame routing is unnecessary.

6. Disagreement-only heuristic

- simple threshold on `||a_base - a_lora||`;
- kill role: if heuristic matches FCAR, method can be simplified or novelty weakens.

7. Rank-8 LoRA baseline

- optional if cheap;
- kill role: if rank-8 LoRA alone fixes negative transfer, FCAR is unnecessary.

## Ablation Metrics

Use the same metrics as the main run:

- action L2;
- translation/rotation/gripper breakdown;
- route fraction;
- oracle recovery;
- help/hurt count;
- train/eval gap;
- action range validity.
