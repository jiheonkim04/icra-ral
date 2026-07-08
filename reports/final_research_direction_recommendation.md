# Final Research Direction Recommendation

Date: 2026-07-08

## Final Decision

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

## Recommended Direction

Target-Grounded ActionMap, also called Language-Grounded Action Heatmap for VLA Manipulation, is the only direction worth preserving from the previous work.

The direction is not approved for method implementation yet. It should proceed only after an ActionMap-style anchor reproduction clears the simple-baseline gate.

## Why This Direction Survives

The old Target-Prior TCA route had one rare reusable positive signal: when the target prior was correct and non-leaking, fixed-prior target conditioning improved offline action-decoder proxies and wrong-target behavior. That signal is method-relevant, not merely a benchmark observation.

The old route failed because the action head was weak and the online 7D diagnostic did not beat mean action. Therefore the salvage must change the method contribution:

- discard TCA-Select;
- discard weak 7D MLP TCA heads;
- keep non-leaking target/object prior construction;
- replace the weak direct 7D head with an ActionMap-style voxel action heatmap decoder;
- condition the heatmap with target/object semantic priors through FiLM, AdaLN, or residual gating;
- evaluate counterfactual target sensitivity and object lexical/paraphrase robustness.

## Why This Is Not A Direct GO

The local ActionMap mini-anchor failed the first reproduction gate:

- mean-action action L2: `0.466767673`;
- simple MLP action L2: `0.501926707`;
- ActionMap-style action L2: `0.529931357`;
- candidate diversity collapse: unique trans/rot/grip `5 / 1 / 2`.

That does not kill ActionMap as a literature anchor, but it does kill direct extension from the current local mini-head. A target-conditioned heatmap method is not RA-L-stable unless ActionMap alone first beats mean action, linear/L1, and cheap MLP in a credible official-style setup.

## Novelty Positioning

The method-level novelty would be:

1. ActionMap-style action heatmap decoding.
2. Explicit semantic target/object prior conditioning of the heatmap.
3. Counterfactual target consistency.
4. Paraphrase/object lexical robustness objective and evaluation.
5. LoRA/adapter use only as a low-resource training tool.

The novelty is not LoRA, not TCA-Select, not a wrapper, not a benchmark, not canonicalization, and not a single grounded 3D point.

## Latest-Paper Gap

The latest anchor scan raises the bar:

- ActionMap solves action-space geometry but not explicit target/object semantic heatmap grounding.
- Direct 3D point injection solves one-point spatial grounding but not target-conditioned heatmap distributions or paraphrase robustness.
- LIBERO-Para exposes object lexical and paraphrase fragility but does not solve it with decoder design.
- w2 VLA, GuidedVLA, CAC-VLA, and RoVLA already cover adjacent semantic conditioning, object grounding, latent action conditioning, and consistency. This candidate must differentiate as target-conditioned voxel action-distribution decoding.

## RA-L Stability Estimate

Estimated kill risk before ActionMap anchor reproduction: `0.65`.

Main risks:

- ActionMap-style head again loses to mean action, linear/L1, or simple MLP.
- Target conditioning fails to beat ActionMap alone.
- Canonicalization-only explains paraphrase gains.
- Single-point or destination-only target injection explains grounding gains.
- Latest adjacent work makes the novelty too narrow unless the method is precisely framed.
- Real VLA/LoRA path remains non-green.

Risk after a clean ActionMap anchor win and non-leaking target-prior feasibility: approximately `0.40`.

## Required Next Gate

Before any target-grounded method work:

1. Reproduce an ActionMap-style anchor with an official-style feature/model path or a clearly justified stronger bounded path.
2. Compare against mean action, linear/L1, and simple MLP.
3. Check candidate diversity/collapse.
4. Preserve an oracle candidate upper bound only as invalid method evidence.
5. Stop if the anchor fails again.

## Final Report Fields

| Field | Value |
| --- | --- |
| branch | `codex/research-reset-target-grounded-actionmap-scout` |
| experiments happened | no |
| training happened | no |
| download/GPU/OpenVLA-OFT happened | no / no / no |
| all previous prompts/routes consolidated | yes |
| salvage matrix complete | yes |
| recommended direction | Target-Grounded ActionMap / Language-Grounded Action Heatmap, gated by ActionMap anchor reproduction |
| final decision | `NEED_ACTIONMAP_ANCHOR_REPRO_FIRST` |
| estimated kill risk | `0.65` before anchor reproduction |
| exact next prompt if decision is `GO_TARGET_GROUNDED_ACTIONMAP_STATE1` | not applicable; decision is not GO |
