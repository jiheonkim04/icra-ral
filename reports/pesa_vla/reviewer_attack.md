# PESA-VLA Reviewer B Attack

Date: 2026-07-15 KST

Reviewed frozen proposal: `reports/pesa_vla/researcher_proposal.md`

Proposal hash: `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Closest Prior Attack

Closest prior: PriorVLA, https://arxiv.org/abs/2605.10925.

PriorVLA already claims the central idea that a frozen pretrained VLA prior should remain an explicit read-only source while an adaptation expert learns downstream specialization. PESA-VLA cannot claim broad novelty for prior-preserving adaptation, frozen prior experts, expert queries, or using Base as a read-only action source.

The possible novelty is narrower:

- a local frozen SmolVLA 7D implementation with exact Base-passthrough initialization;
- combining a PriorVLA-style proxy with LoRA-SP/VLA-GSE-inspired spectral capacity allocation;
- enforcing clean-retention gates before official closed-loop rollout;
- a fair matched comparison against Base, a PriorVLA-style proxy, standard LoRA/simple clean-retention baseline, and a no-spectral/no-prior-query ablation.

The proposal must state clearly that `priorvla_style_proxy` is a faithful transparent local proxy, not an official PriorVLA reproduction, unless exact official equivalence is independently established.

## Secondary Prior Attack

LoRA-SP already claims adaptive spectral capacity allocation for VLA fine-tuning, including SmolVLA. VLA-GSE already claims generalized/specialized spectral experts with official code. CLARE, VLA-GSE, PriorVLA, and related PEFT papers crowd generic adapter routing, expert decomposition, and prior preservation.

PESA must not claim any of the following as new:

- fixed or adaptive LoRA for VLA adaptation;
- spectral low-rank capacity allocation;
- generalized and specialized experts;
- preserving pretrained VLM/VLA knowledge;
- adapter routing or expert gating by itself.

The local claim survives only if the full method's closed-loop behavior requires the combination of read-only prior action, spectral capacity activation, and clean-retention prior query under the frozen five-policy comparison.

## Trivial Equivalence Risks

1. Standard LoRA explains the method

If fixed-rank 7D LoRA matches or beats PESA under the same data and inference budget, PESA is not a method. The single simple killer baseline must be a strong standard LoRA or clean-retention LoRA baseline chosen on validation before confirmatory testing.

2. PriorVLA-style proxy explains the method

If the local PriorVLA-style proxy matches PESA, then spectral capacity allocation did not add a useful contribution. The proxy must use the same train/validation data, the same Base prior action, and comparable inference budget.

3. LoRA-SP or VLA-GSE explains the method

If spectral capacity alone, without explicit prior-query retention, matches PESA, then the method is just LoRA-SP/VLA-GSE under a local name. The no-prior-query and no-spectral ablations must isolate this.

4. MTF no-retention baseline warning

MTF's no-retention ablation previously beat the full MTF method and reached `32 / 40` in Stage B. PESA cannot ignore that a simple adaptation variant may dominate a structured method. Stage 0 must include a strong simple adaptation baseline before any rollout.

5. Clean-retention-only explains the method

If the gain is due only to clipping adaptation deltas or mixing back toward Base, then a simple clean-retention mixture explains the contribution. The mandatory simple killer must be allowed to be clean-retention LoRA when that is strongest on validation.

## Leakage And Partition Risks

- Adapter capacity thresholds, query gate definitions, clean-retention coefficients, and simple-killer selection must use discovery/validation only.
- Confirmatory task/reset identities may not influence spectral thresholds, LoRA rank, retention weights, policy list, or decision thresholds.
- If Base actions are generated for labels, the exact frozen Base checkpoint and preprocessing path must be recorded.
- Split overlap must be checked at sample, frame, task, episode, and reset identity levels.
- Any train-only query label based on "adaptation improves over Base" must not use confirmatory closed-loop outcomes.

## Mathematical Risks

- Do not compute KL between deterministic 7D action vectors.
- Define every spectral variable: `U_l`, `V_l`, `s_l(x)`, energy `E_l(k)`, active rank `k_l`, masking rule, tensor shapes, and gradient flow.
- Explain whether `s_l(x)` is normalized, nonnegative by construction, temperature-scaled, or clipped.
- Report term magnitudes and gradient norms for imitation, prior retention, spectral concentration, query BCE if used, and delta regularization.
- If the emitted action is clipped at inference, specify whether clipping is included in training or treated as a bounded deployment safeguard.
- Report per-group translation, rotation, and gripper deltas; a single 7D L2 can hide destructive gripper behavior.

## Data Health Risks

Reject before rollout if:

- fixed 7D action labels or Base actions are unavailable;
- standard LoRA has no development headroom over Base and mean/trivial baselines;
- the PriorVLA-style proxy cannot be constructed faithfully enough to be a fair closest-prior comparison;
- prior-query labels collapse or are phase/task shortcuts;
- spectral activation is always on, always off, or identical across tasks;
- the full method and ablation receive effectively identical targets;
- one task family dominates the adaptation-positive examples.

## Identity-Preservation Risks

PESA can damage a strong Base policy if the adaptation expert acts globally. Before rollout, require:

- exact initial equality to Base;
- finite nonzero gradients in the intended adapter/query parameters;
- action validity for all 7D outputs;
- action delta summaries by translation, rotation, and gripper;
- clean validation behavior retained;
- Base, proxy, full, ablation, and simple killer all loaded from frozen identities;
- checkpoint persists to disk and reloads to the same policy identity.

## Required First Comparison

The first serious comparison must use exactly five policies:

1. `frozen_smolvla`
2. `priorvla_style_proxy`
3. `pesa_full`
4. `pesa_no_spectral_no_prior_query_ablation`
5. `standard_lora_or_clean_retention_baseline`

Additional internal controls may not precede this comparison unless Stage 0 exposes a concrete implementation ambiguity that would otherwise invalidate the five-policy test.

## Reviewer Verdict

Do not reject before implementation. PESA-VLA is not an exact duplicate of PriorVLA, LoRA-SP, or VLA-GSE because it proposes a local frozen SmolVLA 7D integration where the prior action remains a read-only inference source and spectral capacity is constrained by identity-preserving clean-retention gates.

However, the novelty is conditional and narrow. The method is viable only if Researcher A narrows the claim, labels the closest-prior proxy honestly, and proves in Stage 0 that the full method differs from standard LoRA, PriorVLA-style proxy, and no-spectral/no-prior-query ablation before rollout.
