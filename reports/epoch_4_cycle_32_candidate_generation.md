# Epoch 4 Cycle 32 Candidate Generation

Date: 2026-07-16 KST

Decision: `LCG_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Previous method: `S2C-VLA`

Previous fixed result: `S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE`

S2C remains closed. No S2C repair, rescue, threshold change, task change, proxy
change, or reinterpretation is allowed.

Exactly three candidates were generated.

## Candidate 1: LCG-VLA

Full name: Language-Contrastive Guidance for Base-preserving SmolVLA actions.

Closest prior: Counterfactual Action Guidance.

Primary source: `https://arxiv.org/abs/2602.17659`

Contribution type: `PRIOR_EXTENSION`.

Scientific mechanism: compare SmolVLA action chunks under the original task
instruction and a legal language-null or counterfactual-language branch, then
learn an identity-initialized action-cell gate that allows bounded edits only
where language contrast predicts a vision-shortcut risk.

Why it is distinct: CAG uses a dual-branch inference scheme with a
language-unconditioned VA module. LCG keeps frozen SmolVLA Base as default,
uses existing demonstrations to learn a deployment-observable language-contrast
gate, and requires exact Base passthrough when the contrast is absent.

First serious comparison:

1. `smolvla_base`
2. `counterfactual_action_guidance_proxy`
3. `lcg_full`
4. `lcg_no_language_contrast_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `23 / 25`;
- importance of problem: `15 / 15`;
- positive prior anchor: `19 / 20`;
- technical mechanism quality: `18 / 20`;
- data/supervision feasibility: `9 / 10`;
- decisive experiment feasibility: `9 / 10`;
- total: `93 / 100`.

Assessment: selected. It targets a failure axis not tested by S2C: language
conditioning versus vision shortcut. It can be audited with existing
instructions, SmolVLA Base actions, and demonstration actions without privileged
inference inputs.

## Candidate 2: TAGR-VLA

Full name: Target-Agnostic Residual Guidance for clutter-biased SmolVLA
actions.

Closest prior: TAG.

Primary source: `https://arxiv.org/html/2603.24584v1`

Contribution type: `PRIOR_EXTENSION`.

Scientific mechanism: contrast original-observation and erased-observation
SmolVLA chunks, then route a bounded residual only when distractor-sensitive
action drift is observed.

First serious comparison:

1. `smolvla_base`
2. `tag_target_agnostic_guidance_proxy`
3. `tagr_full`
4. `tagr_no_observation_contrast_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `22 / 25`;
- importance of problem: `14 / 15`;
- positive prior anchor: `18 / 20`;
- technical mechanism quality: `17 / 20`;
- data/supervision feasibility: `7 / 10`;
- decisive experiment feasibility: `8 / 10`;
- total: `86 / 100`.

Assessment: not selected. The anchor is strong, but local object-erasure quality
is uncertain without masks or a new perception stack, increasing data-failure
risk.

## Candidate 3: PGP-VLA

Full name: Progress-Gated Policy correction for SmolVLA chunks.

Closest prior: ProgressVLA.

Primary source: `https://arxiv.org/abs/2603.27670`

Contribution type: `PRIOR_EXTENSION`.

Scientific mechanism: train a deployment-observable progress probe from
demonstration frame position and cached visual/action features, then use
progress residuals to gate bounded Base-preserving chunk corrections.

First serious comparison:

1. `smolvla_base`
2. `progressvla_proxy`
3. `pgp_full`
4. `pgp_no_progress_gate_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `21 / 25`;
- importance of problem: `14 / 15`;
- positive prior anchor: `18 / 20`;
- technical mechanism quality: `16 / 20`;
- data/supervision feasibility: `8 / 10`;
- decisive experiment feasibility: `8 / 10`;
- total: `85 / 100`.

Assessment: not selected. It is plausible, but previous campaign cycles already
found several progress/phase-like proxies easy to overfit or explain with task
phase baselines. LCG has a sharper prior-first comparison and a cleaner
deployment contrast.

## Selection

Selected method: `LCG-VLA`

Selection decision: `LCG_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Next stage: freeze the LCG-VLA Researcher A proposal before Reviewer B attack,
mathematical audit, preregistration, prototype protocol, implementation,
validation search, training, or rollout.
