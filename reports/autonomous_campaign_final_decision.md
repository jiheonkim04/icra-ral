# Autonomous Campaign Final Decision

Date: 2026-07-11 KST

Final decision: `NO_METHOD_AFTER_3_VALID_CYCLES`

## Stop Basis

This batch did not produce `PAPER_READY_EXPERIMENTAL_PACKAGE`.

It also did not launch a new prototype, because all three remaining genuinely distinct method families failed before implementation under the governance rule that Reviewer B must demand implementation, decisive experiment, or KILL. Here the correct outcome is KILL.

## Distinct Cycle Count

Cycle 01: action conditioning and action representation.

- changed core problem: action-generation interface and temporal representation;
- changed representation: latent action/effect-equivalence/action heatmap;
- changed objective: action/effect representation learning;
- decision: `KILL_NOVELTY_AND_LOCAL_HEADROOM_COLLAPSE`.

Cycle 02: intervention-censored correction credit.

- changed core problem: temporal credit under corrections;
- changed training signal: censored success credit and positive/negative corrective chunks;
- changed closed-loop intervention: correction/truncation/residual recovery;
- decision: `KILL_RECENT_WORK_AND_FEASIBILITY_COLLAPSE`.

Cycle 03: contact barrier and irreversibility boundaries.

- changed core problem: contact/constraint/irreversibility;
- changed representation: phase-conditioned barrier or recoverability margin;
- changed action-generation mechanism: constrained generation or boundary-triggered intervention;
- decision: `KILL_BASELINE_PRIOR_ART_AND_EVIDENCE_COLLAPSE`.

The cycles are genuinely distinct under the governance correction because each changes at least two of core problem, representation, training signal, objective, action-generation mechanism, and closed-loop intervention.

## Evidence That Prevented Implementation

Local evidence:

- ECHO final headroom: `NO_ECHO_HEADROOM_CONFIRMED`; official-policy oracle improvement `0.0` percentage points and recoverable default-failure rate `0.0`.
- ActionMap mini-anchor: local learned heatmap/candidate head lost to mean-action and cheap MLP and collapsed.
- Quantized OpenVLA-OFT INT4 hard slice: `20/20` successes, while matched SmolVLA was `11/20`; the hard-slice mechanism was SmolVLA-specific under this gate.
- Closed-loop visual gate: no single repeated mechanism across enough independent resets/tasks.
- ContactSet and ContactTube routes: killed by simple geometry/retargeting baselines.

Latest literature:

- CAC-VLA, ACoT-VLA, LaRA-VLA, ActionMap, LARA, LAWM, and AEM occupy action-conditioning and latent/action-effect representation.
- TORL-VLA, SDP, AFIL, BORA, VLA-Corrector, and Pre-VLA occupy correction, censored/intervention credit, failure-negative guidance, residual adaptation, and verification.
- VeriSpace, Pre-VLA, VLA-Corrector, AAC, SEAM, Legato, and TORL-VLA occupy the contact/barrier/irreversibility neighborhood unless new tactile/robot hardware or much stronger local mechanism evidence exists.

## Resource Use

- new downloads: `0 GiB`
- active GPU time: `0 h`
- new training: false
- new simulator rollout: false
- pushed to main: false

## Paper-Ready Status

Paper-ready status: `false`

No selected method, no new GO prototype, no second-backbone positive result for `ours`, no second condition, and no final paper package exist.

## Recommended Future Reopen Conditions

Do not reopen this campaign with another generic VLA method prompt. Reopen only if at least one concrete unblocker appears:

- official ActionMap or another action-representation baseline is reproduced locally and leaves a nontrivial residual not solved by simple MLP/mean baselines;
- a new cross-backbone failure appears where both SmolVLA and Quantized OpenVLA-OFT INT4 fail under matched exact states;
- new hardware/data becomes available, especially physical robot intervention data, tactile/force signals, or a 24GB+ GPU enabling full-precision second-backbone training/evaluation;
- a new paper explicitly exposes a gap not covered by current action-conditioning, correction, verification, progress, chunking, or contact-adaptation methods.

