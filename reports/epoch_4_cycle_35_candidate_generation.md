# Epoch 4 Cycle 35 Candidate Generation

Date: 2026-07-16 KST

Decision: `MHS_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Candidate count: exactly `3`

Previous method: `BRID-VLA`

Previous decision: `BRID_STAGE_0_NO_RESIDUAL_HEADROOM`

Governance: current post-COVI minimum-sufficient governance with one genuinely
new mechanism, LoRA only as implementation infrastructure, and the closest
external prior in the first serious comparison.

## Candidate 1: MHS-VLA

Full name: Mamba History State for Base-preserving SmolVLA

Closest prior: MTIL

Primary sources:

- https://arxiv.org/abs/2505.12410
- https://arxiv.org/html/2505.12410v3
- https://github.com/yulinzhouZYL/MTIL

Positive prior: MTIL reports full-history state-space imitation learning that
outperforms ACT and Diffusion Policy on ACT, Robomimic, LIBERO, and real-world
sequential manipulation tasks, with an official repository.

Contribution type: `PRIOR_EXTENSION`

Scientific method: learn a deployment-observable recurrent history state from
past observations/actions and the current instruction/observation, then use it
to produce a bounded residual gate around the frozen SmolVLA Base action chunk.
The default path is exact Base passthrough; LoRA or a lightweight adapter may
only parameterize the history encoder or small residual head.

Minimal difference from prior: MTIL replaces the policy with a full-history
Mamba imitation learner. MHS keeps SmolVLA Base fixed and uses the history state
only as a selective, identity-preserving residual mechanism when current-frame
Base behavior is history-ambiguous.

Mechanism chain:

- problem condition: current observation/instruction is insufficient for
  stage-dependent LIBERO decisions;
- intermediate failure mechanism: Base emits the same or near-same chunk for
  different hidden histories;
- policy representation/action behavior: MHS recurrent state disambiguates the
  episode phase and activates bounded corrections only in ambiguous states;
- expected closed-loop improvement: fewer skipped, repeated, or premature
  substeps without global action replacement.

Data and supervision viability: existing LIBERO demonstrations provide ordered
history, observations, instructions, and expert chunks. Cached SmolVLA Base
chunks provide residual targets. No rewards, success flags, object poses, future
frames, or held-out reset identities are used at inference.

Identity-preserving integration: residual branch initialized to zero; gate
initialized to Base passthrough; action deltas are capped by translation,
rotation, and gripper groups; clean-retention validation is mandatory.

First serious comparison:

1. `smolvla_base`
2. `mtil_history_state_proxy`
3. `mhs_full`
4. `mhs_no_history_state_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `24 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `95 / 100`

Rationale: MHS is selected because it changes the central failure axis from
residual action-density estimation to history-state disambiguation, has a
current positive prior with LIBERO evidence, preserves Base identity, and can be
audited from existing demonstrations before any rollout.

## Candidate 2: VQRM-VLA

Full name: Vector-Quantized Residual Modes for Base-preserving SmolVLA

Closest prior: Behavior Transformer

Primary sources:

- https://mahis.life/bet/
- https://github.com/notmahi/bet
- https://arxiv.org/abs/2206.11251

Positive prior: Behavior Transformer models multimodal continuous behavior with
discrete action bins and learned continuous offsets, and reports improved
performance over prior demonstrated-task baselines while capturing major modes.

Contribution type: `PRIOR_EXTENSION`

Scientific method: cluster Base residual chunks into discrete residual-mode
tokens, predict a token and small offset from deployment-observable inputs, and
reconstruct a capped residual around the frozen Base chunk. LoRA, if used, is
only a low-compute parameterization of the token/offset predictor.

Minimal difference from prior: BeT directly predicts full actions from
demonstration context. VQRM predicts only residual modes around SmolVLA Base and
retains exact zero-token Base passthrough.

Mechanism chain:

- problem condition: Base residuals may be multimodal and poorly fit by a
  denoising score model;
- intermediate failure mechanism: continuous regression or diffusion averages
  incompatible residual modes;
- policy representation/action behavior: discrete residual tokens preserve
  separate correction modes while offsets restore continuous precision;
- expected closed-loop improvement: mode-correct corrections without replacing
  strong Base actions.

Data and supervision viability: existing action chunks and cached Base chunks
are sufficient for residual codebooks, token labels, offsets, and validation
splits. The main risk is collapsed code usage.

Identity-preserving integration: zero residual code and offset initialize to
Base passthrough; intervention frequency and per-group deltas are capped.

First serious comparison:

1. `smolvla_base`
2. `behavior_transformer_action_mode_proxy`
3. `vqrm_full`
4. `vqrm_no_offset_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `23 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `91 / 100`

Rationale: VQRM is attractive because it directly tests residual mode structure
after BRID, but it is less directly VLA/LIBERO aligned than MTIL and could be
explained by simple code-frequency or mean-residual baselines.

## Candidate 3: IBER-VLA

Full name: Implicit Base Energy Reranking for SmolVLA action chunks

Closest prior: Implicit Behavioral Cloning

Primary sources:

- https://implicitbc.github.io/
- https://github.com/google-research/ibc
- https://arxiv.org/abs/2109.00137

Positive prior: IBC reports that energy-based implicit policies often
outperform explicit MSE or mixture-density behavior cloning on robotic tasks,
including discontinuous, multivalued, and contact-rich behaviors, with official
code.

Contribution type: `PRIOR_EXTENSION`

Scientific method: learn an energy function over observation/instruction and
bounded candidate action chunks around the frozen Base chunk. At inference,
IBER returns Base unless a candidate has a preregistered energy margin and
passes action-validity and clean-retention checks.

Minimal difference from prior: IBC chooses actions from an implicit policy over
the full action space. IBER restricts inference to a small Base-centered
candidate set and uses energy only as a conservative reranker.

Mechanism chain:

- problem condition: explicit residual predictors may average discontinuous or
  set-valued corrections;
- intermediate failure mechanism: Base is close but chooses the wrong local
  branch in ambiguous contact/ordering states;
- policy representation/action behavior: energy scoring ranks several bounded
  Base-near candidates without modeling a normalized action distribution;
- expected closed-loop improvement: choose a better local branch while avoiding
  global replacement of Base behavior.

Data and supervision viability: positives come from demonstration action chunks;
negatives come from bounded Base perturbations and mismatched discovery chunks.
No test identity, reward, success flag, object pose, or future observation is
used at inference. The main risk is trivial equivalence to L2 residual scoring.

Identity-preserving integration: Base is always an included candidate; energy
margin defaults to Base; candidate deltas are capped and invalid candidates are
discarded.

First serious comparison:

1. `smolvla_base`
2. `implicit_bc_energy_proxy`
3. `iber_full`
4. `iber_no_base_anchor_ablation`
5. `explicit_l2_residual_reranker`

Scores:

- provisional novelty: `22 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `87 / 100`

Rationale: IBER is a clean non-diffusion implicit-action mechanism, but the
local experiment must work hard to prove it is not just explicit residual L2
selection with a more complicated scorer.

## Selection

Selected method: `MHS-VLA`

Selected score: `95 / 100`

Selection decision: `MHS_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

MHS is selected because it is the best positive-prior-anchored shift away from
BRID's failed residual diffusion axis. It uses a current full-history imitation
prior with LIBERO evidence, introduces a distinct history-state representation,
preserves the pretrained SmolVLA policy by default, and has a bounded Stage 0
audit path from existing LIBERO demonstrations and cached Base chunks. Unknown
empirical performance is not a rejection reason. No MHS implementation,
training, validation search, rollout, or confirmatory-test access has happened.
