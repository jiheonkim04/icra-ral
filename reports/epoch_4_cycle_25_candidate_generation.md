# Epoch 4 Cycle 25 Candidate Generation

Date: 2026-07-16 KST

Exactly three candidates are evaluated under the active prior-first,
performance-oriented, minimum-sufficient-method governance. VDR repair or
rescue is not a candidate.

## Candidate 1: RAP-VLA

Name: `RAP-VLA`, Retrieval-Anchored Prior residualization for VLA action flows.

Contribution type: `PRIOR_EXTENSION`.

Closest positive prior: OptimusVLA.

### Scientific Method

Build a deployment-legal action memory from discovery demonstrations. Each
memory entry stores current visual-policy features, proprioception, task text,
phase features, and the normalized expert action chunk. At inference and
training time, the current legal observation retrieves top-k similar memory
chunks and forms an anchor distribution in action-chunk space.

RAP does not directly replay the retrieved action. Instead, it trains SmolVLA
with an anchor-centered residual flow objective:

`expert_action_chunk = retrieved_anchor_chunk + bounded_policy_residual`.

The retrieved anchor supplies a legal action prior; the policy learns only the
state-specific residual needed to adapt the anchor to the current observation.
The anchor influence is controlled by a zero-initialized gate, so initial
behavior is Base passthrough. LoRA is only the low-compute implementation
scaffold for the residual/gate path, not the scientific mechanism.

Unlike OptimusVLA, RAP does not combine a GPM sampler and an LCM smoother as a
black-box memory controller. Its claim is narrower: residualizing action-flow
learning around retrieved legal action anchors improves action generation more
than a transparent OptimusVLA-style memory-prior proxy, an anchor-only
ablation, or standard LoRA.

Mechanism chain:

`isotropic action-flow prior + limited local demonstrations -> generated chunks
must discover a narrow task action distribution from noise -> invalid or
unstable chunks and weak task-phase commitment -> closed-loop failure`

`retrieved legal action anchor + residualized bounded flow learning -> policy
starts near a plausible task/phase chunk but must explain current-state
deviation -> bounded valid action changes and better temporal/task alignment ->
improved closed-loop success`.

### Quality Screen

Provisional novelty:

- RAP is distinct from OptimusVLA because it residualizes learning around the
  retrieved action prior instead of using memory actions as a sampler prior plus
  LCM correction.
- RAP is distinct from RAR because it retrieves from current observation/task
  similarity and predicts a bounded residual around an external memory anchor,
  not an action-history residual state.
- RAP is distinct from KITE/VDR because it supervises legal action-chunk
  residuals, not future end-effector or future visual-feature consequences.

Prior-anchor strength:

- OptimusVLA has a current CVPR 2026 primary-source result, official
  repository, inference code, LIBERO memory assets/checkpoints, and reported
  LIBERO gains.
- The closest-prior proxy can be implemented transparently using the same local
  memory, top-k retrieval, Gaussian anchor distribution, and optional
  lightweight LCM-style action-history smoothing.
- Ours, prior proxy, ablation, Base, and standard LoRA can share SmolVLA,
  demos, tasks, reset identities, inference budget, and action postprocessing.

Mechanism plausibility:

- Problem condition -> current SmolVLA flow must generate a narrow legal action
  chunk from a broad prior on small task-local data.
- Intermediate failure -> sampled chunks can drift from valid task/phase
  action modes or rely on generic Base behavior.
- Policy representation/action behavior -> output has unnecessary action
  variance or poor commitment to the correct local phase.
- Closed-loop failure -> grasp, placement, or approach phase destabilizes.
- Proposed method -> retrieve legal action anchors and train only a bounded
  residual around them.
- Intended internal change -> the action flow is organized around task-relevant
  legal modes while still adapting to current observation features.
- Intended action behavior -> lower invalid/destabilizing deviations, stronger
  phase consistency, and better task progress.
- Expected closed-loop improvement -> better success than Base, OptimusVLA
  proxy, anchor-only ablation, and standard LoRA.

Data and supervision viability:

- Required labels exist in local LIBERO demonstrations: observations,
  proprioception, language/task identity, timestamps/phase features, and expert
  action chunks.
- Positive/negative examples are noncollapsed if nearest anchors beat task/phase
  mean chunks on validation and residual variance remains positive.
- No privileged inference input is required: retrieval uses only current
  observation/language/proprioception and stored training memory.
- Supervision can be generated within local budget from cached SmolVLA visual
  features and existing HDF5 actions.

Identity-preserving integration:

- Gate initializes to Base passthrough.
- Residual branch initializes at zero.
- Anchor residual magnitude is bounded and audited before rollout.
- Clean validation behavior and postprocessed action validity are mandatory
  gates.

Decisive experiment feasibility:

- Stage 0 can audit retrieval health, residual variance, nearest-anchor
  headroom, noncollapsed residual targets, action validity, identity, reload,
  and gradients without simulator access.
- The first serious comparison is exactly five policies: Base, OptimusVLA proxy,
  RAP full, anchor-only/no-residual ablation, and standard LoRA.
- The simple explanation "ordinary fine-tuning explains the gain" is tested by
  matched standard LoRA; direct retrieval is tested by the anchor-only ablation.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 23 |
| Problem importance | 15 | 14 |
| Positive prior anchor | 20 | 20 |
| Technical mechanism | 20 | 19 |
| Data/supervision feasibility | 10 | 9 |
| Decisive experiment feasibility | 10 | 9 |
| Total | 100 | 94 |

## Candidate 2: PBC-VLA

Name: `PBC-VLA`, Past-Behavior Consistency for SmolVLA flow policies.

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`.

Closest positive prior: Past-Token Prediction for long-context diffusion
policies.

### Scientific Method

Train a small auxiliary head to reconstruct recent executed action tokens from
the current policy representation while preserving ordinary future action-flow
training. At inference, sample a bounded candidate set and select the candidate
whose implied past-action reconstruction is most consistent with the actual
executed history.

The mechanism is a VLA-local transfer of PTP's past-token self-verification
idea, not a generic action-history residual memory.

### Quality Screen

- Provisional novelty: meaningful as a VLA flow self-verification transfer, but
  close to RAR and other action-history routes already explored locally.
- Prior anchor: PTP has an official project and code, and reports strong
  long-context diffusion-policy gains, but it is not a VLA/LIBERO-specific
  positive prior.
- Data viability: local demonstrations contain past/future action chunks, but
  LIBERO short-horizon tasks may not require enough long-context dependency.
- Identity: can use zero-initialized auxiliary head and Base passthrough.
- Decisive experiment: feasible, but reviewer risk is high if the gain is
  explained by ordinary temporal smoothing or RAR-like memory.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 18 |
| Problem importance | 15 | 13 |
| Positive prior anchor | 20 | 17 |
| Technical mechanism | 20 | 17 |
| Data/supervision feasibility | 10 | 8 |
| Decisive experiment feasibility | 10 | 8 |
| Total | 100 | 81 |

## Candidate 3: AHE-VLA

Name: `AHE-VLA`, Attention-Horizon Execution for SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest positive prior: AutoHorizon.

### Scientific Method

Use action self-attention maps from SmolVLA's flow transformer to estimate the
reliable prefix length of each generated chunk, then execute only that prefix
before replanning. A lightweight calibration layer may map attention turning
points to one of a small number of legal execution horizons.

### Quality Screen

- Provisional novelty: low-to-moderate because EAC/AAC already tested adaptive
  chunk execution in this campaign, though AHE would use attention structure
  rather than entropy.
- Prior anchor: AutoHorizon has a current primary-source result and official
  repository with Pi0.5/LIBERO example code.
- Data viability: no labels are required, but SmolVLA attention extraction may
  not expose comparable action self-attention maps without invasive hooks.
- Identity: inference-only, no weight change; horizon defaults to Base.
- Decisive experiment: feasible if hooks work, but the first serious comparison
  risks repeating the EAC Stage B simple-baseline failure.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 15 |
| Problem importance | 15 | 13 |
| Positive prior anchor | 20 | 18 |
| Technical mechanism | 20 | 15 |
| Data/supervision feasibility | 10 | 7 |
| Decisive experiment feasibility | 10 | 7 |
| Total | 100 | 75 |

## Selection

Select exactly one candidate: `RAP-VLA`, `94 / 100`.

RAP has the strongest current positive prior anchor, the cleanest path to use
existing LIBERO demonstrations without privileged inference inputs, and the
clearest novelty separation from the immediately closed KITE/VDR mechanisms.
It keeps a single scientific mechanism: retrieved legal action anchors plus
bounded residualized action-flow learning. LoRA is only implementation
infrastructure.

## Baseline Rationale

| Comparison | Scientific question |
| --- | --- |
| Base vs RAP | Does retrieved-anchor residualized action-flow learning improve SmolVLA? |
| OptimusVLA proxy vs RAP | Does residualizing around retrieved anchors beat a memory-prior sampler plus lightweight consistency proxy? |
| Anchor-only ablation vs RAP | Is learned current-state residualization necessary beyond direct retrieved action priors? |
| Standard LoRA vs RAP | Is any gain explained by ordinary data-matched adaptation rather than the retrieval-anchor residual objective? |

The first serious comparison contains exactly five policies: Base, transparent
OptimusVLA proxy, RAP full, anchor-only/no-residual ablation, and matched
standard LoRA.

## Frozen Next Gate

Researcher A must now write one bounded proposal for RAP-VLA. Before expensive
training or rollout, Stage 0 must prove on discovery/validation identities only:

- discovery/validation/test memory separation and zero overlap;
- noncollapsed retrieval neighborhoods and top-k diversity;
- nearest retrieved anchors beat task/phase mean action chunks on validation;
- residual targets have positive variance and are predictable above a trivial
  baseline from legal deployment inputs;
- anchor-only ablation is distinct from RAP's learned residual path;
- postprocessed action validity and Base-relative deltas are bounded;
- checkpoint reload, Base passthrough, and finite nonzero trainable gradients;
- no simulator rollout, reward/success/done field, or confirmatory identity
  access occurs in Stage 0.
