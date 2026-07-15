# FAMR-VLA Researcher A Proposal

Date: 2026-07-15 KST

Decision: `FAMR_RESEARCHER_PROPOSAL_READY_FOR_REVIEW`

## Method Identity

Method: `FAMR-VLA`, Function-Aware Model Retention for VLA policies.

Contribution type: `PRIOR_EXTENSION` and
`CROSS_DOMAIN_MECHANISM_TRANSFER`.

Closest external prior: RETAIN,
https://arxiv.org/abs/2512.08333, official code
https://github.com/yajatyadav/RETAIN_code.

Secondary mechanism prior: Fisher-weighted model merging,
https://arxiv.org/abs/2111.09832.

## Research Question

Can a new-task VLA checkpoint be merged with its generalist initialization by
measuring module-group action responses, so that it learns held-out LIBERO-90
skills better than Base and scalar RETAIN while preserving the original
40-task policy better than ordinary finetuning?

The narrow claim is new-skill adaptation with generalist retention. It is not a
claim about generic PEFT efficiency, universal model merging, robustness to all
distribution shifts, or a repair of IARC.

## Scientific Method

Let `theta_0` be the frozen `lerobot/smolvla_libero` checkpoint and let
`theta_ft = theta_0 + Delta` be a standard new-task finetuned endpoint. Split
the effective task vector into fixed groups `Delta_m`.

For development observation `x_i`, evaluate Base and one group at a time with
the same flow noise and obtain the postprocessed 7D action response

`d_im = a(x_i; theta_0 + Delta_m) - a(x_i; theta_0)`.

With `D_i = [d_i1, ..., d_iM]`, solve

`min_{0 <= c <= 1} mean_{i in T} Huber((a_0i + D_i c - a_i*) / s)`
`                    + lambda mean_{i in R} ||D_i c / s||_2^2`.

`T` contains new-task demonstration rows, `R` contains original-task retention
rows, `s` is a frozen per-action scale from discovery data, and `lambda` is
selected on validation only. The resulting checkpoint is

`theta_FAMR = theta_0 + sum_m c_m Delta_m`.

The linear response is a development model, not confirmatory evidence. Every
selected policy is materialized, disk reloaded, and evaluated directly.

Primary objective: normalized Huber target-action fit.

One necessary auxiliary: normalized original-task action-drift retention.

Key ablation: identical selected grouping and solver with `lambda = 0`.

## Groupings

Coarse grouping, `M = 3`:

1. VLM attention LoRA (`vlm_with_expert` q/v projections);
2. action-flow LoRA (action input/output and time MLP projections);
3. state-projection LoRA.

Fine grouping, `M = 5`:

1. VLM layers `0-7`;
2. VLM layers `8-15`;
3. action input/output projections;
4. action time MLP projections;
5. state projection.

Every trainable parameter must map to exactly one group. Missing, duplicated,
or unassigned trainable parameters are implementation failures.

## Low-Compute Parameterization

`SCIENTIFIC_METHOD`: constrained checkpoint merging in VLA action-function
space.

`LOW_COMPUTE_PARAMETERIZATION`: one zero-effect rank-4 LoRA task vector on the
official SmolVLA checkpoint, using the same target modules already validated
locally. LoRA only makes `Delta` trainable and group-scalable on one GPU. FAMR
is defined equally for full checkpoints.

Each coefficient scales the effective low-rank update by scaling the group's
LoRA `B` tensors after training. Since `Delta W = scale * B A`, multiplying
`B` by `c_m` multiplies that group's effective task vector by exactly `c_m`.
Base weights and LoRA `A` tensors are unchanged.

## New-Task Source

The local SmolVLA checkpoint was trained on `lerobot/libero`, whose official
dataset card reports `40` tasks. The new-skill source is the separate official
raw `libero_90` distribution.

Three target tasks are frozen for the prototype:

1. `KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf`;
2. `LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray`;
3. `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy`.

They span distinct scenes and object/receptacle relations while remaining
single-skill tasks suitable for bounded adaptation.

Raw demonstration split per task:

- training episodes: `0-34`;
- validation episodes: `35-44`;
- sealed offline test episodes: `45-49`.

The HDF5-to-policy mapping must be audited against official LIBERO observation,
state, action, camera, and task semantics before training. No test episode is
decoded during development.

## Original-Task Retention Source

Use the existing official 40-task stable prediction artifact partitions:

- discovery/train rows may fit the response objective and action scales;
- validation rows may select one configuration;
- the `1,200` test rows remain sealed until configuration freeze.

Clean retention is ultimately a closed-loop success claim, not an offline
action-L2 claim.

## Headroom Audit

Before full merge search:

1. verify all three new-task HDF5 sources contain finite, noncollapsed actions
   and at least `45` usable demonstrations;
2. verify Base has at least `25%` failure on the fixed discovery reset manifest;
3. verify expert demonstrations are successful diagnostic oracles;
4. train the one standard-LoRA endpoint and verify a fixed subset fit;
5. verify the endpoint has a nontrivial new-task action effect and does not
   catastrophically destroy original-task validation behavior;
6. verify group responses are finite, nonzero, and identifiable.

No headroom, invalid raw mapping, or a nonacting endpoint stops before
confirmatory rollout and is classified precisely.

## Bounded Validation Search

Exactly six selectable configurations share one trained endpoint:

| ID | Mechanism | Grouping | Coefficient |
| --- | --- | --- | --- |
| `famr_fine_l01` | FAMR | fine | `lambda=0.1` |
| `famr_fine_l1` | FAMR | fine | `lambda=1.0` |
| `famr_fine_l10` | FAMR | fine | `lambda=10.0` |
| `famr_coarse_l1` | FAMR | coarse | `lambda=1.0` |
| `retain_a05` | scalar RETAIN proxy | scalar | `alpha=0.5` |
| `retain_a08` | scalar RETAIN proxy | scalar | `alpha=0.8` |

The selected FAMR grouping is then reused with `lambda=0` for the key ablation.
Base and standard LoRA are fixed endpoints, not search configurations.

All six receive the same offline validation rows. The top two FAMR
configurations by frozen offline composite receive the fixed validation rollout
screen; RETAIN uses its better of two scalar configurations on the same
validation identities. One final FAMR configuration is selected by the frozen
validation score. No confirmatory identity may influence selection.

Validation score:

`0.45 * target_success_proxy`
`+ 0.20 * original_task_retention`
`+ 0.15 * action_validity`
`+ 0.10 * response_fidelity`
`+ 0.10 * zero_extra_inference_overhead`.

When validation closed-loop results exist, they replace the target offline
proxy and clean offline proxy in their corresponding components. Ties within
`0.01` select the coarser grouping, then larger retention, then smaller
coefficient norm.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base`;
2. `retain_scalar_proxy`;
3. `famr_full`;
4. `famr_target_only`;
5. `standard_lora_new_task`.

All adapted arms share checkpoint initialization, demonstrations, split,
training steps, optimizer, seed, rank, target modules, processors, and action
semantics. The RETAIN arm is a transparent local proxy, not an official openpi
reproduction.

## Paper-Candidate Gate

FAMR becomes a prototype GO only if it beats Base, scalar RETAIN, the
target-only ablation, and standard LoRA on matched new-task resets; retains
original-task success; preserves action validity; and shows that nonuniform
functional coefficients, rather than simple shrinkage, account for the gain.

After GO, apply the same scientific method to Quantized OpenVLA-OFT INT4 and
add one claim-specific second condition before paper packaging.

## Current Boundary

This proposal authorizes Reviewer B attack only. It does not authorize
training, validation search, or rollout.
