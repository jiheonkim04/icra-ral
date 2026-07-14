# PESA-VLA Prototype Protocol

Date: 2026-07-15 KST

Method: `PESA-VLA`.

Proposal hash: `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`

## Prototype Policies

The first serious comparison contains exactly five policies:

1. `frozen_smolvla`
   - Frozen official SmolVLA policy.

2. `priorvla_style_proxy`
   - Closest external-prior proxy.
   - Frozen Base prior action plus standard adaptation expert and retention/query gate.
   - Faithful transparent local proxy, not an official PriorVLA reproduction.

3. `pesa_full`
   - Frozen SmolVLA prior action plus identity-preserving spectral-capacity adaptation and prior-query gate.
   - Must use the selected validation config and disk-reloaded checkpoint.

4. `pesa_no_spectral_no_prior_query_ablation`
   - Same available development data and comparable adapter budget where feasible.
   - Removes spectral energy selection and learned prior-query gating.

5. `standard_lora_or_clean_retention_baseline`
   - One strongest simple reviewer-killer baseline.
   - Selected on validation before confirmatory testing from standard fixed-rank LoRA/adapter or clean-retention LoRA mixture.

No additional internal controls may precede this five-policy comparison unless Stage 0 exposes a concrete implementation ambiguity that would otherwise invalidate one of the five policies.

## Stage 0 Command

Planned command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_pesa_vla_development.py --mode audit
```

Stage 0 writes:

- `reports/pesa_vla/development_audit.json`
- `reports/pesa_vla/development_audit.md`
- `reports/pesa_vla/query_label_manifest.json`
- `reports/pesa_vla/spectral_activation_manifest.json`
- `reports/pesa_vla/split_manifest.json`

Stage 0 must not train a policy or launch closed-loop rollout.

## Validation Search Command

Planned command after Stage 0 pass:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_pesa_vla_development.py --mode validation
```

The bounded search may try at most the six preregistered configurations in `reports/pesa_vla/preregistration.md`. It must save all tried configurations and negative results.

## Training Artifacts

Every trained policy must have:

- config JSON;
- training seed;
- source split manifest;
- query-label manifest;
- spectral-activation manifest;
- checkpoint path;
- checksum;
- base checkpoint identity;
- dataset identity;
- validation metrics;
- action-delta metrics;
- query activation metrics;
- spectral rank metrics;
- disk-reload verification.

Stage A cannot begin until `priorvla_style_proxy`, `pesa_full`, `pesa_no_spectral_no_prior_query_ablation`, and the selected `standard_lora_or_clean_retention_baseline` have disk-reloadable identities or the policy is explicitly nontrainable by design.

## Closed-Loop Manifest

Stage A and Stage B must use matched paired manifests:

- identical task keys across policies;
- identical reset identities across policies;
- no overlap with validation identities;
- no duplicate `(policy, task, reset)` keys;
- official LIBERO success condition;
- no post-result task or reset selection.

The manifest is frozen before each stage begins.

## Metrics

Primary:

- official closed-loop task success;
- task-balanced success;
- paired full-minus-baseline success deltas.

Secondary:

- paired wins/losses/ties;
- paired bootstrap confidence interval;
- relative failure-rate reduction;
- per-task success;
- clean-retention success;
- query gate activation;
- active spectral rank distribution;
- spectral entropy;
- translation, rotation, and gripper delta from Base;
- residual/adaptation norm;
- action validity;
- latency;
- VRAM;
- training time.

## Resume Policy

For long-running WSL training or rollout:

- run detached;
- save Linux PID;
- save heartbeat/status JSON;
- save stdout and stderr logs;
- save partial result JSON;
- save exact resume command;
- resume only missing `(policy, task, reset)` keys after interruption.

## Scientific Decisions

PESA cannot become a paper candidate unless `pesa_full` beats Base, the PriorVLA-style proxy, the no-spectral/no-prior-query ablation, and the selected standard-LoRA or clean-retention simple baseline under the matched protocol.

If `priorvla_style_proxy` wins, PESA is not a useful local extension of the closest prior.

If `pesa_no_spectral_no_prior_query_ablation` wins, the spectral/prior-query mechanism is not useful.

If `standard_lora_or_clean_retention_baseline` wins, the result is explained by simple adaptation or clean-retention mixing.

## Nonretroactivity

PESA cannot change MARC, DAGR, MTF, RAC, CAVM, RCV, PSE, or earlier results. All prior kills and non-GO decisions remain fixed.
