# MARC-VLA Prototype Protocol

Date: 2026-07-15 KST

Method: `MARC-VLA`.

Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`

## Prototype Policies

The first serious comparison contains exactly five policies:

1. `frozen_smolvla`
   - Frozen official SmolVLA policy.

2. `openvla_oft_l1_proxy`
   - Closest external-prior proxy.
   - Continuous L1/Huber action adapter with action chunk semantics.
   - Faithful transparent local proxy, not an official OpenVLA-OFT reproduction.

3. `marc_full`
   - Frozen SmolVLA plus identity-preserving median anchor and learned disagreement gate.
   - Must use the selected validation config and disk-reloaded checkpoint.

4. `marc_no_disagreement_gate_ablation`
   - Same anchor and comparable parameter budget.
   - Removes the learned state-dependent disagreement gate.

5. `static_l1_mixture_baseline`
   - One strongest simple reviewer-killer baseline.
   - Static validation-selected convex mixture of Base and the L1 proxy.

No additional internal controls may precede this five-policy comparison unless Stage 0 exposes a concrete implementation ambiguity that would otherwise invalidate one of the five policies.

## Stage 0 Command

Planned command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_marc_vla_development.py --mode audit
```

Stage 0 writes:

- `reports/marc_vla/development_audit.json`
- `reports/marc_vla/development_audit.md`
- `reports/marc_vla/disagreement_label_manifest.json`
- `reports/marc_vla/split_manifest.json`

Stage 0 must not train a policy or launch closed-loop rollout.

## Validation Search Command

Planned command after Stage 0 pass:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_marc_vla_development.py --mode validation
```

The bounded search may try at most the six preregistered configurations in `reports/marc_vla/preregistration.md`. It must save all tried configurations and negative results.

## Training Artifacts

Every trained policy must have:

- config JSON;
- training seed;
- source split manifest;
- disagreement-label manifest;
- checkpoint path;
- checksum;
- base checkpoint identity;
- dataset identity;
- validation metrics;
- action-delta metrics;
- gate activation metrics;
- disk-reload verification.

Stage A cannot begin until `marc_full`, `openvla_oft_l1_proxy`, and `marc_no_disagreement_gate_ablation` have disk-reloadable identities or the policy is explicitly nontrainable by design.

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
- gate activation;
- action delta from Base;
- residual/correction norm;
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

MARC cannot become a paper candidate unless `marc_full` beats Base, the L1 prior proxy, the no-gate ablation, and the static mixture baseline under the matched protocol.

If `openvla_oft_l1_proxy` wins, MARC is not a useful local extension of the closest prior.

If `marc_no_disagreement_gate_ablation` wins, the learned disagreement gate is not useful.

If `static_l1_mixture_baseline` wins, the result is explained by static mixing rather than state-dependent median anchoring.

## Nonretroactivity

MARC cannot change DAGR, MTF, RAC, CAVM, RCV, PSE, or earlier results. All prior kills and non-GO decisions remain fixed.
