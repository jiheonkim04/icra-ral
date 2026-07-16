# CCIF-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `CCIF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Proposal: `reports/ccif_vla/researcher_proposal.md`

Proposal SHA-256:
`2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1`

Reviewer attack: `reports/ccif_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Accepted Narrow Novelty

Researcher A accepts Reviewer B's narrowed novelty boundary.

CCIF-VLA is not a generic coarse-to-fine VLA, not a latent-action interface in
the CAC-VLA sense, not a Coarse-to-Control reproduction, and not a new discrete
action-token decoder. The method claim is limited to:

`Base-preserving continuous coarse motor-intent residual constraint around an
already trained continuous SmolVLA chunk`.

If later mathematical audit or Stage 0 cannot preserve this boundary, CCIF must
stop before validation search or rollout.

## Accepted Prior Proxy Condition

Researcher A accepts that `coarse_to_control_continuous_proxy` remains policy 2
unless official Coarse-to-Control assets are installed and verified before the
first serious comparison.

The proxy must not be a strawman. It will use:

- the same legal deployment-observable inputs as CCIF;
- the same discovery/validation rows;
- the same coarse-intent labels and waypoint summaries;
- comparable optimizer, step, and parameter budget where applicable;
- direct intent-conditioned action generation or refinement;
- no worse feature access than CCIF.

The proxy will be labeled as a transparent local proxy, not as official
Coarse-to-Control, unless official assets are actually integrated and verified.

## Accepted Ablation And Simple Killer

Researcher A accepts:

- key ablation: `ccif_no_coarse_intent_ablation`;
- closest-prior policy: `coarse_to_control_continuous_proxy`;
- mandatory simple reviewer-killer: matched `standard_lora`.

The ablation will share the same adapter/scaffold, optimizer budget, residual
cap, clean retention policy, and action loss while removing the predicted
coarse intent from the residual path.

Standard LoRA remains live because CCIF modifies actions through trainable
low-compute policy infrastructure.

## Accepted Cheap Diagnostics

Before mathematical audit or implementation, the Stage 0 plan must include
cheap diagnostics for:

- task/phase mean intent;
- endpoint-only intent;
- low-pass or waypoint-only intent if it is cheaper than rollout and can expose
  trivial equivalence;
- Base-to-expert coarse residual headroom;
- prior-proxy residual headroom.

These diagnostics do not become extra closed-loop policy identities unless
they explain the validation signal and Reviewer B requires a frozen comparison.

## Accepted Freeze Requirements

Researcher A accepts that the mathematical audit and preregistration must freeze
before implementation:

- exact coarse-intent vector components;
- units and normalization statistics;
- waypoint count;
- split used to fit normalization;
- residual cap and gate range;
- pass/stop gates;
- train/validation/confirmatory identities;
- no confirmatory decode or action access during Stage 0;
- no simulator success, reward, done, object state, or reset identity in label
  construction.

Offline action L2 alone cannot select a configuration or support a paper claim.

## Accepted Identity And Leakage Conditions

Researcher A accepts:

- initialized and disk-reloaded CCIF must reproduce Base within `1e-6`;
- no privileged inference inputs are allowed;
- no confirmatory-test tuning is allowed;
- no task/reset changes after results are allowed;
- no TSC repair, rescue, threshold change, proxy change, or reinterpretation is
  allowed.

## Rebuttal Position

Reviewer B's concern is correct: coarse intent by itself is not novel enough.
The method only remains worth a mathematical audit if it is treated as a
conservative residual constraint around a strong pretrained continuous policy.

The next audit should therefore document the exact variables, tensor shapes,
intent construction, objective scales, gradient paths, identity-preserving
initialization, prior-proxy definition, no-intent ablation, cheap diagnostics,
and Stage 0 stop classes before any implementation.

Researcher A accepts all Reviewer B conditions and advances CCIF-VLA to
mathematical mechanism audit.
