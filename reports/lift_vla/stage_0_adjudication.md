# LIFT-VLA Stage 0 Adjudication

Date: `2026-07-15 KST`

Proposal hash: `3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`

Final decision: `LIFT_COMPUTE_INFEASIBLE`

## Protocol Integrity

The final admissible run used the frozen `libero_goal` target partitions,
same-scene pairing rule, two reset identities per target, empty task string,
same-noise coupling, ten Euler steps, three guidance scales, four policies,
native-space CAG, matched-compute last-step ablation, thresholds, and action
bridge. It decoded zero confirmatory policy observations, computed zero
confirmatory policy actions, ran no validation search, and ran no rollout.

The source/scorer manifest retained all `20 / 20` rows. It contains `4 / 3 / 3`
discovery/validation/confirmatory target tasks, `14` scoreable development
episodes, one shared scene signature, zero partition overlap, target-BDDL
success scorers, and no old `offline_proxy_only` pair.

## Bounded Repairs

Three implementation-only repairs restored the preregistered execution:

- the first native MuJoCo attempt produced no result artifact after caching ten
  environments; target environments were changed to bounded one-at-a-time
  ownership without changing any row or scorer;
- the preserved `implementation_blocker.json` records a missing batch dimension
  on direct nested robot observations; adding the one-env batch dimension
  restored the canonical vector-env processor input;
- `omega = 1` originally incurred floating cancellation in
  `v_u + (v_c - v_u)`; both fields remain evaluated, while the algebraically
  identical conditioned field is used directly for the frozen identity audit.

No repair changed a task, reset, policy, scale, threshold, sampler, branch,
postprocessor, or observed scientific outcome.

## Passing Gates

- native shape: `[1, 50, 32]`;
- policy shape: `[1, 50, 7]`;
- language and empty-language shapes: `[1, 48]`;
- Base and `omega = 1` LIFT native/postprocessed maximum error: `0.0`;
- conditioned-minus-empty activation: `1.0` at every audited scale;
- CAG, LIFT, and ablation field evaluations: `20` each;
- peak CUDA allocation: `0.9200425148010254 GiB`, below `15.5 GiB`;
- median LIFT/Base latency ratio: `2.013133036365988`, below `4.0`;
- all sampled outputs finite;
- practical discovery separation exceeded the frozen thresholds.

The discovery thresholds were persisted before any validation decode:
`tau_native = 0.00897083342075348` and
`tau_exec = 0.008641777038574218`.

## Decisive Stop

Executed-action bound validity was `0.8023809523809524`, below the frozen
required value `1.0`; the out-of-range fraction was
`0.19761904761904758`. Variant-specific clipping was not applied. This is the
preregistered action-validity hard stop and therefore
`LIFT_COMPUTE_INFEASIBLE`.

Adding clipping, changing the environment bridge, reducing a guidance scale,
selecting only valid dimensions, or proceeding to headroom would alter or
rescue the frozen protocol after observing the result. Those actions are
forbidden. The Base/CAG headroom rollout, bounded validation search, clean
retention evaluation, Stage A, and confirmatory test remain unexecuted.

## Continuation

LIFT Cycle 15 closes at the pre-rollout feasibility gate. This is not a
confirmatory scientific result about all pathwise language-guidance methods,
but the current frozen configuration may not be rescued. Epoch 4 Cycle 16 must
generate exactly three new candidates under the active governance and select
one materially distinct method.

