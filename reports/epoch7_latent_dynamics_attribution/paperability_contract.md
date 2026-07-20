# Latent-Dynamics Attribution Paperability Contract

Frozen: 2026-07-20T16:14:42+09:00, before any claim-specific X-VLA dynamics-intervention outcome.

Status: `PRIMARY_PROBLEM_VERIFICATION_ONLY`. This contract does not assert that the benchmark will work and authorizes no Ours design or training.

## Proposed archetype and claim

Archetype: concise RA-L benchmark/evaluation paper.

One-sentence claim: a reproducible paired simulator protocol that holds instruction, scene geometry, initial state, rendering, and policy RNG fixed while changing only latent object/environment dynamics can distinguish target-grounding failures from physical-completion failures that aggregate VLA task success cannot identify.

Minimum reviewer-defensible novelty: the deliverable must combine all four elements below and demonstrate that their combination changes scientific interpretation across multiple VLA policy families and manipulation families:

1. visually matched, semantics-fixed object/environment dynamics interventions rather than visible pose/lighting/scene changes or robot-joint faults;
2. independently validated correct-target contact;
3. official completion conditioned on correct-target contact;
4. public task/property mappings, intervention code, paired identity manifests, and auditable per-rollout artifacts.

The paper may not claim that decomposed grounding, physical perturbations broadly, or distributional evaluation broadly are novel. LIBERO-CF, RoboSemanticBench, Eva-VLA, LIBERO-PRO, J-PARC, PhAIL, and Beyond Binary Success own those neighboring components.

## Frozen Base, comparators, tasks, and identities

Discovery Base: `2toINF/X-VLA-Libero` revision `129e71460678b7236cee6fc9707f09d9fa0c3590`, source `2toinf/X-VLA` revision `6bc2513f5f1cbec715cc668b414392a6cae5c671`, float32, ten denoising steps, domain id 3, absolute end-effector controller, 360x360 agent and wrist cameras, ten standard settle steps, horizon 300.

Strong comparator/control ecosystem:

- matched standard-dynamics X-VLA rollouts;
- no-op factor-1 intervention to test implementation identity;
- exact-init official demonstration replay, with demonstration identity selected only by a frozen standard-replay rule, as a task-feasibility oracle rather than policy-performance evidence;
- LIBERO-CF-style MuJoCo contact traversal, refined to exact target-body subtrees and independently unit-tested;
- if discovery passes, retained SmolVLA and OpenVLA-OFT INT4 are competence-gated cross-policy comparators, with quantization disclosed.

Discovery tasks and state indices are frozen in `discovery_protocol.json`: Goal eval0 drawer opening, eval2 plate pushing, eval3 bowl-on-plate, and eval9 stove activation; state indices 0, 1, and 2 only. Confirmation indices 10, 20, and 30 are sealed. Policy and simulator seeds are deterministic functions recorded in the protocol and shared within every standard/intervention pair.

## Exact conditions and endpoint

All interventions occur after the paired environment has been reset to the saved state and completed the same ten standard-dynamics settle steps. Both paired rollouts consume the exact cached observation returned by the final settle step as their first policy input. The intervention changes only a named MuJoCo model array, followed by `sim.forward()` without observation regeneration or an environment step. Subsequent observations come only from ordinary paired `env.step` calls.

- eval0: multiply damping for `wooden_cabinet_1_middle_level` by 4.
- eval2: multiply all three friction components of collision geoms whose body is `plate_1_main` by 0.25.
- eval3: multiply both `body_mass` and `body_inertia` for `akita_black_bowl_1_main` by 8.
- eval9: multiply damping for `flat_stove_1_button` by 4.

The standard condition uses factor 1 with no mutation. The cached agent-view, wrist-view, and proprioceptive arrays used for the first policy call must be exactly shared; simulator qpos/qvel and render-defining model-array hashes must remain unchanged across the mutation. This rule is an outcome-independent repair after the first preflight showed that calling LIBERO's observation regeneration path itself changes observable/post-process state even for a no-op. That failed preflight is retained and no policy, reward, or success was queried.

Main endpoint: the paired change in official `OffScreenRenderEnv.check_success`, interpreted jointly with `target_contact_any` and `success_given_target_contact`. A claim-defining physical-completion failure is a standard-success/intervention-failure pair in which the intervention rollout contacts the correct target-body subtree.

Minimum practically meaningful discovery effect, calculated only on tasks whose altered-condition expert replay passes:

- at least a 20 percentage-point paired official-success drop;
- at least `max(3, ceil(0.25 * eligible_pairs))` standard-success/intervention-failure pairs;
- adverse pairs across at least two tasks and two manipulation families;
- at least two correct-target-contact-but-failed-completion episodes across at least two families;
- intervention-to-standard wins must not exceed standard-to-intervention losses.

## Retention, headroom, and falsification

Standard-retention requirement: X-VLA must complete at least 75% of eligible standard discovery episodes and succeed on at least three tasks spanning all three families. The new runner's no-op factor-1 control must preserve first observations and model arrays and must reproduce standard outcome/action semantics on its smoke identities.

Headroom test: for every task admitted to the primary analysis, a frozen official HDF5 action sequence must succeed under standard replay and at least one sequence selected by the standard-only rule must also succeed after the exact dynamics intervention. At least three tasks spanning all three families must pass. Expert actions are never policy inputs and expert replay is never counted as VLA success.

Closest-overlap falsification: rotate if the implemented protocol does not materially exceed J-PARC on object-versus-robot dynamics and grounding-conditional attribution, Eva-VLA/LIBERO-PRO on visually matched latent interventions, and LIBERO-CF/RoboSemanticBench on the physical-completion endpoint. A generic “VLAs fail under physics changes” result is insufficient.

Key controls:

- factor-1 no-op mutation and initial-observation equivalence;
- named-array before/after values and checksums;
- exact target-contact detector fixtures, including negative contacts with wrong objects;
- standard versus intervention pairs sharing all non-dynamics inputs;
- if the route reaches Stage A, a matched irrelevant-body mutation control and a severity response are mandatory.

## Generalization and resources

Paper-level generalization requires at least two competent VLA policy families, all three manipulation families, multiple reset identities, at least two physical-property axes, held-out intervention severities, and a repeatable failure taxonomy. If only one additional backbone is competent, the paper must add another task suite or a stronger public comparator; otherwise it does not receive paper-GO.

Resource envelope: discovery uses one live simulator and one model, less than 16 GiB WSL memory, less than 8 GiB CUDA allocation, less than 1 GiB new storage, and less than one GPU-hour. Full evidence must remain serially below 24 GiB host RAM, 16 GiB VRAM, 20 GiB new storage, 18 GPU-hours, and the existing 102 GiB free-space reserve.

Predicted main table columns: backbone, task family, property axis, standard success, altered success, paired effect and interval, target contact, completion conditional on contact, expert feasibility, time to contact/success, and counts for misgrounding, contact-without-completion, and post-contact recovery.

## Decision conditions

- Kill: invalid observation matching, failed no-op identity, fewer than three feasible tasks/all three families, incompetent standard Base, no practically meaningful paired gap, no contact-preserving failures across two families, or metric/infrastructure defects explaining the effect.
- Narrow: one of the two articulated tasks may be excluded only for a pre-outcome mapping or expert-feasibility failure, leaving at least eval2, eval3, and one articulated task. No other post-outcome task deletion is legal.
- Rotate: any kill condition, direct novelty collapse, or need for simulator-only property labels at policy inference. A failed benchmark cannot be rescued as a dynamics-adaptation method.
- Problem pass: `PROBLEM_VERIFIED_STRONG_COMPARATOR_RESIDUAL` only if all discovery gates pass with valid execution and legal headroom.
- Bounded expansion: `PROBLEM_PROMISING_NEEDS_BOUNDED_EXPANSION` only if every validity, competence, feasibility, and coverage gate passes and exactly one numerical effect gate misses by one episode; the expansion identities and size must already be frozen.
- Paper-GO: after a problem pass, confirmatory identities must reproduce the effect, at least two competent policy families must support the attribution, controls and severity analyses must pass, statistics and artifacts must be complete, and the novelty matrix must remain defensible against all closest work.
