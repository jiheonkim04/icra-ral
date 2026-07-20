# Epoch 7 Thesis Portfolio and Selection

Decision date: 2026-07-20 (Asia/Seoul)

This file records two portfolio cycles. Cycle 1 selected paired semantic equivalence and intent selectivity, then closed it after the frozen action-energy mechanism test. Cycle 2 is a fresh method/benchmark/systems rotation informed by primary work available through 2026-07-20. Ratings are evidence gates, not promises to write a paper.

## Cycle 1 closure

The initial language-grounding route had a real Base problem: the retained original X-VLA checkpoint succeeded on 30/30 canonical LIBERO-Goal episodes and 19/30 matched LIBERO-Para episodes. A MiniLM canonicalizer still left 5/30 mapping errors, and a corrected CAG-TF comparator remained incompetent rather than saturating the problem. However, the selected real-action equivalence/selectivity mechanism failed its frozen Stage-0 falsifier. All 180 CUDA forwards were valid, but only 2/30 pairs across two tasks and one family violated the required ranking, below the gates of 6/30, three tasks, and two families. Candidate A is therefore closed. Paraphrase augmentation, post-outcome negative sampling, and a renamed energy ranker are prohibited rescues.

## Cycle 2 evidence boundary

The refresh materially changes the thesis boundary:

- LIBERO-Para and LIBERO-CF already separate paraphrase and counterfactual language failures, and RoboSemanticBench independently measures semantic target selection versus grasp success.
- Eva-VLA perturbs visible pose, lighting, and patches; LIBERO-PRO perturbs objects, positions, language, tasks, and environments.
- J-PARC evaluates robot-side joint locks, range limits, and joint friction on OpenVLA-OFT and pi0.5, and adds a learned residual calibrator.
- The June 2026 physical-reasoning position paper identifies a remaining attribution problem: task success alone cannot distinguish semantic grounding from physical execution, and it calls for controlled semantics-fixed changes in physical configuration, including mass, friction, and dynamics.
- No primary paper or official artifact found in the refresh supplies a manipulation-VLA protocol that pairs visually matched object/environment dynamics interventions with a separate correct-target-contact endpoint and conditional physical completion. This is a bounded novelty inference, not a universal absence claim.

## Candidate F — causal attribution under latent object dynamics

Assessment: `CLOSED_NO_LEGAL_HEADROOM`; selected primary failed its frozen pre-policy feasibility gate.

- One-sentence thesis: a paired, visually matched benchmark that holds instruction, geometry, and reset fixed while changing object mass, contact friction, or articulated resistance can reveal whether a VLA grounds the right target yet fails physical execution, which ordinary success rate cannot identify.
- Robotics problem and importance: deployed manipulators encounter payload, surface, and mechanism dynamics that are not recoverable from a single RGB frame; failure after correct approach/contact calls for a different remedy from semantic or visual misgrounding.
- What current evaluation misses: Eva-VLA changes visible inputs; LIBERO-PRO changes objects, positions, semantics, tasks, or environments; LIBERO-CF and RoboSemanticBench isolate language/target grounding; J-PARC changes the robot embodiment and reports task success. None of these tested artifacts jointly holds the visible task fixed, changes object/environment dynamics, and reports target grounding before conditional completion.
- Closest primary Prior: the physical-reasoning position paper (arXiv:2606.30686), J-PARC (arXiv:2606.10501), Eva-VLA (arXiv:2509.18953), LIBERO-PRO (arXiv:2510.03827), LIBERO-CF (arXiv:2602.17659), and RoboSemanticBench (arXiv:2606.02277).
- Base and comparator ecosystem: retained X-VLA-Libero is the discovery Base; retained SmolVLA and OpenVLA-OFT INT4 are cross-family expansion candidates. Standard dynamics, exact-init expert-action replay as a feasibility oracle, and the published grounding/contact construction from LIBERO-CF are comparators or controls, not Ours.
- Exact repeatable claim-specific condition: after `env.set_init_state`, preserve instruction, qpos/qvel, geometry, camera, textures, lights, and policy RNG while applying a frozen multiplier to only the claim-relevant MuJoCo array: target-body mass, target/support contact friction, drawer-joint damping, or stove-button joint damping. A condition is valid only if the first observations are byte-identical or numerically identical within a frozen tolerance and the model-array mutation is logged by name and checksum.
- Residual hypothesis: under moderate physically feasible changes, Base correct-target contact remains high while official success and success conditional on correct contact fall across at least two manipulation families.
- Diagnostic/task headroom: exact-init expert replay must remain successful under the altered condition, or a separately frozen legal controller oracle must establish task feasibility; standard X-VLA competence is already 30/30 on the ten-task Goal panel.
- Legal inference inputs: each policy receives only its normal RGB/proprioception and the unchanged instruction. MuJoCo property values, intervention labels, rewards, success, future observations, and expert actions are forbidden policy inputs.
- Possible method mechanism: none is proposed because this is a benchmark/evaluation thesis. A later diagnostic baseline may use observable contact or motion history, but it cannot receive simulator-only property labels.
- Closed-loop success path: official `OffScreenRenderEnv.check_success` under paired standard and altered dynamics, with complete serial rollouts and finite action checks.
- Standard-retention path: the protocol must reproduce the frozen standard-condition success of each selected Base within a prespecified tolerance before its altered-dynamics results count.
- Generalization path: at least three manipulation families (push, pick/transport/place, articulated actuation), at least two VLA policy families if both pass competence, multiple reset identities, held-out intervention severities, and more than one physical-property axis.
- Strongest novelty objection: the benchmark may look like a direct implementation of the position paper's recommendation or an object-side variant of J-PARC.
- Strongest experimental objection: a position-controlled LIBERO arm may overpower mass/friction changes, or extreme parameters may create only simulator artifacts.
- Strongest feasibility objection: task-specific target/contact mappings and exact expert replay under altered dynamics may not be reliable for enough families.
- Archived-route overlap: uses prior exact-init replay and serial runner infrastructure only. It does not revive action rescaling, contact-tube augmentation, recovery, memory, missing-view, uncertainty, language ranking, or executable-spec repair claims.
- Resource envelope: discovery is one live environment and one X-VLA model, under 4 GiB measured CUDA allocation, under 16 GiB WSL memory, under 1 GiB new artifacts, and roughly 10–30 minutes. A multi-backbone expansion is expected to remain under 16 GiB VRAM, 24 GiB host RAM, 20 GiB new storage, and 18 serial GPU-hours.
- Official artifacts and licenses: X-VLA source/checkpoint and LIBERO data are Apache-2.0; OpenVLA-OFT code is MIT with checkpoint-specific upstream terms; the retained SmolVLA/LeRobot path is Apache-2.0; LIBERO-CF code is MIT. Every immutable revision must be copied into the protocol.
- Cheapest falsifying experiment: three competent X-VLA Goal tasks representing drawer, push, and pick/place, three frozen reset identities each, paired standard versus one moderate family-specific dynamics change, plus initial-observation equivalence and exact mutation logs. No method is run.
- Required main table: per backbone and family, standard success, altered success, correct-target contact, completion conditional on contact, paired effect with interval, expert-feasibility rate, and taxonomy counts for misgrounding, contact-without-completion, and no-contact failures.
- Contribution if successful: a reproducible causal-attribution protocol showing that visually and semantically matched dynamics changes expose physical-execution failures hidden by aggregate VLA task success.

Closure: the outcome-free simulator preflight passed 4/4 tasks, but the frozen standard-only demonstration oracle verified altered-condition success for only drawer opening and bowl placement. Plate pushing and stove-button activation failed under their interventions despite target contact. This leaves two tasks across only two collapsed families, below the contract's required three tasks and all three families. No policy was loaded or queried and no X-VLA dynamics outcome was generated. The result is `NO_LEGAL_HEADROOM`, not a claim that the altered tasks are impossible. Alternate-demo outcome selection, post-hoc severity changes, family deletion, and a dynamics-method rescue are prohibited.

## Candidate G — language-grounded active physical-property selection

Assessment: `BLOCKED`.

- One-sentence thesis: a robot should actively interact with visually matched objects to resolve an instruction referring to a latent property such as “the heavier cup,” then manipulate the semantically correct target.
- Robotics problem and importance: physical attributes are often decision-relevant but invisible before contact, so web-semantic recognition is insufficient.
- What current evaluation misses: RoboSemanticBench tests explicit knowledge-to-target selection but not hidden physical attributes; the position paper proposes this conditional test but supplies no executable benchmark.
- Closest primary Prior: RoboSemanticBench, the physical-reasoning position paper, interactive perception/system-identification literature, and active information-gathering robot policies.
- Base/comparator ecosystem: no retained VLA is trained for probing-then-selecting latent properties, and no official local benchmark supplies paired objects, language, demonstrations, and success predicates.
- Exact condition and residual: two visually matched candidates differ only in a hidden property; the instruction identifies the target by that property. Success requires an information-gathering interaction, correct target selection, and task completion.
- Headroom and legal inputs: touch-derived motion is legal; simulator property labels are forbidden. An oracle property label would show headroom but would not make the Base competent.
- Possible mechanism: an interaction-memory state that converts observed action effects into an object-property belief before target commitment.
- Closed-loop, retention, and generalization paths: probing and selection success across mass, friction, and compliance, with ordinary named-object task retention and held-out property values/objects.
- Main objections: the closest paper already states the design; building valid tasks and demonstrations is the core contribution; a physical robot may be necessary for credible contact semantics.
- Archived overlap: distinct from language paraphrase/selectivity but overlaps closed memory and history-conditioned adaptation routes.
- Resources/artifacts: simulator compute fits, but data and competent checkpoints do not exist locally; a defensible release would require substantial new task authoring and training. LIBERO is Apache-2.0 and RoboSemanticBench is public, but its task format is not a drop-in source of latent-property demonstrations.
- Cheapest falsifier: verify whether any retained policy can complete a two-stage probe/select toy task without training. This is not worth running because the missing supervision and benchmark are already a hard portfolio veto.
- Required main table: property-conditioned target-selection accuracy, probe efficiency, and conditional completion versus oracle/no-probe controls across properties and objects.
- Contribution if successful: an embodied conditional test of semantic grounding to nonvisual physical properties. It is not selected because the local artifact path is absent.

## Candidate H — policy-RNG reliability as a closed-loop evaluation primitive

Assessment: `PLAUSIBLE`; selected fallback, but only after Candidate F closes or reaches a terminal decision.

- One-sentence thesis: stochastic action-generation seed is an unreported deployment variable whose closed-loop outcome distribution should be measured and controlled separately from environment-reset variability.
- Robotics problem and importance: identical scenes and commands can yield different diffusion/flow action samples, making a single reported rollout seed an unreliable estimate of a stochastic robot policy.
- What current evaluation misses: binary benchmark reports usually mix or freeze policy RNG; PhAIL models time-to-success distributions but not policy-noise identity; SDN exploits diffusion-noise selection but does not establish a paired LIBERO protocol separating policy RNG from physical reset variance.
- Closest primary Prior: SDN (arXiv:2606.14084), PhAIL (arXiv:2605.29710), Beyond Binary Success (arXiv:2603.13616), and historical action-ensemble/uncertainty work.
- Base and comparator ecosystem: stochastic X-VLA and SmolVLA are locally runnable; OpenVLA-OFT provides an architecture/control comparison where applicable. Fixed policy seed, randomized policy seed, repeated environment seed, and repeated joint seed are the controls.
- Exact repeatable condition: identical BDDL, init-state tensor, simulator seed, observation, instruction, and horizon, varying only the frozen model sampling seed at every policy query according to a logged seed schedule.
- Residual hypothesis: within-reset outcome variance across policy RNG is practically large, task-dependent, and not predicted by action smoothness alone.
- Headroom: selecting among legal candidate chunks using only current observations and action self-consistency could improve worst-tail reliability, but no selection method is authorized before the evaluation gap is verified.
- Legal inputs: normal policy observations/instruction plus candidate actions generated at the same decision point; reward, success, future observations, expert actions, and post-outcome seed choice are forbidden.
- Possible mechanism: a prespecified seed scheduler or candidate selector using action-distribution geometry, subject to direct comparison with SDN.
- Closed-loop, retention, and generalization paths: full official rollouts; average success cannot drop under standard conditions; test multiple tasks, resets, and two stochastic policy families.
- Strongest novelty objection: it may reduce to “run more random seeds,” while SDN already treats noise as controllable.
- Strongest experimental objection: X-VLA's reset method may make model sampling effectively deterministic or the variance may be too small.
- Strongest feasibility objection: multi-seed rollout counts grow quickly and full-precision comparator checkpoints may not fit.
- Archived overlap: intersects closed stochastic schedule, uncertainty, and chunk-selection routes; only a new evaluation result with distributional insight could survive.
- Resource envelope: cheapest falsifier is about 45 X-VLA episodes, under 4 GiB CUDA and roughly 10–20 minutes; a paper-scale matrix could require 10–20 serial GPU-hours and under 10 GiB artifacts.
- Official artifacts/licenses: same retained X-VLA, SmolVLA, OpenVLA-OFT, and LIBERO artifacts as Candidate F.
- Cheapest falsifier: three competent Goal tasks, three init states, five model-query seed schedules, with all simulator inputs fixed. Kill if success and trajectory distributions are essentially invariant or SDN's published mechanism fully occupies the residual.
- Required main table: variance decomposition by reset RNG and policy RNG, distributional success intervals, tail risk, trajectory dispersion, cross-backbone replication, and a direct SDN-aligned control.
- Contribution if successful: a paired protocol showing when policy sampling noise, rather than environment variability, dominates closed-loop robot reliability.

## Candidate I — serial low-memory VLA evaluation runtime

Assessment: `WEAK`; infrastructure only.

- One-sentence thesis: one-live-environment execution with resumable atomic manifests can reproduce official VLA evaluation on memory-constrained workstations.
- Robotics problem: current evaluators often pre-create environments or assume large RAM, limiting reproducibility.
- Missed space and Prior: vla-evaluation-harness already provides broad interoperability/sharding; PhAIL and Beyond Binary Success address rigorous evaluation; the scientific residual is engineering rather than robotics.
- Ecosystem/condition: reproduce identical task/reset/model seeds while varying only lifecycle and resume behavior; compare peak RAM, wall time, and score identity.
- Headroom/mechanism: lazy environment construction and transactional per-episode manifests. Legal inputs and closed-loop semantics are unchanged.
- Retention/generalization: bitwise task manifests and statistically identical success across X-VLA, SmolVLA, and OpenVLA-OFT, multiple suites, and forced interruption/restart.
- Main objections: no new robotics capability, direct overlap with existing harness infrastructure, and success would not by itself support a strong RA-L contribution.
- Archived overlap/resources/artifacts: this is already the retained Epoch 7 execution discipline; under 24 GiB host RAM and modest storage. It remains supporting software, not a paper fallback.
- Cheapest falsifier and required table: compare stock versus serial peak memory, wall time, and identical outcomes on a small panel; even a pass is insufficient for selection.
- Contribution if successful: a practical reproducibility runtime, not an independently selected RA-L thesis.

## Selection and execution order

- Closed primary: Candidate F, thesis id `latent_dynamics_attribution`, decision `NO_LEGAL_HEADROOM` before any policy rollout.
- Active fallback: Candidate H, thesis id `policy_rng_reliability`, clearly different stochastic-evaluation question. It is authorized for a focused closest-overlap and archived-route audit, followed by a frozen paperability contract only if that audit survives.
- Candidates G and I are not empirically authorized.
- No Ours design or training is authorized in Cycle 2.

Candidate F rotated because exact-init expert replay did not establish legal headroom across all three required families. Candidate H now receives only its paperability/overlap audit; empirical execution remains unauthorized until its own claim, prior boundary, controls, and kill conditions are frozen.
