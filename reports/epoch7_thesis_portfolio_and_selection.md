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

Assessment: `CLOSED_TOO_OVERLAPPING_OR_TRIVIAL`; the prespecified fallback failed its paperability audit before outcomes.

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

Closure: Candidate H changes the archived Epoch 6 schedule-induced noise reassignment into a direct policy-seed factor, but retains the same stochastic-policy problem axis, X-VLA/LIBERO stack, noise intervention, trajectory/success endpoints, and evaluation archetype. Epoch 6 already established action-level schedule dependence and remains scientifically unresolved at closed loop. A reset-versus-policy variance decomposition is a standard analysis rather than a second material scientific change, while any candidate selector collides directly with SDN. The decision is `TOO_OVERLAPPING_OR_TRIVIAL`; no policy was loaded and no rollout was run.

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
- Closed fallback: Candidate H, thesis id `policy_rng_reliability`, decision `TOO_OVERLAPPING_OR_TRIVIAL` at the paperability audit with zero rollouts.
- Candidates G and I are not empirically authorized.
- No Ours design or training is authorized in Cycle 2.

Candidate F rotated because exact-init expert replay did not establish legal headroom across all three required families. Candidate H then closed because it supplied fewer than two material changes relative to the unresolved Epoch 6 stochastic-schedule route and collided with SDN at the method boundary. A fresh portfolio rotation is required; Candidates G and I remain blocked/weak and are not promoted.

## Cycle 3 fresh rotation

Rotation 3 reconsiders only scientifically unadjudicated routes and adds current July 2026 overlap. SafeVLA-Bench already measures successful-but-unsafe rollouts with task-aware temporal constraints; vla-eval exposes protocol ambiguity but supports only native success; VLA-SCT uses a visual-memory termination detector; Pre-VLA and agentic RL already occupy generic action verification and recovery. The surviving boundary is narrower: whether a native first-hit goal predicate certifies a state that remains achieved when the policy stops and the controller neutrally holds.

### Candidate J - stability-qualified task completion

Assessment: `CLOSED_NO_REPEATABLE_GAP`; the frozen expert-replay gate found an `On`-placement-specific effect but failed cross-mechanism diversity.

- One-sentence thesis: a manipulation rollout should count as complete only when its native goal remains satisfied during a short policy-independent neutral dwell, because first-hit success can certify transient states and distort VLA comparisons.
- Robotics problem and importance: deployment requires the placed object, grasp, mechanism, or multi-step goal to remain achieved after motion stops; a one-frame predicate hit is not necessarily durable completion.
- What current evaluation misses: LIBERO runners terminate on native first-hit success; vla-eval standardizes that host metric; PhAIL models time-to-first-success; SafeVLA-Bench adds safety constraints while preserving native success; VLA-SCT detects when to stop but does not validate that the achieved native goal persists after stopping.
- Closest primary Prior: vla-eval (arXiv:2603.13966), SafeVLA-Bench (arXiv:2606.00773), VLA-SCT (arXiv:2602.01811), PhAIL (arXiv:2605.29710), and Beyond Binary Success (arXiv:2603.13616).
- Base/comparator ecosystem: official demonstrations first establish predicate behavior and recoverability; retained X-VLA, SmolVLA, and OpenVLA-OFT INT4 are conditional policy comparators. Native first-hit success, immediate neutral dwell, full expert suffix plus dwell, and last-action-repeat are controls.
- Exact condition: on the same exact-init trajectory, identify the first native success, replace future actions with 30 zero-motion controller commands that preserve the last gripper command, and evaluate the unchanged official predicate after every hold step. In a separate branch, execute the unused demonstration suffix before the same hold.
- Residual hypothesis: native first-hit success fails the immediate dwell on multiple tasks and mechanisms, while the frozen expert suffix recovers stable completion, proving that the first-hit state is premature rather than the task being unsatisfiable.
- Headroom: a selected standard-success demonstration must become persistent after its unused suffix and neutral dwell. Expert actions are controls, not policy successes.
- Legal inputs: the policy is not modified; the evaluator may use native success only to switch into the post-policy hold. The neutral action uses zero pose delta and the last executed gripper sign. Rewards, future observations, and expert actions never enter policy inference.
- Possible mechanism: none before problem verification. A later benchmark paper may include a simple consecutive-success/dwell endpoint; a learned termination method would have to exceed VLA-SCT and is not assumed.
- Closed-loop path: official policy rollouts report both native and persistent success, persistence curves, and native-success/persistent-failure counts.
- Retention/generalization: native success remains reported unchanged; persistent completion is additional. Paper level requires at least two competent policy families, Goal and Long or a second compatible suite, and placement, containment/insertion, push, and articulation mechanisms.
- Main objections: a neutral dwell may be a benchmark bugfix; the action may be controller-specific; SafeVLA-Bench's stable-object constraints may already explain the effect; rankings may not change.
- Archived overlap: the route was an unexecuted Epoch 6 backup blocked by the then-shared simulator resource rule. It does not reopen schedule, delay, recovery, or outcome-selection axes. The current mission explicitly permits serial one-environment replay and treats small pagefile-allocation jitter as nonfatal.
- Resources/artifacts: no new download, one 64x64 environment at a time, CPU-only expert gate, below 8 GiB expected host increment, negligible storage, and roughly 10-30 minutes. LIBERO source/data are Apache-2.0/CC BY 4.0 as previously audited.
- Cheapest falsifier: all ten LIBERO-Goal tasks, lowest-index standard-success demo selected without persistence outcomes, then paired immediate-hold and expert-suffix-hold branches. Kill if fewer than three tasks/two mechanisms show native-to-persistent disagreement or fewer than two tasks/two mechanisms show suffix-recoverable headroom.
- Required main table: task/mechanism, native first-hit, immediate-dwell persistence curve, suffix-dwell recovery, policy-native and policy-persistent success, paired disagreements, confidence intervals, ranking changes, and hold overhead.
- Contribution if successful: a reproducible completion endpoint showing when native first-hit success overstates durable manipulation and changes conclusions across policies.

Closure: all ten standard-selected demonstrations reached native success with valid deterministic execution. Immediate neutral dwell failed for task IDs 2, 4, and 8, and each was recovered by the unused expert suffix. However, all three are `On` placements; one predicate family explained 100% of disagreements and both problem and recoverable headroom spanned only one mechanism. The frozen two-mechanism and single-predicate-explanation gates failed, so the decision is `NO_REPEATABLE_GAP`. No policy was loaded, no policy result exists, and no task/suite expansion is legal.

### Candidate K - typed non-gripper contact-transition supervision

Assessment: `PLAUSIBLE`; selected fallback if Candidate J closes.

- Thesis: training-only supervision of typed non-robot contact-edge births and deaths may improve contact-boundary arm actions beyond binary contact, gripper events, phase, and action history without privileged deployment inputs.
- Importance/gap: manipulation is defined by object-environment relation changes, but common VLA losses and coarse stage labels may not emphasize lift, placement, insertion, and release boundaries.
- Prior/ecosystem: FD-VLA, HapticVLA, CALAMARI, TacCoRL, StaKe, GAP, binary contact, stage/gripper, history, and shuffled-label controls; retained SmolVLA is the trainable Base.
- Exact condition/headroom: replay frozen LIBERO states, remove all robot/gripper geoms, debounce typed non-static contact-edge births/deaths, then require task-held-out visual predictability and arm-action oracle information beyond all controls before any VLA training.
- Legal inputs/path: MuJoCo contacts are training-only supervision; inference is ordinary RGB/language/proprioception/history. Official closed-loop success, Base retention, task-held-out tasks, and multiple contact mechanisms are mandatory.
- Objections/overlap: the label may be an incremental force/tactile or phase variant; the historical exact route is scientifically unadjudicated but resource-blocked. Reconsideration is allowed because the blocker was operational and the current serial simulator path plus corrected pagefile policy materially changes feasibility, not the scientific formulation.
- Resources: state replay and small probes first; later one SmolVLA plus a small auxiliary head. No new checkpoint. Expected under 16 GiB VRAM, 20 GiB host RAM, 5 GiB artifacts, and 12 GPU-hours.
- Cheapest falsifier: execute the already frozen Stage 0A label gate under a documented outcome-free resource amendment, then Stage 0B only if labels are valid.
- Contribution if successful: evidence that typed scene-contact transitions provide task-held-out action supervision beyond generic phase/contact signals and improve official closed-loop manipulation.

### Candidate L - false-premise selective non-execution

Assessment: `BLOCKED`.

- Thesis: a VLA should detect absent, impossible, or contradictory instruction premises and avoid unsafe plausible actions while retaining true-premise task success.
- Importance/path: false premises are deployment-relevant and directly language-grounded; evaluation would pair true and false instructions in matched scenes and measure action suppression, explanation, and ordinary success.
- Prior and veto: DoWhat?, IVA, IGAR/ICBench, LIBERO-CF, ProGAL-VLA, and current safety benchmarks already occupy detection, rejection, correction, and counterfactual language-action coupling. The retained checkpoints do not expose a trained clarification/refusal channel. A no-op threshold or prompt-only wrapper would be weak and duplicate current work.
- Artifacts/resources: simulator construction is possible, but no competent runnable Base/comparator with the required accept/clarify/refuse interface exists locally. No empirical execution is authorized.
- Contribution if successful: calibrated selective execution under false premises; presently a direct-collision/artifact hard veto.

### Candidate M - serial low-memory evaluation runtime

Assessment: `WEAK`; systems support only.

- Thesis: transactional one-environment execution can reproduce official VLA scores within workstation memory limits while supporting exact resume.
- Prior/objection: vla-eval already provides a unified, versioned, massively parallel harness; serial lifecycle and atomic manifests are useful infrastructure but not a new robotics result.
- Path/resources: already locally implemented and used; a throughput/RAM/score-identity table is feasible, but success would not satisfy a RA-L contribution sentence without a distinct scientific endpoint.

## Cycle 3 selection

- Closed primary: Candidate J, thesis id `persistent_completion`, decision `NO_REPEATABLE_GAP` after a valid ten-task pre-policy gate.
- Active fallback: Candidate K, thesis id `contact_transition_topology`, method archetype, authorized only for a source/fidelity audit and an outcome-free resource-rule amendment before resuming its already frozen Stage 0A label gate.
- Candidates L and M are not empirically authorized.
- No Ours, policy rollout, method training, or confirmatory outcome is authorized by this selection.

Candidate K's resource amendment is now frozen at SHA-256
`7CCDCE5D9AA0B24C356AF873D0481AF76312D3C7FCF6871C4CA80FD6621ACFEB`.
The scientific protocol remains unchanged at SHA-256
`7FA28AAEEAC9886F36DD5CCD059CA7AC4CD65B21FABFBBCA4AFFA53B0A256240`.
Only a passing outcome-free actual-path resource smoke may now authorize the
already-frozen Stage 0A label gate.

Stage 0A subsequently returned `CONTACT_TOPOLOGY_LABEL_GATE_GO` with all 18
frozen gates passing. This establishes a deterministic, nontrivial typed
contact-transition signal but not visual predictability or action headroom.
Only the already-frozen Stage 0B probes are now authorized; Candidate K remains
premethod and no Ours, VLA training, policy rollout, or paper work is legal.
