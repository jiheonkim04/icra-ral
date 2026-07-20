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

Stage 0B then validly closed Candidate K as `STAGE0_TRIVIAL_EQUIVALENCE`.
Visual any-transition and typed-bin prediction were both worse than the causal
nonvisual control, and typed topology slightly worsened aggregate held-out arm
NRMSE beyond the strongest binary-contact control. The exact formulation is
closed with zero VLA training and zero policy rollout. Cycle 3 is exhausted;
a fresh portfolio rotation is required rather than a weaker contact-label
rescue.

## Cycle 4 exhaustive refresh

Cycle 4 is the required post-primary/post-fallback refresh. It combines a
current primary-source scan through 2026-07-20 with the complete 95-route local
history, all three earlier Epoch 7 cycles, official artifact availability, and
the unchanged workstation resource envelope. The five candidates below are
materially different at the paper-thesis level; none is promoted merely to
satisfy a quota.

### Candidate N - consequence-conditioned deployment adaptation

Assessment: `BLOCKED`.

- One-sentence thesis: use realized observation-action-consequence triplets to adapt a VLA online to hidden camera, calibration, or actuation shifts while preserving standard behavior.
- Robotics problem and importance: the same nominal command can produce different observed motion after deployment, so a reactive policy can be systematically miscalibrated.
- What is missed: a compact frozen-policy implementation would be cheaper than retraining an in-context VLA, but that efficiency difference is not by itself a new scientific problem.
- Closest primary Prior: [Reflective VLA](https://arxiv.org/abs/2606.25215), which directly conditions on structured observation-action-consequence triplets and reports LIBERO-Plus gains over a matched history-only control.
- Base/comparator ecosystem: retained SmolVLA/X-VLA could support a small adapter, but Reflective VLA is the scientific Prior and the local `RAC-VLA` route is the already-executed frozen-policy comparator.
- Exact condition and residual: paired deployment action-channel or camera-calibration shifts, with identical tasks/resets; Ours would need to beat both reactive Base and action/history-only controls using only realized past observations and executed actions.
- Headroom source: Reflective VLA's published cross-environment effect is external headroom; no new local headroom remains after RAC.
- Legal inference inputs: past observations, past executed actions, current observation, instruction, and proprioception; no reward, success, simulator parameters, or future observations.
- Possible mechanism: a bounded consequence-conditioned calibration state around a frozen policy.
- Closed-loop/retention/generalization path: paired official rollouts under standard and held-out calibration/camera shifts, with standard retention and a second policy family.
- Strongest novelty objection: this is a resource-reduced Reflective VLA extension, not a new mechanism class.
- Strongest experimental objection: the small adapter may only learn a static affine correction.
- Strongest feasibility objection: no official Reflective VLA code/checkpoint is linked by the paper or project page, so a fair official reproduction is unavailable locally.
- Archived overlap: local RAC already tested the same frozen-policy action-consequence premise at Stage B; full/Base/proxy were each `1/40`, while the ablation and inverse control were `2/40`. Reopening would change only architecture scale, not two scientific dimensions.
- Resources/artifacts: a new adapter would fit locally, but the direct Prior artifact is absent and the closest local formulation is scientifically closed. Reflective VLA's project page exposes no code or checkpoint link as of the audit.
- Cheapest falsifier: already executed by RAC's 40-episode Stage B; no new rollout is warranted.
- Required main table: shifted and standard success for Base, Reflective Prior, Full, history-only, and affine controls across two policy families.
- Contribution if successful: a low-compute consequence-conditioned deployment adapter. It is blocked because current Prior and local evidence already answer the proposed formulation.

### Candidate O - temporal-integrity correction for robot demonstrations

Assessment: `WEAK`.

- One-sentence thesis: detect and correct observation-action timestamp misalignment in robot demonstrations before VLA training.
- Robotics problem and importance: sensor/control desynchronization can attach actions to the wrong scene state and corrupt imitation supervision.
- What is missed: existing VLA pipelines mention temporal alignment and execution-time staleness, but a public, task-level audit of naturally occurring timestamp error could still be useful on a real heterogeneous corpus.
- Closest primary Prior: [Green-VLA](https://arxiv.org/abs/2602.00919) includes temporal alignment and quality filtering over 3,000 hours; [DEFLECT](https://arxiv.org/abs/2605.19294) learns from fresh/stale pairs for asynchronous execution; the June 2026 UR5 study identifies temporal alignment as a deployment variable.
- Base/comparator ecosystem: SmolVLA is trainable, but the retained LIBERO files are simulator-generated and index aligned rather than independently timestamped.
- Exact condition and residual: estimate an unknown per-modality time offset from a raw timestamped demonstration, correct it without task outcomes, and improve held-out official closed-loop success over uncorrected and fixed-shift controls.
- Headroom source: a naturally occurring, independently measured timestamp mismatch must precede method design; synthetic frame shifts do not establish the deployment problem.
- Legal inference inputs: raw sensor/control timestamps and past observations/actions during preprocessing; no task outcomes or confirmatory identities.
- Possible mechanism: differentiable offset estimation from action-conditioned visual/proprioceptive change, followed by timestamp-preserving resampling.
- Closed-loop/retention/generalization path: train matched policies on raw versus corrected data, retain clean-data performance, and generalize across two sensors or collection systems.
- Strongest novelty objection: temporal alignment is already a named part of large VLA data pipelines and stale/fresh correction is directly studied by DEFLECT.
- Strongest experimental objection: an artificial shift in synchronized simulation would measure robustness to injected corruption, not a real data problem.
- Strongest feasibility objection: one inspected official LIBERO Goal HDF5 file had equal-length indexed observation/action arrays and no timestamp- or clock-named dataset; no local raw asynchronous corpus or physical collection path exists.
- Archived overlap: delay, retiming, DICD, and A2C2 routes are closed locally; the only material change would be moving the delay to training data, without a verified real-data premise.
- Resources/artifacts: synthetic tests fit easily; a defensible study needs raw independently timestamped robot data and likely a collection-specific reference, neither retained nor credential-free locally.
- Cheapest falsifier: the outcome-free HDF5 interface audit already found no independent timestamps; do not fabricate a mismatch.
- Required main table: measured native offset distribution, correction accuracy, matched training success, clean retention, and cross-collector generalization.
- Contribution if successful: a data-integrity method linking measured synchronization error to robot success. The current local path cannot establish the premise, so success on synthetic shifts would remain weak for RA-L.

### Candidate P - executable-policy integrity and architecture-aware monitoring

Assessment: `BLOCKED`.

- One-sentence thesis: certify the complete executable VLA specification and choose architecture-matched black-box action monitors before controller execution.
- Robotics problem and importance: identical weights can send different physical actions under different normalizers/controllers, and naive motor bounds do not predict all policy failures.
- What is missed: combining static specification integrity with dynamic monitoring could be useful software, but both scientific components now have direct primary precedents.
- Closest primary Prior: [Same Weights, Different Robot](https://arxiv.org/abs/2606.03724) formalizes the executable-policy object and an action-normalization certificate; [How VLAs Fail Differently](https://arxiv.org/abs/2605.28726) supplies SafeContract and architecture-specific conformal action monitoring.
- Base/comparator ecosystem: X-VLA, SmolVLA, OpenVLA-OFT, the local ExecSpec replay suite, and the official Apache-2.0 `vla-edge` repository at remote HEAD `fa445837b4f6214cd2bbeff8d96f79aac1d724f0`.
- Exact condition and residual: matched checkpoint/observation with a controlled executable-spec mismatch or naturally failing action trace; a joint system must detect the mismatch/failure before execution and preserve task success.
- Headroom source: the Prior papers already show metadata replay collapse and architecture-specific monitoring AUROC; no unoccupied local residual is identified.
- Legal inference inputs: checkpoint/config hashes and current/past action vectors; no reward, success, future state, or simulator privilege at runtime.
- Possible mechanism: a signed executable-spec manifest plus a calibrated monitor selected by action-decoder family.
- Closed-loop/retention/generalization path: official task success with zero silent mismatch and bounded false interventions across discrete and continuous policies.
- Strongest novelty objection: this is a direct union of ExecSpec and SafeContract rather than a new robotics insight.
- Strongest experimental objection: monitoring may correlate with failure without improving success or safety.
- Strongest feasibility objection: a strong multi-architecture study needs policy families and failure datasets beyond the one-backbone local discovery path.
- Archived overlap: local ExecSpec-Repair achieved `17/19`, exactly tying diagonal affine calibration at `17/19`; TL-ChunkRepair and multiple action filters also failed to translate cleaner actions into success.
- Resources/artifacts: static checks fit; SafeContract code is public under Apache-2.0, but a successful implementation would still duplicate current work and prior local systems evidence.
- Cheapest falsifier: the existing 19-replay baseline-dominance audit already defeats a specialized repair claim.
- Required main table: mismatch detection, failure AUROC, false-intervention rate, official success, latency, and cross-architecture calibration.
- Contribution if successful: integrated executable-policy assurance. It is blocked by direct current duplication and the absence of a new causal residual.

### Candidate Q - action-chunk continuity and speed-aware commitment

Assessment: `BLOCKED`.

- One-sentence thesis: adapt chunk commitment and execution speed to keep flow-policy trajectories smooth without sacrificing task success.
- Robotics problem and importance: independently sampled chunks can disagree at boundaries, while fixed-speed execution is inefficient away from contact.
- What is missed: a joint scheduler could combine continuity and speed, but that combination is an aggregation of occupied mechanisms rather than a verified new problem.
- Closest primary Prior: [Adaptive Action Chunking](https://arxiv.org/abs/2604.04161), [SEAM](https://arxiv.org/abs/2607.04609), [TempoVLA](https://arxiv.org/abs/2606.06491), and DEFLECT.
- Base/comparator ecosystem: trainable SmolVLA and retained X-VLA; direct local controls include fixed queue, direct chunk index, short requery, AAC proxy, DICD, and EAC.
- Exact condition and residual: identical reset/policy RNG under a frozen boundary or speed condition; Ours must reduce discontinuity and improve official success beyond SEAM/AAC/Tempo-aligned controls.
- Headroom source: must be a repeated success gap, not jerk alone. No such independent local residual remains.
- Legal inference inputs: current observation, current and previous chunks/actions, and a deployment-visible speed request; no rewards or future observations.
- Possible mechanism: analytic tail-consistency steering plus calibrated commitment length.
- Closed-loop/retention/generalization path: paired official success, boundary jerk, latency, and standard retention across contact and transit tasks and two flow policies.
- Strongest novelty objection: SEAM already uses the unexecuted previous tail as an analytic consistency reference, while TempoVLA and AAC occupy speed and commitment.
- Strongest experimental objection: smoother actions need not improve task success.
- Strongest feasibility objection: an official pi0.5 comparator/checkpoint identity is not retained locally.
- Archived overlap: DICD full scored `1/10` versus direct chunk-index `2/10`; EAC full scored `29/40` versus Base/AAC/ablation `30/40`; the corrected local async-delay axis had no repeatable gap.
- Resources/artifacts: a SmolVLA port fits, but the closest official methods are not all locally reproducible and the exact local family is already closed.
- Cheapest falsifier: existing DICD/EAC Stage A/B evidence and the corrected A2C2 problem audit already supply it.
- Required main table: official success, paired discordance, jerk/discontinuity, policy calls, latency, and standard retention versus fixed, AAC, SEAM, and speed controls.
- Contribution if successful: success-preserving smooth and speed-aware flow control. It is prohibited as a near-duplicate rescue without a new independently verified residual.

### Candidate R - schedule-invariant stochastic VLA evaluation

Assessment: `BLOCKED`; scientifically the strongest unresolved candidate, but not locally executable under the frozen intervention.

- One-sentence thesis: official VLA success should be invariant to semantically irrelevant cross-episode batching/arrival schedules, or the evaluation must expose schedule as part of policy identity.
- Robotics problem and importance: a stochastic batched policy can assign noise to episodes according to queue arrival, changing actions for an identical task/reset and undermining reproducibility.
- What is missed: action-level dependence is already locally established, but its effect on official task success remains unknown.
- Closest primary Prior: stochastic-noise selection (SDN), distributional evaluation (PhAIL and Beyond Binary Success), and schedule/reproducibility systems; none substitutes for the archived frozen four-shard intervention.
- Base/comparator ecosystem: official X-VLA-Libero, the frozen schedule-preserving and schedule-perturbed implementations, and the unchanged 40-episode official LIBERO panel.
- Exact condition and residual: four simultaneously live official environments feed one shared model under the two frozen arrival schedules, with identical tasks, resets, model, and call-addressed randomness except the claim-defining scheduling intervention.
- Headroom source: frozen Stage 0 returned `ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO`; only closed-loop task-success headroom remains unobserved.
- Legal inference inputs: ordinary policy observations and instruction. Reward/success never affect noise assignment, queue order, or model inference.
- Possible mechanism: none before the evaluation result; this is a benchmark/evaluation thesis, not an authorized schedule-invariance method.
- Closed-loop/retention/generalization path: execute the frozen 40-episode paired panel, then expand only under a new preregistered contract if task outcomes show a meaningful repeated effect.
- Strongest novelty objection: it may be interpreted as a reproducibility bug rather than a robotics result.
- Strongest experimental objection: action differences may not change success.
- Strongest feasibility objection: the intervention requires four simultaneous environments; they alone drove the current 24.87 GB host to 85.16% before the model loaded.
- Archived overlap: this is the exact unresolved Epoch 6 route, reconsidered only now after the required complete fresh audit. It is not renamed or altered.
- Resources/artifacts: conservative projected full-path peak is 28.66 GB used; the frozen 82% ceiling requires 34.96 GB total, and 32 GiB is not certifiable. A standard 48 GB-or-larger host is the smallest defensible tier. Zero WSL swap and no CPU/disk model offload remain mandatory.
- Cheapest falsifier: the existing exact four-environment outcome-free resource smoke already proves the current host cannot launch the model safely; serial/two-shard execution would change the intervention and is forbidden.
- Required main table: per-task paired official success under the two schedules, discordant pairs and exact interval, action divergence, standard validity, and resource telemetry.
- Contribution if successful: evidence that execution schedule is part of the effective stochastic robot policy and a reproducible evaluation protocol that removes or reports this hidden variable.

## Cycle 4 selection and terminal adjudication

No Cycle 4 candidate clears both paperability and local feasibility, so selecting
an active primary/fallback would violate the hard vetoes. Candidate R is the
strongest scientifically unresolved route after normal competition, but it is
not selected for local execution: its exact intervention is externally blocked
by host memory. Candidates N, P, and Q are directly occupied and answered by
valid local controls; Candidate O lacks a genuine local problem instance.

Across four Epoch 7 cycles, method, benchmark/evaluation, and systems
archetypes have now been evaluated. Together with the inherited 95-route audit,
the result is `HARD_EXTERNAL_BLOCKER_REQUIRES_USER`, not a paper GO and not a
claim that the field has no open problems. The smallest scientific continuation
is the unchanged Candidate R panel on a clean scientifically equivalent host
with at least 48 GB physical RAM. Serial or two-shard substitution is not an
equivalent repair.
