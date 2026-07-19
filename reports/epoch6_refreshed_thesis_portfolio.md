# Epoch 6 Refreshed Thesis Portfolio

Selection date: 2026-07-19 KST
Evidence partition: literature, official-source, and historical discovery only.
Epoch 6 outcome access at selection: none.
Naming rule: descriptive thesis labels only; no method acronym.

The initial portfolio was exhausted by current-prior collisions. An independent
method refresh found no active-quality method card. An independent benchmark and
systems refresh produced two gate-only residuals; a separate adjudicator selected
the highest-ranked residual for a frozen problem gate while explicitly withholding
paper-candidate and method authorization.

## Selected for problem verification: schedule-invariant stochastic VLA evaluation

- One-sentence thesis: a fixed robot-evaluation episode manifest is not
  reproducible when batching or sharding changes which stochastic action sample
  is assigned to each episode, so policy randomness must be addressed by stable
  model/task/episode/policy-call identity rather than process-global draw order.
- Robotics importance: different sampled action chunks change the physical
  trajectory and can change which controller is judged competent or deployed;
  this is not only a logging discrepancy.
- Exact claim scope: simulator evaluation of stochastic VLA policies on one
  fixed software/hardware stack. No cross-hardware bitwise-determinism, physical-
  robot, safety, or universal benchmark claim.
- Competent runnable Base: the pinned X-VLA LIBERO checkpoint and official-style
  local runner. Historical X-VLA sweeps establish strong standard competence,
  but all old outcomes remain discovery evidence. SmolVLA is the planned second
  stochastic family after the problem gate; OpenVLA-OFT is a deterministic
  negative control.
- Closest Prior: vla-eval supplies episode sharding, batching, IDs, provenance,
  and broad benchmark/model integration. *What Are We Actually Benchmarking?*
  addresses closed-loop hardware nondeterminism. vLLM batch invariance is the
  closest non-robotics serving precedent.
- Strongest compatible runnable comparator: process-global seeding, fixed
  process-per-shard seeding, and serialized inference under the same policy,
  inputs, root seed, action semantics, and episode manifest.
- Artifact/fidelity: local official vla-eval revision
  `a7eb023a962456bb0b6be40aa4336c31b7ac4ce6` and X-VLA revision
  `6bc2513f5f1cbec715cc668b414392a6cae5c671` are pinned. vla-eval's X-VLA
  adapter calls official `generate_actions`; X-VLA starts generation with
  process-global `torch.randn`. The local X-VLA checkpoint revision is pinned in
  the resource inventory.
- Repeated problem plan: first run an outcome-suppressed action-hash/RMS gate on
  actual X-VLA with one fixed input tensor and reversed logical request order. Only
  after that passes, run a frozen 20-reset, four-task LIBERO panel under serial,
  prescribed four-lane, and reversed four-lane schedules.
- Recoverable headroom: the proposed episode-addressed contract must later make
  per-episode noise and action traces invariant to request order while preserving
  distinct samples across episode keys and the model's marginal sampling law.
- Legal inputs: model ID/revision, task ID, episode/reset identity, policy-call
  index, root seed, and ordinary policy observations. No outcome, reward, future
  observation, success, or simulator-private state enters policy sampling.
- Central novelty opportunity: make stochastic robot-policy sampling a named,
  episode-addressed part of the evaluation object and show that this removes a
  scheduling intervention without serializing evaluation.
- Simplest equivalent: serialize all inference under one global seed. If this is
  sufficient at acceptable throughput or schedule changes do not alter real
  conclusions, the thesis is infrastructure rather than research.
- Strongest objections: counter-based randomness is conceptually simple and has
  non-robotics precedent; the effect may not change success; vla-eval may already
  make aggregate results stable; exact equality is stack-local; and a two-model,
  simulation-only study may be too narrow.
- Closure overlap: distinct from the closed asynchronous inference-delay axis and
  from the rejected outcome-separated model-selection thesis. It changes random-
  draw assignment under identical synchronous policy semantics; it does not
  adapt the controller, select checkpoints, or claim a latency method.
- Data/supervision: no training labels. Fixed LIBERO initial states and ordinary
  observations are available locally; reward/success is withheld until the
  source-level action gate passes.
- Resource estimate: no download is required for X-VLA Stage 0. One model is
  resident at a time. A current actual-path RAM/VRAM/swap smoke is mandatory
  before model inference. The later 40-episode two-schedule problem panel must
  be checkpointed and remain under the 12-hour heavy-job boundary.
- First cheap falsification: the frozen action-level schedule-dependence protocol
  in `reports/epoch6_schedule_invariant_evaluation/problem_verification_protocol.json`.
- Closed-loop/generalization: four tasks across the standard LIBERO suites for
  problem verification; after a valid problem only, add SmolVLA, OpenVLA-OFT,
  a second suite/condition, actual harness sharding, and throughput overhead.
- Six-page story: hidden schedule intervention; episode-addressed randomness
  contract; source/equation definition; multi-policy paired trajectory evidence;
  decision stability and throughput; limitations.
- Why positive results could still fail: scheduling may change many actions but
  not success or rankings, the keyed contract may reduce to a small patch, or the
  effect may be specific to X-VLA/LIBERO and one execution stack.

Selection decision: `SELECT_FOR_PROBLEM_GATE`. This is not method-design,
Stage-A, confirmatory, or paper authorization.

## Backup 1: non-gripper object-environment contact-transition topology

- Thesis: training-only supervision of changes in object-environment contact
  relations may improve contact-boundary arm actions beyond binary contact,
  gripper events, and temporal stage, with the auxiliary head removed at
  deployment.
- Importance/scope: contact changes constrain manipulation; simulation-only
  LIBERO/MuJoCo, no force/tactile magnitude or real-world claim.
- Base/Prior/comparator: competent local SmolVLA; FD-VLA, HapticVLA, CALAMARI,
  TacCoRL, StaKe, and GAP are the closest priors; binary contact, stage/gripper,
  phase, shuffled labels, action history, and gripper history are mandatory
  controls.
- Artifact/data: exact state restoration and MuJoCo geom-pair contacts are
  statically supported. HDF5 stores flattened states but no contact labels;
  replay extraction and robot/gripper geom exclusion must be validated.
- Legal inputs: RGB, language, proprioception, and action history only. Contact
  topology is training supervision and unavailable at inference.
- Cheap falsification/headroom: fresh task-held-out contact prevalence,
  predictability beyond controls, and oracle arm-action headroom at non-gripper
  transition boundaries. Kill for class collapse, confounding, unobservability,
  or no residual beyond binary/stage controls.
- Resources/path: simulator replay plus a small probe, then one-model SmolVLA
  training only after problem authorization; feasible in principle under the
  local envelope.
- Main reject reason: the central sensor-free privileged-contact claim may still
  be judged an incremental label variant of FD-VLA/HapticVLA and stage work even
  after positive simulation results.
- Closure distance: excludes the closed TCA/contact-map, ContactTube,
  ContactSet, wrist-dropout, and async-delay formulations.
- Six-page viability: only if topology yields repeated multi-task closed-loop
  gains that binary contact and stage/gripper supervision cannot explain.

Status: `BACKUP_GATE_ONLY`; no label extraction or experiment authorized while
the selected thesis has a valid next step.

## Backup 2: persistent rather than first-hit task success

- Thesis: task success should remain true through a policy-independent controller
  hold, not merely occur on the first simulator step when an instantaneous
  predicate fires.
- Importance/scope: transient placement, stacking, or articulation can overstate
  stable robot completion; LIBERO simulation only.
- Base/Prior/comparator: official demonstrations for the replay gate, then
  X-VLA/SmolVLA/OpenVLA-OFT; closest priors are vla-eval's termination audit,
  SafeVLA-Bench's temporal safety predicates, and PhAIL's time-to-success.
- Artifact/data: official LIBERO rewards/done use instantaneous `_check_success`;
  vla-eval terminates on this native signal. Exact replay is locally available.
- Legal inputs: success evaluation occurs after the policy stops; no privileged
  signal changes policy inference.
- Cheap falsification: stratified successful expert replays followed by frozen
  controller hold, last-action repeat, and gripper-preserving controls across
  placement, stacking, articulation, switching, and held-object mechanisms.
- Headroom/kill: require repeated native-versus-persistent disagreement and a
  changed multi-policy comparison. Kill if persistence holds, one malformed
  predicate explains the effect, or no policy conclusion changes.
- Resources/path: simulator-only and low download cost, but the post-success hold
  semantics require task-family-specific validation before policy evaluation.
- Main reject reason: it may reduce to a benchmark bugfix or overlap SafeVLA-
  Bench rather than constitute a new evaluation method.
- Closure distance: it does not reopen delay, recovery, or selection-bias axes.
- Six-page viability: only with broad task-mechanism coverage, multiple policies,
  and actionable ranking/decision changes.

Status: `BACKUP_GATE_ONLY`; no replay outcome access is authorized now.

## Hard-filtered during this refresh

- Visibility-deletion memory: incremental against Present-but-Not-Remembered,
  Embodied-SlotSSM/LIBERO-Mem, NativeMEM, TFP, AURA, SOMA, and LIBERO-Occ; local
  memory/retrieval routes also repeatedly failed.
- Generic uncertainty/failure detection, backdoor defense, proprioceptive early
  fusion, future-kinematic supervision, cross-embodiment action priors, and
  active perception: current direct priors or unavailable interfaces remove the
  claimed residual.

Exactly one thesis is active, and only at its problem-verification stage.
