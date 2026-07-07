# Next Topic Candidates V2

No topic is implemented here. This is a pre-screen only.

Recent literature context checked: VLA safety surveys and benchmarks, action-space design analyses, action-chunking/latency work, diffusion/action policy updates, and target/object guidance work. The gap must survive simple-baseline scrutiny before implementation.

Source pointers:
- `https://arxiv.org/html/2604.23775v1`
- `https://arxiv.org/html/2602.23408v2`
- `https://arxiv.org/html/2606.01865v1`
- `https://papers.nips.cc/paper_files/paper/2025/file/300ccb2187dedd4edcc07f7e76d8e553-Paper-Conference.pdf`
- `https://vla-survey.github.io/`

## Candidate 1: Phase-Locked Action Chunk Retiming

- task definition: recover execution when an otherwise valid expert/VLA action chunk is temporally misaligned by lag, dropped actions, or early/late gripper phase.
- latest-paper gap: recent action-chunking and real-time execution work emphasizes fast chunk execution and latency robustness, but a small baseline-first LIBERO diagnostic for state-progress-locked retiming under phase slippage remains a clear local gap.
- method novelty: use observed EEF/object/contact progress to retime the next action index, rather than scaling action magnitudes or replaying by wall-clock step.
- why not solved by trivial baseline: the failure is temporal ordering, not action magnitude or clipping; global scale and diagonal affine should not repair a delayed gripper/contact phase.
- strongest simple baseline that could kill it: fixed time shift, nearest-progress demo index, or gripper-only phase heuristic.
- first 48-hour executable test: exact-init expert replay, then inject one-step/two-step lag or action-drop faults and compare raw lag, fixed time shift, repeat-last, global scale, diagonal affine, and progress-locked retiming.
- exact kill criteria: kill if fixed time shift, nearest-progress demo, or gripper-only heuristic matches recovery; kill if no reward/progress gap appears within one task.
- expected assets: existing LIBERO HDF5 demos, exact-init replay runner, object/EEF state access.
- rollout/control metric within 48 hours: yes, replay/control metric.
- why RA-L-stable: clear fault model, fast replay table, direct baselines, low dependence on native VLA quality.
- why it might fail: nearest-progress demo indexing may be the real solution, or injected lag may not degrade reward.
- pre-screen status: best next candidate.

## Candidate 2: Contact-Event Gripper Phase Recovery

- task definition: recover failures caused by early/late gripper open-close timing during replay or policy chunks.
- latest-paper gap: action-space and chunking papers discuss action representation, but gripper event timing remains under-tested as a first-class executable failure mode.
- method novelty: trigger gripper phase from contact/object-distance events instead of fixed timestep.
- why not solved by trivial baseline: only valid if fixed gripper sign, thresholding, and simple phase shifts fail.
- strongest simple baseline that could kill it: gripper-only calibration or fixed gripper phase shift.
- first 48-hour executable test: replay expert with gripper timing offset and compare fixed phase shift, gripper sign/threshold baselines, and contact-event gating.
- exact kill criteria: kill if gripper-only or fixed phase shift recovers success; kill if timing perturbation does not degrade exact-init replay.
- expected assets: LIBERO HDF5 actions, exact-init replay, object/EEF distance keys.
- rollout/control metric within 48 hours: yes.
- why RA-L-stable: narrow, interpretable, robotics-relevant executable failure.
- why it might fail: too narrow for RA-L, or a trivial gripper baseline solves it.

## Candidate 3: Progress-Sentinel Chunk Abort

- task definition: detect when a replayed or VLA action chunk is moving away from required object progress and abort before compounding failure.
- latest-paper gap: VLA safety work catalogs risks and filters unsafe actions, but progress-sensitive chunk abort under manipulation state regressions is less directly isolated.
- method novelty: use short-horizon EEF/object progress certificates, not semantic labels, to stop bad chunks.
- why not solved by trivial baseline: must beat safety-only, clipping-only, and action-norm thresholds on false positives and recovery.
- strongest simple baseline that could kill it: safety-only or action-norm threshold.
- first 48-hour executable test: inject wrong-direction or delayed chunks during exact-init replay and measure object progress, false aborts, and recovery under safety/clipping/norm/progress-sentinel variants.
- exact kill criteria: kill if safety-only or norm threshold matches progress-sentinel recovery with equal false positives; kill if no recovery action can be defined without replay leakage.
- expected assets: LIBERO simulator, HDF5 expert replay, object/EEF metrics.
- rollout/control metric within 48 hours: yes.
- why RA-L-stable: safety/deployment framing with direct control metrics.
- why it might fail: resembles CSS-Shield if the progress signal collapses to generic safety intervention.

## Candidate 4: Multi-Demo Event-Tube Replay Selection

- task definition: choose among multiple demonstration subsegments based on current event state rather than nearest raw state or global action statistics.
- latest-paper gap: object/attention guidance and few-demo work exist, but a strict simple-baseline event-tube replay diagnostic could isolate whether event ordering adds value.
- method novelty: event-state tube membership over raw nearest-demo matching.
- why not solved by trivial baseline: only valid if nearest-demo, mean-action, and exact-init replay controls fail under mismatch.
- strongest simple baseline that could kill it: nearest-demo replay.
- first 48-hour executable test: use two to three local demos, perturb reset/default state, and compare nearest-demo, mean-action, global scale, and event-tube subsegment selection.
- exact kill criteria: kill if nearest-demo matches event-tube progress or success; kill if event labels require BDDL/eval leakage.
- expected assets: local LIBERO demo set, object/EEF states, replay runner.
- rollout/control metric within 48 hours: probably yes.
- why RA-L-stable: could become a baseline-first replay robustness table.
- why it might fail: nearest-demo is likely strong and may kill novelty.

## Candidate 5: Action-Space Adequacy Stress Benchmark

- task definition: determine which action representation or controller convention is minimally adequate for a manipulation task before training or rollout.
- latest-paper gap: recent action-space design work highlights representation choices, but a small LIBERO executable adequacy stress suite could be useful if it avoids becoming another calibration story.
- method novelty: predeclared stress probes that separate translation, rotation, gripper, timing, and scale adequacy under replay.
- why not solved by trivial baseline: weak unless it tests faults not solved by diagonal affine or global scale.
- strongest simple baseline that could kill it: diagonal affine and global scale.
- first 48-hour executable test: replay controlled perturbations of expert actions and report which action dimensions are necessary for reward/progress.
- exact kill criteria: kill as a method topic if diagonal/global calibration explains recovery; keep only as benchmark tooling if useful.
- expected assets: HDF5 demos, exact-init replay, action perturbation code.
- rollout/control metric within 48 hours: yes.
- why RA-L-stable: possible as a benchmark/audit paper, not as a method claim.
- why it might fail: ExecSpec and ResetSpec already showed calibration baselines are too strong.

## Recommendation

Recommended next topic: Candidate 1, Phase-Locked Action Chunk Retiming.

Why:
- fastest real metric using existing exact-init replay,
- least dependent on native VLA competence,
- attacks temporal phase mismatch rather than scale, clipping, or semantic target selection,
- has clear simple baselines that can kill it early,
- can produce a compact RA-L-style table: raw lag, fixed shift, repeat-last, global scale, diagonal affine, gripper-only phase, nearest-progress demo, and phase-locked retiming.

Do not implement it until the user explicitly starts the next state.
