# Epoch 6 Bounded Thesis Portfolio

Initial ranking date: 2026-07-19 KST
Current status: initial portfolio exhausted at the thesis hard filter; no active
thesis was selected and no Epoch 6 empirical outcomes were exposed. A fresh
opportunity refresh is in progress.
Outcome access: no Epoch 6 empirical outcomes have been run or exposed.
Naming rule: these are descriptive thesis labels, not method names.

Hard filters were applied before scores. The controller-dynamics candidate was
removed because APEX is a near-exact collision. Target/action maps, wrist
dropout, asynchronous delay, generic splines, and unbounded natural-reset
mining were removed by current literature or the closure registry.

## Initial rank 1 — rejected: counterfactual preservation of task-causal low-motion actions

Post-ranking official-code audit: VLA-Arena revision
`babe582ebffc82b979b77964a7e56417d02f63a4` already uses exact simulator
outcomes to progressively retain 4/8/12/16 post-gripper no-op actions and keeps
the first successful trajectory. The proposed segment-level causal label shares
the same central supervision, intervention, and adjudicator. It is classified
`TOO_OVERLAPPING_OR_TRIVIAL`, not selected, and received no experiment.

- One-sentence robotics thesis: robot demonstrations contain low-motion actions
  that are physically necessary for later task success, so magnitude-only
  filtering discards useful supervision and should be replaced by
  simulator-measured causal preservation.
- Robotics importance: small holds, settling commands, and contact-maintaining
  actions change the physical state reached by later actions even when their
  instantaneous magnitude is near zero; treating them as empty data can alter
  closed-loop manipulation rather than merely an offline metric.
- Exact claim scope: LIBERO-style continuous-control demonstrations with exact
  initial-state replay and simulator access; simulation-bounded evidence only.
  No physical-robot, safety, or universal no-op claim.
- Competent runnable Base: the pinned local OpenVLA-OFT policy has historical
  standard closed-loop competence (`11/15` on the corrected A2C2 Base panel)
  and serves as the high-credibility frozen Base. A smaller trainable official
  Base must separately pass competence and resource gates before a training
  comparison is authorized.
- Closest known Prior: VLA-Arena's fixed-N preservation around transitions;
  its official repository is public and must be pinned/license-checked before
  reuse. OpenVLA-OFT's complete no-op filtering is the opposed preprocessing
  baseline. FrameSkip is the closest heuristic temporal-importance method.
- Strongest compatible runnable comparator path: implement the explicitly
  documented fixed-N gripper-neighborhood rule on the same frozen local
  demonstrations, plus retain-all and remove-all controls. This needs no
  unsupported model port.
- Official artifact/fidelity status: local LIBERO code/data and exact-init
  replay are already validated and pinned in the resource inventory. The local
  OpenVLA-OFT code/checkpoint are pinned but its 7B training path exceeds VRAM.
  VLA-Arena and a smaller official Base require artifact/license/resource audit.
- Repeated problem-verification plan: before any Ours design, freeze disjoint
  discovery and validation task/demo manifests; replay each exact demonstration
  and matched counterfactuals that replace a maximal near-zero segment with a
  controller-valid neutral action while leaving all other actions unchanged.
  Repeat across tasks/demos and separately report gripper-adjacent versus
  non-gripper segments.
- Recoverable headroom diagnostic: a causal segment is eligible only when the
  untouched replay succeeds, the deletion is execution-valid and causes a
  preregistered downstream task/progress change, and restoring only that segment
  restores the trajectory under the identical initial state and action suffix.
  This oracle replay is diagnostic evidence, never policy success.
- Legal deployment-time inputs: RGB observation, language instruction, and the
  Base policy's ordinary proprioception/action history only. Simulator state,
  contacts, future actions, success, and counterfactual outcomes are training-
  time labels and are unavailable at inference.
- Central novelty opportunity: downstream physical necessity, measured through
  matched simulator interventions, determines which low-motion supervision is
  retained. This is not magnitude, duration, phase, gripper proximity, or
  progress prediction.
- Simplest likely equivalent baseline: preserve a fixed number of actions around
  gripper changes. The thesis is killed if this explains essentially all
  eligible segments or matches the eventual closed-loop effect.
- Strongest objections: novelty may reduce to VLA-Arena/FrameSkip; replay
  interventions may be unstable; the signal may be only gripper timing; a
  simulation-only preprocessing result may not transfer; a smaller trainable
  Base may not be competent; and positive replay labels may not improve a VLA.
- Closure overlap: it does not use target maps, missing-view reconstruction,
  wrist dropout, delay recovery, memory replay at deployment, natural-reset
  residual mining, or a previously killed action head. Historical exact-init
  expert replay is infrastructure/discovery evidence only.
- Data and supervision: local LIBERO HDF5 demonstrations include actions and
  initial simulator states. Outcome/contact/progress variables are used only to
  construct discovery/development labels under frozen rules.
- Resource estimate: first falsification is CPU/simulator-heavy with no model
  download and negligible new disk relative to the 40 GB budget. A later small-
  Base study is estimated to require one-model residency within 16 GB VRAM and
  must pass a microbatch/resource smoke; no 7B full or official OFT training is
  assumed.
- First cheap falsification: a small preregistered replay-only panel comparing
  untouched, remove-all-near-zero, fixed-N preservation, and segment-restored
  variants, with no learning and no claim that replay is policy success.
- Closed-loop/generalization path: after the problem gate only, train one
  competent smaller Base under identical budgets with remove-all versus causal
  preservation; test closed-loop on untouched task/demo/reset partitions, then
  replicate on a second task family and, if resource-valid, a second backbone.
- Six-page story: a concrete supervision bug; causal simulator labeling;
  inference-free training intervention; controlled replay evidence; closed-loop
  policy results; gripper and heuristic controls; held-out transfer and limits.
- Why it could still fail after positive results: the labeler may help one
  dataset/backbone only, improvements may be fully explained by retaining more
  data, or the fixed-N/gripper baseline may match it.

## Initial rank 2 — active-candidate adjudication: outcome-separated VLA model selection and reporting

- One-sentence robotics thesis: VLA checkpoint/configuration selection must use
  outcomes disjoint from final reporting because adaptive reuse can change
  policy rankings and overstate closed-loop competence.
- Robotics importance: a reported winner determines which physical controller
  is deployed; selection bias can therefore choose a less competent robot policy,
  not merely perturb a benchmark statistic.
- Exact claim scope: episode-level simulator evaluation and checkpoint/config
  selection; no allegation about individual papers and no hardware-safety claim.
- Competent runnable Base: pinned OpenVLA-OFT inference through official LIBERO
  or the vla-eval harness after compatibility validation.
- Closest Prior: vla-eval standardizes inference/benchmark execution; RouterVLA
  uses outcome-disjoint cross-fitting for routing. Neither directly supplies a
  checkpoint-reporting confidence protocol.
- Strongest comparator: naïve best-of-K, final checkpoint only, and ordinary
  unpaired holdout under the identical evaluation harness.
- Artifact/fidelity: vla-eval has mature official code and OFT/LIBERO support,
  but its container and every candidate checkpoint need pinned hashes/licenses.
  The required multi-checkpoint candidate set is not yet verified locally.
- Problem plan: freeze candidate checkpoints, selection folds, untouched report
  folds, reset pairing, and ranking/effect criteria; repeat split assignment and
  quantify selection-versus-report ranking changes.
- Headroom: selection-adjusted lower bounds must identify cases where naïve
  selection chooses a different or materially overestimated candidate.
- Legal inputs: completed evaluation outcomes are legal to the selection
  protocol only in their assigned fold; confirmatory outcomes are inaccessible.
- Novelty opportunity: an episode ledger with nested paired selection and an
  adjusted lower bound tied to robot-policy choice.
- Simplest equivalent: ordinary held-out evaluation. If pairing/adjustment does
  not alter decisions or precision, the thesis is only hygiene.
- Strongest objections: not a method contribution; missing candidate snapshots;
  one benchmark/backbone; statistical novelty may be standard; no hardware.
- Closure overlap: none with the closed problem axes, but historical adaptive
  outcomes cannot be reused as confirmatory data.
- Data/supervision: closed-loop outcomes only; no new labels.
- Resource estimate: inference-heavy; the public harness container is reported
  near 6 GB, while model downloads must remain inside the 40 GB Epoch budget.
  Runtime depends on a verified multi-checkpoint set.
- Cheap falsification: inventory official candidate snapshots and run a tiny
  outcome-suppressed harness equivalence smoke before any selection study.
- Closed-loop/generalization: multiple task families, paired resets, untouched
  report folds, then a second compatible model family.
- Six-page story: reproducibility gap, outcome-separated protocol, statistical
  method, ranking changes, multi-task/model evidence, audit checklist.
- Post-positive failure risk: reviewers may view nested holdout as standard
  practice or the effect may vanish beyond one candidate pool.
- Independent hard-filter decision: `TOO_OVERLAPPING_OR_TRIVIAL`. RouterVLA
  already freezes candidate experts, selects from outcome probes, and scores on
  an excluded fold with cross-fitting and leakage controls. RSS 2026 N-SCORE
  already gives safe, anytime-valid and sample-efficient robot-policy
  comparison across bounded metrics. With a truly untouched final fold, an
  ordinary interval for the single frozen selected policy has valid coverage;
  pairing improves precision but is not a new inferential contribution. A
  checkpoint/configuration relabel therefore does not support a standalone
  RA-L thesis without a genuinely new finite-sample multi-candidate method.
- Execution decision: no problem rollout or harness experiment was run. The
  suggested small chunk-size study could establish that selection optimism
  exists, but even a positive result would only support an audit note.

## Initial rank 3 — active-candidate adjudication: training-only non-gripper contact-mode distillation

- One-sentence robotics thesis: training a VLA to recognize environmental
  contact modes can improve contact-rich action learning even when privileged
  simulator contacts are discarded at inference.
- Robotics importance: contact establishes and breaks kinematic constraints; a
  visually plausible action can fail when the policy misses the actual contact
  regime.
- Exact claim scope: simulation-only LIBERO/MuJoCo tasks with non-gripper
  environmental contacts and standard RGB/proprioceptive inference.
- Competent runnable Base: a resource-validated official 1B InSpire/miniVLA path
  is the leading candidate; no Base is selected until its local checkpoint,
  competence, and training memory are verified.
- Closest Prior: CALAMARI spatial contact maps; TacCoRL tactile-conditioned
  simulation/RL; StaKe stage and next-gripper-event supervision.
- Strongest comparator: an unstructured binary-contact auxiliary head and a
  StaKe-style stage/gripper-event head on identical data and compute.
- Artifact/fidelity: LIBERO simulator contact labels are locally available.
  InSpire code/models are official but not yet locally pinned; TacCoRL is a
  conceptual closest Prior and may not be protocol-compatible.
- Problem plan: on a frozen competent Base rollout panel, determine whether
  failures concentrate at missed/premature non-gripper contacts beyond gripper
  state, phase, and action magnitude.
- Headroom: an oracle contact-mode feature must improve a preregistered action
  or closed-loop diagnostic without becoming a legal inference input.
- Legal inputs: RGB, language, proprioception; no tactile signal, contact state,
  reward, or simulator privilege at inference.
- Novelty opportunity: a training-only, nonspatial environmental-contact state
  distilled into ordinary features and removed at deployment.
- Simplest equivalent: gripper transition or stage prediction. If it explains
  the signal, this thesis is killed.
- Strongest objections: contact may be visually unobservable; collision with
  stage/tactile work; privileged-label benefit may not survive distillation;
  simulation contacts may not transfer.
- Closure overlap: distinct from CALAMARI/TCA spatial maps and from wrist/delay
  axes; must not reuse the killed ContactTube or ContactSet formulations.
- Data/supervision: simulator contacts are available at training time; exact
  non-gripper label construction needs a frozen audit.
- Resource estimate: small auxiliary head, but the required official Base is not
  yet installed/validated; expected download and peak memory remain an artifact
  gate, not assumed facts.
- Cheap falsification: offline label prevalence/predictability plus a frozen
  failure-association audit, with no training if gripper state explains it.
- Closed-loop/generalization: untouched contact-rich tasks, non-contact retention,
  second task family, and a second Base only if resource-valid.
- Six-page story: contact-mode failure, privileged training label, distilled
  auxiliary mechanism, strong stage/gripper controls, closed-loop transfer.
- Post-positive failure risk: positive results may reflect privileged simulator
  leakage or generic temporal segmentation rather than contact reasoning.
- Independent hard-filter decision: `BACKUP_GATE_ONLY`, `NO-GO` as the active
  thesis. FD-VLA and HapticVLA already distill force/tactile knowledge into
  sensor-free deployment; CALAMARI, TacCoRL, StaKe, and GAP further occupy
  contact representations, tactile refinement, stage supervision, and
  phase-guided optimization. The generic privileged-contact claim is not
  defensible as new.
- Only residual formulation: training-only prediction of non-gripper
  object-environment contact-transition *topology* from MuJoCo geom pairs, with
  all robot/gripper contacts, tactile/force magnitude, spatial contact maps,
  and inference-time contact tokens excluded. It remains a lower-confidence
  backup, not an active thesis, because exact state restoration and geom-role
  mapping are unverified and the stored demonstrations have no contact labels.
- Required pre-Ours gate if this backup is ever reconsidered: on fresh,
  outcome-disjoint tasks, the topology label must be sufficiently prevalent,
  visually/proprioceptively predictable beyond gripper, phase, and action
  history, and yield oracle arm-action headroom beyond binary-contact and
  StaKe-style controls without degrading non-contact rows. Thresholds and task
  identities must be justified and frozen in a new protocol before outcomes.

## Historical scorecard before independent adjudication

Scores are 1–5 ranking aids, not gates. Confidence is `HIGH`, `MEDIUM`, or
`LOW`; evidence refers to the current literature map, closure registry, and
resource inventory.

| Criterion | Causal low-motion | Outcome-separated evaluation | Contact-mode distillation |
|---|---:|---:|---:|
| Novelty defensibility | 4 (MEDIUM) | 3 (MEDIUM) | 3 (LOW) |
| Robotics significance | 4 (MEDIUM) | 4 (MEDIUM) | 4 (MEDIUM) |
| Base/comparator credibility | 3 (MEDIUM) | 3 (MEDIUM) | 2 (LOW) |
| Repeated recoverable-gap probability | 4 (MEDIUM) | 3 (LOW) | 2 (LOW) |
| Technical mechanism quality | 4 (MEDIUM) | 3 (MEDIUM) | 3 (LOW) |
| Local feasibility | 4 (HIGH) | 3 (MEDIUM) | 2 (LOW) |
| Empirical decisiveness | 4 (MEDIUM) | 4 (MEDIUM) | 3 (LOW) |
| Simulation-only defensibility | 4 (MEDIUM) | 5 (HIGH) | 3 (LOW) |
| Generalization path | 4 (MEDIUM) | 4 (MEDIUM) | 3 (LOW) |
| Time to first falsification | 5 (HIGH) | 3 (LOW) | 4 (MEDIUM) |
| Distance from closures | 5 (HIGH) | 5 (HIGH) | 3 (MEDIUM) |

Final initial-portfolio decision: all three cards fail active-thesis selection.
The causal low-motion route and outcome-separated evaluation route are archived
without execution because of near-exact current-prior collisions. The broad
contact-distillation route is not novel enough; only the narrower contact-
topology residual remains a gated, lower-confidence backup. No thesis, Ours
method, training, problem-verification rollout, or confirmatory access is
authorized. Per the campaign state machine, the next action is a current
literature/artifact refresh for a materially new opportunity, not a forced
selection from this exhausted set.
