# Next Topic Candidates

Date: 2026-07-07

This file defines candidate topics only. It does not authorize experiments, training, rollout, downloads, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, or paper claims.

## Candidate 1: Active Micro-Probe Goal Disambiguation

Task definition:

Use small, bounded robot micro-actions to gather physical evidence about which object, contact region, or approach direction matches an ambiguous instruction before committing to a longer manipulation action. The first metric is wrong-target movement and target-progress change under controlled ambiguous-object scenes.

Latest-paper novelty hypothesis:

Recent VLA work improves general policies, action decoding, and fine-grained instruction following, but mostly treats the robot as a passive executor of an interpreted instruction. FineVLA targets fine-grained steerability from language and annotations, while A2C2/VLA-Corrector target stale action chunks and SafeVLA targets constrained safety. The hypothesis here is that active physical probing before commitment can reduce goal ambiguity in ways passive instruction alignment and safety filters do not address.

What existing papers do not solve:

They do not make a pre-commitment micro-action an explicit evidence-gathering step with a direct control metric against random-probe, no-probe, safety-only, and clipping-only controls.

Method novelty:

A probe planner selects a low-amplitude action expected to reveal target/object response; a response observer measures object-state or image-delta evidence; a commit gate chooses whether to proceed, switch target, or stop. The mechanism is active sensing/control, not calibration, clipping, or semantic shielding.

Simplest strong baseline:

No-probe direct commit, random micro-probe, nearest-object heuristic probe, clipping-only, safety-only stop rule, and mean-action/hold action. Diagonal affine is included only if learned/generated actions are used.

First 48-hour executable test:

Use local LIBERO/RoboSuite object-state keys or a tiny toy manipulation scene to build 10-20 ambiguous-target trials. Run no-probe, random-probe, heuristic-probe, and active-probe variants for 1-3 probe steps plus a short scripted commit. Report wrong-target displacement, intended-target displacement, unsafe/contact violation, and task-progress proxy. This first test must not require VLA model loading.

Kill criteria:

- no direct control metric within 48 hours,
- active probe does not beat random-probe and no-probe by at least 10 percentage points or an equivalent predeclared effect size within 72 hours,
- safety-only or clipping-only explains the improvement,
- probe actions regularly damage task progress or move the wrong object,
- the method requires native VLA competence before the first metric.

Expected data/model/simulator assets:

Local LIBERO/RoboSuite, HDF5 metadata, object-state observations where available, existing WSL simulator path, optional toy MuJoCo scene, no heavy VLA requirement for the first gate.

Why it can be RA-L-stable:

The evidence path is direct robot control behavior. Failure cases are easy to generate, baselines are obvious, and the method mechanism is active information gathering rather than a passive VLA wrapper.

Why it might fail:

LIBERO object-state keys may not expose enough evidence, micro-probes may be too weak to disambiguate, random probes may match the planned probe, or the probe may hurt downstream task progress.

## Candidate 2: Demonstration-Tube Recovery Library

Task definition:

Inject small execution deviations during exact-init demonstration replay, then recover by selecting a local recovery segment from a demonstration tube using current state, object progress, and remaining task context. The primary metric is recovery success or target-progress recovery after controlled perturbations.

Latest-paper novelty hypothesis:

Retry-supervised value learning treats retry events as useful supervision, and path-consistent safety filtering keeps policies close to intended paths. This candidate asks whether a lightweight executable recovery library can use demonstration-tube structure to recover from near-miss states without training a large policy or collapsing into action calibration.

What existing papers do not solve:

They do not provide a baseline-first, replay-first recovery gate that compares a state-indexed recovery library against continue-original, nearest-action, mean-action, clipping-only, and diagonal-affine baselines on exact-init perturbation recovery.

Method novelty:

Build a state-progress index over demonstrations, detect off-tube deviation, retrieve a recovery segment with object/proprioceptive compatibility, and execute a short correction before returning to the nominal trajectory.

Simplest strong baseline:

Continue original replay after perturbation, nearest-neighbor action, nearest-neighbor future segment without recovery scoring, mean-action, zero/hold action, clipping-only, and diagonal affine action correction.

First 48-hour executable test:

Use one validated exact-init LIBERO demonstration. Inject a bounded action or state deviation before the known success window, then compare recovery variants over a short horizon. Report reward/success if reachable, otherwise target distance and return-to-tube error. The first test uses existing replay infrastructure and no VLA model loading.

Kill criteria:

- no perturbation recovery metric within 48 hours,
- nearest-neighbor or mean-action baseline matches the method within tolerance by 72 hours,
- diagonal affine or clipping-only explains the improvement,
- recovery requires future expert actions from the same evaluation trajectory in a way that would be unavailable at deployment,
- the method only works on one hand-selected perturbation.

Expected data/model/simulator assets:

Local LIBERO HDF5 demos, exact-init replay path, object/proprioceptive states, existing simulator setup. No heavy VLA needed for the first gate.

Why it can be RA-L-stable:

It targets a direct manipulation-control problem: recovering from near-miss states. It is not a VLA wrapper, and it can be judged against hard, simple replay/control baselines early.

Why it might fail:

Demonstration coverage may be too sparse, exact-init perturbation recovery may be brittle, or a nearest-neighbor segment baseline may already capture most of the benefit.

## Candidate 3: Path-Consistent Event-Triggered Chunk Guard

Task definition:

Monitor execution of action chunks and trigger truncation, slowdown, or path-consistent replanning when live state deviates from the expected state tube. The primary metric is control success, target progress, jerk, and failure rate under injected observation delay or object perturbation.

Latest-paper novelty hypothesis:

RTC, A2C2, VLA-Corrector, and PACS show that action chunking, stale observations, and path-consistent safety are important. The remaining possible novelty is a lightweight, baseline-first guard that combines state innovation, adaptive chunk horizon, and path-consistent intervention without relying on a heavy VLA backbone or learned latent monitor for the first metric.

What existing papers do not solve:

The crowded literature does not necessarily provide a low-compute, local, baseline-audited guard that must beat fixed-horizon execution, temporal ensembling, clipping-only, safety-only, and mean-action controls before scaling.

Method novelty:

Use a state-tube innovation monitor to decide whether to execute, slow, truncate, or replan the remainder of a chunk. Unlike pure safety shielding, the guard is judged by preserving task progress while reducing stale-chunk failures.

Simplest strong baseline:

Fixed-horizon chunk execution, execute-one-step, temporal ensembling, clipping-only, safety-only stop/brake, mean-action/hold, and a simple innovation threshold without path projection.

First 48-hour executable test:

Chunk local HDF5 expert actions or a tiny scripted policy, inject delay or object motion in a toy MuJoCo/LIBERO-style scene, and compare fixed chunking against event-triggered guarding. Report success/progress, jerk, intervention rate, and stale-chunk failure rate. Avoid heavy VLA loading for the first gate.

Kill criteria:

- no direct control metric within 48 hours,
- fixed-horizon, temporal ensemble, clipping-only, or safety-only matches the guard by 72 hours,
- novelty cannot be separated from RTC, A2C2, VLA-Corrector, or PACS,
- the guard simply stops too often and becomes another full-intervention shield,
- the first metric requires native VLA competence or heavy model imports.

Expected data/model/simulator assets:

Local HDF5 action chunks, existing MuJoCo/LIBERO simulator path, optional toy dynamic scene, no VLA model requirement for the first gate.

Why it can be RA-L-stable:

If it beats strong action-chunk and safety baselines on direct control metrics, it connects to a timely robotics problem with a clear evidence path.

Why it might fail:

The literature is very crowded, simple fixed-horizon or safety baselines may be enough, and the novelty could be too close to recent action-chunk correction work.

## Recommendation

Pick Candidate 1 first. It is the cleanest break from the killed routes because it starts from active control evidence, not offline proxy, not VLA native competence, not calibration, and not passive safety shielding.

