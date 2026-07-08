# SafeManip Anchor Feasibility Scout

Date: 2026-07-08

Status: feasibility scout only. No method implementation, experiments, training,
rollouts, large downloads, GPU use, or OpenVLA-OFT use were performed.

Go/no-go: `TOO_HEAVY_LOCAL`

## Scope

This scout starts from the project reset direction in
[ral_strategy_reset.md](ral_strategy_reset.md): use an official benchmark/source
as the next anchor before proposing any new RA-L method. The immediate target is
SafeManip, with LIBERO-Safety as a fallback only if SafeManip cannot be used as
the anchor.

The stale ActionMap route is stopped. The untracked docs-only ActionMap files
that existed on `main` were parked in a stash and were not merged.

## Sources Checked

- SafeManip paper: [arXiv 2605.12386v2](https://arxiv.org/html/2605.12386v2)
- SafeManip project page: <https://hvkhcm.github.io/projects/safemanip/>
- SafeManip official code: <https://github.com/chengyuehuang511/SafeManip>
- SafeManip README and analysis README in the official repository
- RoboCasa365 checkpoint repository metadata:
  <https://huggingface.co/robocasa/robocasa365_checkpoints>
- RoboCasa/LeRobot documentation:
  <https://huggingface.co/docs/lerobot/main/robocasa>
- LIBERO-Safety fallback references:
  <https://libero-safety.github.io/>,
  <https://github.com/LIBERO-SAFETY/LIBERO-Safety>,
  <https://huggingface.co/datasets/LIBERO-Safety/libero_safety>
- Local reset/context reports:
  [all_killed_routes_summary.md](all_killed_routes_summary.md),
  [simple_baseline_failure_patterns.md](simple_baseline_failure_patterns.md),
  [next_topic_selection_criteria.md](next_topic_selection_criteria.md),
  [project_state.md](project_state.md),
  [next_actions.md](next_actions.md),
  [decision_log.md](decision_log.md)

## What SafeManip Provides

SafeManip is primarily a benchmark and evaluation protocol, not a new VLA
training or deployment method. The paper frames it as a policy-agnostic
temporal safety benchmark for robotic manipulation. It provides reusable LTLf
safety property templates, task-specific predicate bindings, symbolic
monitoring, rollout processing, and analysis scripts.

Method mechanics:

- Rollouts are finite traces, so SafeManip uses Linear Temporal Logic over
  finite traces, LTLf.
- Simulator state is converted into Boolean propositions at each timestep.
  The paper names object poses, contact events, gripper state, fixture state,
  and task-relevant action signals as monitored state.
- Task-specific bindings instantiate reusable safety templates over concrete
  objects, fixtures, regions, and skills.
- Each LTLf formula is compiled to a DFA and updated online over the symbolic
  trace.
- A rollout violates a property when the corresponding monitor reaches a
  rejecting state. The monitor records violation timestep, duration, and
  property category.

Temporal safety categories:

- Collision and contact safety
- Grasp stability
- Release stability
- Cross-contamination safety
- Action-onset safety
- Mechanism safety
- Containment safety
- Enclosure and access safety

Task suite:

- Simulator: RoboCasa with privileged simulator state export.
- Benchmark tasks: 50 RoboCasa365 kitchen manipulation tasks.
- Suites: Atomic and Fixture; Beverage Preparation and Serving; Bread,
  Breakfast, and Reheating; Cooking and Ingredient Preparation; Cleaning,
  Washing, and Sanitation; Storage and Organization; Plating, Serving, and
  Portioning.
- Horizons: atomic, short, medium, and long, using task metadata in
  `taskDiff.csv`.

Metrics:

- Task success rate.
- Overall rollout-level safety violation rate.
- Per-property and per-category violation rates.
- Four rollout outcomes: success-and-safe, success-but-unsafe, fail-but-safe,
  fail-and-unsafe.
- Unsafe-state exposure rate, defined as the share of rollout timesteps spent
  in a violating state.
- RQ1/RQ2/RQ3 analysis scripts for success-vs-safety, category breakdowns,
  suite breakdowns, and horizon breakdowns.

Models evaluated:

- The paper reports six externally provided RoboCasa365-adapted VLA
  checkpoints: pi0, pi0.5, GR00T N1.5, GR00T-pt, GR00T-to, and GR00T-tpt.
- The repository policy mapping also contains two safety-prompt labels,
  `GR00T-tptf` and `GR00T-tpts`; the paper treats safety prompting as an
  exploratory analysis, not the main benchmark policy set.
- The paper states it does not train or fine-tune policies; it evaluates
  externally provided checkpoints.

Official baselines:

- The official benchmark baselines are the evaluated VLA checkpoints and GR00T
  training variants.
- The exploratory prompt variants are a weak safety-guidance baseline.
- There is no official safety filter, stop-on-risk controller, clipping
  baseline, reward-penalty policy, DPO/preference-tuned policy, multi-model
  correction baseline, or deployment-time intervention baseline.

Claimed results:

- Task success does not reliably imply temporal safety.
- In the paper's example, pi0.5 improves task success over pi0 from 8.1% to
  9.3%, while its violation rate rises from 69.7% to 82.8%.
- The paper reports GR00T-tpt at 43.9% overall success and 71.8% overall
  violation rate.
- Short and long safety prompts reduce violations only slightly, while sharply
  lowering success: short prompt 26.4% success and 69.4% violation; long prompt
  6.9% success and 65.1% violation.
- Collision/contact and release-stability failures dominate; longer-horizon
  tasks amplify temporal safety failures.

## Assets Required

SafeManip official reproduction needs:

- SafeManip source repository.
- RoboSuite clone and editable install.
- The SafeManip RoboCasa fork.
- Monitor dependencies, including MONA for LTLf-to-DFA construction.
- GR00T and/or OpenPI policy environments.
- RoboCasa365-adapted policy checkpoints.
- RoboCasa365 simulator assets and any upstream model assets needed by GR00T,
  OpenPI, pi0, and pi0.5.
- Generated rollout outputs: videos, privileged simulator JSON, monitor JSON,
  and `stats.json`.
- Raw monitor JSON files under `SafeManip/analysis/rawData/<policy>/<task>/`
  to reproduce paper figures.

The official analysis code is present, but the full raw monitor JSONs used for
paper figures are not bundled in the repo. The README says `analysis/rawData`,
`analysis/processedData`, and `analysis/plots` are ignored/generated local
folders.

Resource notes:

- The paper states all experiments used NVIDIA A40 GPU nodes, with each task
  allocated one 48 GB A40 GPU.
- The RoboCasa365 checkpoint Hugging Face repository is public and not gated in
  metadata checked on 2026-07-08, but its reported used storage is about
  299.2 GB.
- The SafeManip GitHub repository metadata reports about 60 MB, but cloning it
  would also fetch bundled examples and vendored/forked code.
- Full benchmark rollout generation is 6 policies x 50 tasks x 50 rollouts,
  or 15,000 rollouts, before analysis.
- Training time is not applicable to SafeManip reproduction because the paper
  does not train policies.
- Evaluation runtime is not specified in wall-clock terms; the official setup
  assumes GPU/SLURM-style execution.

License/access notes:

- SafeManip GitHub metadata does not advertise a top-level license.
- The repo contains upstream components with their own licenses, including
  RoboCasa, OpenPI, and Isaac-GR00T related files.
- The RoboCasa365 checkpoint card reports `apache-2.0`, but upstream model
  terms should be audited before redistribution or publication.
- No login/token was needed to inspect the SafeManip GitHub repository or the
  RoboCasa365 checkpoint repository metadata.

## Local 24-48 Hour Reproduction Assessment

Full benchmark reproduction: no.

Reason: the full official protocol requires large checkpoint downloads, GPU
rollout generation, simulator setup, and 15,000 monitored rollouts. This
violates the current no-download, no-GPU, no-rollout constraints.

Small official subset reproduction: no under current constraints.

Reason: the official run scripts include single-task paths, and one task with
one policy could be technically possible on an appropriate GPU with the needed
checkpoint assets. It is not locally feasible in this scout because it still
requires checkpoint downloads, simulator setup, GPU inference, and rollout
generation.

Metric-only reproduction from official logs/data: no.

Reason: the analysis scripts are public, but the full raw monitor JSONs needed
to regenerate paper metrics are not bundled. Without official logs, the metric
pipeline can be audited but not used to reproduce reported figures.

Monitor-only local audit: conditional and non-official.

Reason: the repo includes qualitative example monitor JSON files and a
monitor-only install path. A small monitor parser/analysis audit could be run
after a repo clone or selective file fetch, but this would not reproduce the
official benchmark.

No feasible official local reproduction: yes, under the current constraints.

## Strongest RA-L Topic Gap

SafeManip exposes an evaluation gap rather than solving a method gap. It gives
good temporal safety metrics, but it does not provide a method that improves
safe success.

The strongest gap is utility-preserving temporal safety improvement: improve
SafeManip safe-success and reduce violation/exposure while preserving task
success. This is attractive because the paper's safety-prompt pilot shows the
obvious safety-only response can reduce success much faster than it reduces
violations.

However, this should not become a method route yet. The project rules from
[ral_strategy_reset.md](ral_strategy_reset.md) and
[next_topic_selection_criteria.md](next_topic_selection_criteria.md) require an
official anchor reproduction first.

## Baseline Pressure

Any future SafeManip method would need to beat these baselines from
[simple_baseline_failure_patterns.md](simple_baseline_failure_patterns.md):

- Base policy with official SafeManip metrics.
- Safety-only filter or monitor stop.
- Stop-on-risk or abort-on-uncertainty.
- Clipping-only action constraints.
- Generic reward penalty where labels/data are available.
- Generic DPO/preference tuning where pairwise safety labels are available.
- No-op or abort baseline, which may reduce violations by creating fail-safe
  rollouts.
- Official SafeManip VLA baselines and prompt variants.

Simple baselines are especially dangerous here because a no-op/abort or
over-conservative prompt can reduce violation rate while destroying success.
The benchmark's safe-success and outcome decomposition are therefore mandatory,
not optional.

## Decision

`TOO_HEAVY_LOCAL`

SafeManip is not source-blocked: the paper, project page, code, monitor design,
run scripts, and analysis scripts are public. The blocker is local
reproduction weight under the current constraints. Official reproduction needs
large model/checkpoint assets, GPU rollout generation, and raw monitor logs
that are not bundled.

Recommended next step:

Do not start a SafeManip local reproduction under the current constraints. Ask
for one of two explicit authorizations:

- Allocate cloud/GPU and permit the minimum checkpoint download needed for a
  one-policy, one-task official SafeManip subset.
- Keep the local no-download/no-GPU constraint and open a separate
  LIBERO-Safety official benchmark feasibility scout.
