# SafeLoRA-VLA Source Feasibility

Date: 2026-07-08

No downloads, installs, training, rollouts, GPU jobs, or OpenVLA-OFT execution
were performed.

## Source Matrix

| Source | Official URL/source | Access/license status | Dataset/model size | Token/login | Dependencies | GPU/disk expectation | Small subset status | Safety labels/properties | Baselines | Local rollout/replay | LoRA/QLoRA path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LIBERO-Safety | https://libero-safety.github.io/, https://github.com/LIBERO-SAFETY/LIBERO-Safety, https://huggingface.co/datasets/LIBERO-Safety/libero_safety | Public/ungated. GitHub API reports no top-level SPDX license. HF license fields blank for dataset/assets/model. | HF dataset page reports 21k rows and 19.1 GB total; API used storage about 24.6 GB. Assets are about 10.7 GB. Released pi0.5 model is about 12.4 GB. | No token needed to inspect. | LIBERO-compatible repo, robosuite 1.4, robomimic, hydra, bddl, usd-core, rendering/system deps. | Full benchmark/eval requires simulator assets and likely GPU for VLA policies. Disk is at least dataset plus assets and optional model. | Task map exposes 5 suites x 3 levels x 5 tasks, but no explicit official tiny download/smoke split was found. HF streaming/sampling may be possible later, but it is not an official small split. | Physical and semantic safety suites are present. Property-level temporal unsafe labels for SafeLoRA preferences are not clearly exposed; paper notes current corpus focuses on safe demonstrations rather than hard-negative unsafe trajectories. | Paper evaluates 10 model families; pi0.5 weights released. | Not locally under current constraints because assets/simulator setup would be required. | SmolVLA-on-LeRobot is plausible but not clear enough; QLoRA tooling missing locally; property-conditioned LoRA labels absent. |
| SafeManip | https://hvkhcm.github.io/projects/safemanip/, https://github.com/chengyuehuang511/SafeManip | Public code/paper. GitHub API reports no top-level SPDX license. | Repo about 60 MB; RoboCasa365 checkpoints previously measured about 279 GiB; raw full monitor logs not bundled. | No token needed to inspect source metadata. | RoboCasa fork, robosuite, MONA, GR00T/OpenPI stacks, RoboCasa365 checkpoints. | Paper used A40 48 GB GPU nodes; official scale is 6 policies x 50 tasks x 50 rollouts. | Single-task scripts exist but still need checkpoints, simulator, and GPU rollout generation. | Strong LTLf temporal properties: contact, grasp/release stability, contamination, onset, mechanism, containment, enclosure/access. | Six official VLA policies plus prompt variants. | Too heavy locally. | No official LoRA training route; benchmark evaluates external policies. |
| ForesightSafety-VLA | https://arxiv.org/abs/2606.27079 | Paper public. No official code/data found in this gate. | Not inspectable from official source. | No token needed for arXiv. | RoboTwin mentioned in paper. | Not estimable without source. | Not found. | Strong paper metrics: 13-category taxonomy, CC, RET, four-quadrant outcomes. | Representative VLA baselines in paper. | Source blocked. | Source blocked. |
| SafeVLA-Bench | https://safevla.org/ | Project page/arXiv public. No code package found in this gate. | Not inspectable from official source. | No token needed for project page. | Native LIBERO/RoboCasa rollouts plus safety instrumentation. | Not estimable without code/assets. | Not found. | STL safety families, SR, Safety, SBU, VSI. | Reports LIBERO and RoboCasa model cells. | Source blocked for execution. | No training route; benchmark layer only. |
| Local standard LIBERO | Existing local project assets/reports | Local only, not a safety benchmark. | Existing local HDF5 snippets. | No. | Existing local tooling. | Light. | Yes locally. | Not official safety labels; previous proxy route killed. | Local proxy baselines exist. | Possible, but not acceptable evidence. | Proxy only; disallowed as RA-L evidence for this route. |

## Hard Source Gate

Required gate:

- official temporal/process safety metric,
- small official subset or clearly bounded official split/sample,
- realistic LoRA/QLoRA training path under constraints.

Result: failed.

No inspected source satisfied all three requirements at once. LIBERO-Safety is
the best future candidate, but the property-label and official bounded-subset
questions remain unresolved. SafeManip has the best temporal monitors but is
too heavy locally. ForesightSafety-VLA and SafeVLA-Bench are not source-green
for execution in this run.
