# SafeManip Reproduction Requirements

Date: 2026-07-08

This is a requirements inventory only. No setup, download, rollout, training,
GPU use, or OpenVLA-OFT use was performed.

## Requirement Summary

| Asset | Required for official reproduction | Availability | Local scout assessment |
| --- | --- | --- | --- |
| SafeManip paper | Yes | Public at arXiv | Accessible |
| SafeManip code | Yes | Public GitHub repository | Accessible, not cloned |
| SafeManip top-level license | Publication/redistribution due diligence | GitHub metadata shows no top-level license | Needs license audit before reuse beyond internal feasibility |
| RoboSuite | Simulator dependency | README instructs cloning ARISE-Initiative RoboSuite | Not installed |
| RoboCasa fork | Yes | Bundled in SafeManip repo | Not installed |
| MONA | Yes for DFA construction | `install.sh --monitor-only` can install if absent | Not installed |
| Monitor dependencies | Yes | SafeManip install helper | Not installed |
| GR00T stack | Needed for GR00T policies | Bundled integration plus upstream dependency terms | Not installed |
| OpenPI stack | Needed for pi0/pi0.5 policies | Bundled integration plus upstream dependency terms | Not installed |
| RoboCasa365 checkpoints | Yes for official policies | Public HF repo, not gated in metadata checked | Too large for current constraints |
| Raw official monitor JSON logs | Needed for metric-only paper reproduction | Not bundled in repo | Missing/source unavailable |
| Generated rollout data | Needed if logs unavailable | Must be produced by simulator/policies | Requires GPU rollout |
| Analysis scripts | Yes | Public in `SafeManip/analysis` | Accessible |
| SLURM scheduler | Used by official run scripts | Scripts submit job arrays | Local Windows environment is not a match |
| GPU | Needed for official rollout generation | Paper used A40 nodes | Not available/forbidden by task |

## Code And Repository

Official code: <https://github.com/chengyuehuang511/SafeManip>

The README states the repository contains:

- `Isaac-GR00T/` for GR00T policy and evaluation integration.
- `openpi/` for OpenPI policy and evaluation integration.
- `robocasa/` for a RoboCasa fork with privileged-state export.
- `SafeManip/monitor/` for symbolic predicates, LTLf/DFA logic, monitor
  metrics, and manual monitor invocation.
- `SafeManip/analysis/` for post-evaluation metrics and paper figure scripts.
- `run_scripts/` for SLURM launch and evaluation scripts.
- `examples/` for qualitative safety-category videos and monitor outputs.

Repository metadata checked via the GitHub API:

- Public: yes.
- Default branch: `main`.
- Reported repository size: about 60 MB.
- Top-level license: not reported.

## Checkpoints And Model Assets

SafeManip uses externally provided RoboCasa365-adapted checkpoints. The paper's
main evaluation set is:

- pi0
- pi0.5
- GR00T N1.5
- GR00T-pt
- GR00T-to
- GR00T-tpt

The README expects a local path similar to:

```text
${HOME}/.cache/huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/<snapshot-id>
```

and then points OpenPI and GR00T paths into that snapshot.

Hugging Face metadata for `robocasa/robocasa365_checkpoints` checked on
2026-07-08:

- Public: true.
- Gated: false.
- Card license field: `apache-2.0`.
- Reported used storage: 299,241,594,232 bytes, about 279 GiB.
- Sibling file count: 6,350.

This makes full local download unsuitable for the current constraints.

## Simulator And Dependencies

SafeManip is instantiated in RoboCasa/RoboCasa365. The official setup requires:

- RoboSuite clone and editable install.
- Local SafeManip RoboCasa fork editable install.
- Policy-stack-specific installs for GR00T and/or OpenPI.
- MONA for DFA construction.
- Python dependencies for monitor and analysis scripts.

The official README provides:

```bash
bash install.sh --monitor-only
bash install.sh --with-groot
bash install.sh --with-openpi
```

None of these commands were run in this scout.

## Official Outputs Needed

The analysis pipeline expects raw monitor data shaped as:

```text
SafeManip/analysis/rawData/<policy_name>/<task_name>/privileged_information_*_monitor.json
```

The official README says rollout directories may contain:

- `*.mp4`
- `privileged_information_<episode>.json`
- `privileged_information_<episode>_monitor.json`
- `stats.json`

The analysis README says `analysis/rawData`, `analysis/processedData`, and
`analysis/plots` are local/generated folders. They are not present as full
paper-result assets in the public repo.

## GPU, Runtime, And Disk

Paper protocol:

- 6 policies.
- 50 tasks.
- 50 rollouts per task.
- 15,000 total rollouts.
- All experiments used NVIDIA A40 GPU nodes.
- Each task was allocated one 48 GB A40 GPU.

Local implication:

- Full reproduction is not feasible without GPU/cloud.
- Small official subset still needs simulator install, model checkpoint
  download, and GPU inference.
- Metric-only reproduction from full official logs is blocked because full logs
  are not public in the repo.
- Disk for full checkpoint assets alone is about 279 GiB from HF metadata,
  before rollout JSON/video output.

Expected runtime:

- The paper and README do not provide a wall-clock runtime.
- Given the official use of one 48 GB A40 per task and 15,000 rollouts, local
  CPU/Windows reproduction is not a credible 24-48 hour path.

## Login, Token, And Access

No login or token was needed to inspect:

- SafeManip GitHub source.
- RoboCasa365 checkpoint repository metadata.
- LIBERO-Safety fallback project/code/dataset metadata.

Potential access issues still needing audit before reproduction:

- Upstream model license acceptance for OpenPI/pi0/pi0.5/Gemma-related assets.
- GR00T and NVIDIA-related model or code terms.
- SafeManip top-level license absence.

## 24-48 Hour Local Options

| Option | Feasible under current constraints | Reason |
| --- | --- | --- |
| Full benchmark reproduction | No | Requires checkpoints, GPU, rollouts, and 15,000 evaluations |
| One-policy one-task official subset | No locally | Still requires checkpoint download, simulator setup, GPU, and rollout |
| Metric-only from official full logs | No | Full raw monitor logs are not bundled |
| Monitor-only audit on example JSON | Conditional | Possible only after fetching repo/example JSON; not official reproduction |
| Analysis-code static audit | Yes | Can inspect scripts and CSV schemas without experiments |

## Requirement Verdict

SafeManip is usable as an official benchmark source, but not as a no-download,
no-GPU local reproduction target. The correct gate is `TOO_HEAVY_LOCAL`, not
`SOURCE_BLOCKED`.
