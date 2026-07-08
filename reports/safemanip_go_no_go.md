# SafeManip Go/No-Go

Date: 2026-07-08

Decision: `TOO_HEAVY_LOCAL`

## Decision Basis

SafeManip is not source-blocked:

- The paper is public.
- The project page is public.
- The GitHub repository is public.
- The repository includes monitor code, predicate/specification logic,
  RoboCasa instrumentation, launch scripts, and analysis scripts.
- The RoboCasa365 checkpoint repository is public and not gated in metadata
  checked on 2026-07-08.

SafeManip is too heavy for the current local constraints:

- The current instructions forbid experiments, rollout, training, large
  downloads, GPU use, and OpenVLA-OFT.
- Official reproduction requires simulator setup, checkpoint assets, GPU
  rollout generation, and monitor output generation.
- The paper reports 6 policies x 50 tasks x 50 rollouts, with NVIDIA A40
  48 GB GPU nodes and one GPU per task.
- The RoboCasa365 checkpoint repository metadata reports about 279 GiB of used
  storage.
- The full raw monitor JSON logs used for paper figure reproduction are not
  bundled in the SafeManip repository.

## Local Feasibility Calls

| Reproduction option | Verdict |
| --- | --- |
| Full benchmark reproduction | No |
| Small official subset reproduction | No under current constraints |
| Metric-only reproduction from official full logs | No |
| Monitor-only audit from example JSON | Conditional, not official reproduction |
| Static source/analysis audit | Yes |

## Why Not Other Decisions

`GO_SAFE_MANIP_REPRODUCTION`: no. Current constraints disallow the required
downloads, GPU inference, and rollout generation.

`SOURCE_BLOCKED`: no. The official code and paper are accessible; the blocker is
resource weight and missing bundled raw logs, not total source unavailability.

`NO_CLEAR_METHOD_GAP`: no. A method gap exists: utility-preserving temporal
safety improvement is not solved by the benchmark.

`FALLBACK_TO_LIBERO_SAFETY`: not chosen as the SafeManip go/no-go because
SafeManip is not source-blocked. LIBERO-Safety remains a reasonable fallback if
the user wants to keep a local no-GPU/no-large-download route.

## Fallback Note

Because the selected decision is `TOO_HEAVY_LOCAL`, the fallback is optional
rather than automatic.

If the user declines GPU/cloud and large checkpoint download authorization, the
next scout should be LIBERO-Safety official benchmark feasibility:

- Confirm official code, dataset, model/checkpoint assets, and licenses.
- Determine whether a metric-only or small official subset can run without GPU.
- Check whether its public dataset can support a no-rollout static metric audit.
- Keep the same constraints: no method, no training, no rollout, no large
  downloads, no GPU.

## Exact Next Step

Ask for one explicit direction:

- SafeManip cloud/GPU minimum subset plan: authorize one-policy, one-task
  official reproduction planning, including exact checkpoint subset and compute
  estimate.
- LIBERO-Safety fallback feasibility scout: keep local constraints and audit
  whether LIBERO-Safety has a lighter official reproduction path.
