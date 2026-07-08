# SafeLoRA-VLA Risk Register

Date: 2026-07-08

| Risk | Severity | Evidence | Mitigation |
| --- | --- | --- | --- |
| Generic DPO/ORPO kills the novelty | High | SafeTrace-VLA already collapsed to generic preference and safety-only baselines on local proxy pairs. | Require generic DPO/ORPO LoRA on the same official pair set before any claim. |
| Safety-only or stop-on-risk explains gains | High | Safety prompts/filters can lower violation rate while destroying success; previous routes were killed by safety-only baselines. | Primary metric must be safe-success with no-op/abort reporting. |
| No official unsafe/property labels | High | LIBERO-Safety released safe/collision-free demonstrations and notes lack of dense hard-negative unsafe annotation. | Use only official rollouts/logs or official monitor-derived violations; do not invent local proxy labels as evidence. |
| SafeManip too heavy locally | High | Official route needs RoboCasa365 policy assets, GPU rollouts, and large checkpoints/log generation. | Treat as cloud-only unless the user authorizes minimum subset resources. |
| LIBERO-Safety assets/data size | Medium-high | Dataset API reports about 24.6 GB used storage, assets about 10.7 GB, and released pi0.5 model about 12.4 GB. | Do not download in this run; require explicit user approval and a bounded subset plan. |
| QLoRA not runnable locally now | Medium | Local guard found `peft` and `bitsandbytes` absent. | Defer QLoRA to approved install/Linux/cloud; keep LoRA separate from QLoRA. |
| SmolVLA LoRA path is not official | Medium-high | Local SmolVLA assets are ready, but official docs describe ordinary fine-tuning, not property-conditioned safety LoRA. | Implement only after an explicit adapter design and source-label path are green. |
| License ambiguity | Medium | LIBERO-Safety and SafeManip GitHub API metadata did not report top-level SPDX licenses; HF license fields were blank for LIBERO-Safety dataset/model/assets. | Audit licenses before redistribution, publication, or dataset/model packaging. |
| Method becomes another benchmark observation | Medium | SafeManip, LIBERO-Safety, ForesightSafety, and SafeVLA-Bench are diagnostic/evaluation-heavy. | Require weight-level LoRA adaptation and baseline comparisons. |
| Utility retention unavailable | High | Dataset-only offline training may lack task success/rollout outcomes. | Do not claim continuation unless task utility and safety can be measured together. |

## Estimated Kill Probability

Estimated kill probability before blocker resolution: `0.70`.

Strongest likely killer: generic DPO/ORPO LoRA or standard imitation LoRA on the
same official safe/unsafe pairs, followed by safety-only stop/filter baselines.
