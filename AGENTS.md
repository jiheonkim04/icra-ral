# AGENTS.md

This repository is an autonomous robot-learning research workspace. The current authoritative research governance is:

`reports/current_research_governance.md`

Future Codex sessions must read that file before treating any older report, state file, prompt, or manual entry as active governance. Older TCA-first and bounded-autopilot instructions are historical evidence unless explicitly imported into the current governance file.

## Active Operating Rules

1. Do not fabricate results, hide failed runs, overwrite previous experiment outputs, cherry-pick tasks/resets/seeds/metrics, or claim paper-grade evidence from proxy checks.
2. Do not use privileged simulator state at default inference time. Simulator labels may be used only for training supervision, evaluation metrics, or oracle ablations when predeclared.
3. Separate `DISCOVERY`, `VALIDATION`, and `CONFIRMATORY_TEST` evidence. Use discovery and validation to build the strongest honest method, then freeze proposals, baselines, ablations, task/reset allocation, statistics, configuration, and kill/scale rules before inspecting confirmatory results.
4. Maintain branch safety: do not modify `main` directly, do not revert user work, and do not use destructive git or filesystem commands unless explicitly requested.
5. Run bounded risk assessment before downloads, GPU work, training, rollout, simulator setup, or heavy imports. Proceed autonomously only when source, size, disk, RAM/VRAM, runtime, dependency, license/token, and repository policy checks are inside budget.
6. Stop before token/secret/API-key access, paid services, license click-through, external submission or publishing, deleting user files outside approved repo/cache cleanup, system-wide CUDA/PyTorch/driver changes, credentialed system setup, or unsupported empirical claims.
7. Preserve resource monitoring for GPU memory, runtime, disk, downloads, checkpoints, and resumability.
8. Before commit or merge, inspect the changed-file count and line diff. If the change would include more than 50 files or more than 5,000 changed lines, record the scope and justification before proceeding.

## Performance-Oriented Method Development

After the fixed CAVM-VLA adjudication, future methods should be anchored to a positive external prior when possible and may use bounded validation-only design search before confirmatory testing. This includes literature-derived design principles, discovery diagnostics, validation tasks/reset identities, bounded hyperparameter search, clean-retention diagnostics, mechanism smoke tests, and external-prior reproduction.

After the closed RAC-VLA Stage B result, `reports/current_research_governance.md` also imports the post-RAC objective:

`MAXIMIZE_THE_PROBABILITY_OF_AN_HONEST_PAPER_WORTHY_POSITIVE_RESULT`

This means Researcher A should design stronger methods before confirmatory testing: identify usable headroom, verify data and supervision health, prefer positive external priors, run bounded validation search, preserve the pretrained policy by default, and audit mathematical objectives before expensive rollout. It does not authorize confirmatory-test tuning, cherry-picking, or rescue of a valid kill.

Do not tune on confirmatory test identities or reinterpret confirmatory results to rescue the same method. A major redesign after confirmatory test is a new method cycle.

Future method proposals must include:

- closest positive external prior or a justified reason no anchored prior is feasible;
- pre-experiment headroom and data/contrast audit;
- bounded validation search budget;
- mathematical objective audit with term scale and gradient checks;
- identity-preserving integration audit;
- one default simple reviewer-killer baseline at initial prototype unless a concrete reviewer objection justifies another.

Future long-running WSL experiments should use detached durable execution with PID, heartbeat, logs, partial result, exact resume command, and missing-key-only resume behavior.

## Deprecated Instructions

The following old instructions are no longer active governance:

- one-major-milestone-per-execution stopping;
- TCA-Map as the default publishable path;
- any requirement that TCA-Select, ActionMap, LoRA, QLoRA, SmolVLA-only adaptation, or any other named method family become the final method;
- any global maximum of three method cycles;
- any no-method terminal state after a fixed number of cycles;
- the obsolete 30-minute global GPU limit;
- the obsolete assumption that official SmolVLA/LIBERO assets are missing;
- any blanket prohibition on the already validated quantized OpenVLA-OFT INT4 execution path.

TCA-Map, TCA-Select, ActionMap, ECHO, PhaseBarrier, CensorCredit, DICD, FEDO, GCAP, and related reports remain historical evidence and possible baselines or cautionary examples. They are not mandatory destinations.

## Research Campaign Semantics

The active campaign may run multi-stage autonomous research inside one Goal execution when risk-assessed work remains within local constraints. A failed method does not terminate the campaign. Archive it, record the failed assumption, and pivot automatically.

For every future method, separate the scientific method from its low-compute parameterization. LoRA and QLoRA are implementation infrastructure unless adaptation efficiency is the explicit research problem. The default first comparison is Base, closest Prior, Ours, and the key ablation; add a fifth control only when it tests the strongest plausible alternative explanation. Standard LoRA is conditional, not automatic, and every omission must state why it does not test the claimed mechanism. Prefer one core mechanism, one primary objective, at most one necessary auxiliary term, and one key ablation. A demonstrated adapter-capacity failure is `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`, with at most one bounded capacity adjustment before confirmatory testing.

Normal success is:

`READY_TO_DRAFT_RAL_PAPER_PACKAGE`

Allowed final states are exactly:

1. `READY_TO_DRAFT_RAL_PAPER_PACKAGE`
2. `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`
3. `HARD_EXTERNAL_BLOCKER`
4. `SAFETY_RESOURCE_STOP`

Before any future terminal decision, run:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\check_current_research_governance.py
```

and the relevant test suite.
