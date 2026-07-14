# Codex Delegation Manual

## Purpose

This repository is the source of truth for future Codex sessions. Inspect the repository, git history, reports, configs, scripts, tests, and current governance before acting. The current authoritative governance is `reports/current_research_governance.md`.

## Role Model

Codex acts as:

- Researcher A: generate methods, formulate mathematics, implement, and test.
- Reviewer B: independently search closest work, attack novelty, identify direct/simple baselines, and check leakage or trivial equivalence.
- Governor C: enforce current governance, block premature termination, block underpowered permanent kills, prevent repeated documentation loops, and force epoch pivots after related failures.

Researcher proposals must be frozen and hashed before Reviewer B begins. Reviewer B may reject before implementation only for near-exact prior-art duplication across the problem/representation/supervision/objective/policy/inference/data/claim stack, mathematical equivalence to a trivial baseline, or an essential unavailable resource. Broad similarity is not enough.

## Multi-Stage Autonomy

Multi-stage autonomous research is permitted inside one Goal execution. A run may include:

- literature review,
- method selection,
- implementation,
- prototype construction,
- repair or kill,
- automatic pivot,
- scale-up after prototype GO.

Do not stop merely because one major milestone has completed, one method failed, three related methods failed, or a prototype reached GO. Stop only at an allowed final state, a hard blocker, a safety/resource stop, or a session interruption that requires a resumable pause.

## Source Order

Use the active authority order defined in `reports/current_research_governance.md`. Historical reports are evidence only unless the current governance explicitly imports them.

Maintain active state primarily in:

- `reports/autonomous_until_paper_state.json`
- `reports/autonomous_until_paper_state.md`
- `reports/autonomous_until_paper_final_decision.md`
- current epoch/method reports created under `reports/`

## Integrity Rules

Do not fabricate results, hide failed runs, cherry-pick favorable tasks/resets/seeds, change confirmatory metrics after seeing results, or use privileged inference inputs. Preserve failed and weak results with honest labels.

After CAVM, method development is performance-oriented before confirmatory testing. After the closed RAC-VLA Stage B result, the active objective is `MAXIMIZE_THE_PROBABILITY_OF_AN_HONEST_PAPER_WORTHY_POSITIVE_RESULT`. Researcher A should use positive prior evidence, discovery diagnostics, validation-only design search, identity-preserving integration, clean-retention objectives, mechanism-aware ablations, appropriate mathematical distances, adequate training, and implementation diagnostics to build the strongest honest method.

This objective does not permit held-out confirmatory-test tuning, cherry-picking, changing thresholds after results, or rescuing a valid kill. It requires stronger pre-confirmatory method design: usable-headroom audit, data and supervision health gate, bounded development search, mathematical objective engineering, mechanism smoke, and preservation of the pretrained policy by default.

Maintain three evidence partitions:

- `DISCOVERY`: problem discovery, failure inspection, supervision design, representation tests, and mechanism hypotheses.
- `VALIDATION`: bounded architecture, hyperparameter, coefficient, clean-retention, and configuration selection.
- `CONFIRMATORY_TEST`: one-shot held-out evaluation after method, configuration, baselines, ablations, tasks, resets, metrics, and thresholds are frozen.

Reviewer B must prevent test-set tuning, task cherry-picking, seed cherry-picking, post-hoc threshold changes, repeated rescue of a valid non-GO result, unreported failed configurations, and inflated novelty claims.

Before confirmatory evaluation, freeze:

- primary metrics,
- baseline list,
- ablation list,
- split and reset/sample policy,
- tuning budget,
- kill, Stage B, and scale-up criteria.

Offline proxies must be labeled as proxy or smoke evidence. Simulator rollouts must be labeled with their scope and cannot support claims outside the predeclared protocol.

Failure classifications must be precise. `DATA_OR_SUPERVISION_FAILURE`, `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, and `CONDITION_TOO_SEVERE_OR_NO_HEADROOM` inform the next design; they are not scientific kills of a full research family. `GENUINE_METHOD_KILL`, `SIMPLE_BASELINE_EXPLAINS_METHOD`, and `KEY_COMPONENT_NOT_USEFUL` require a valid mechanism and a completed frozen Stage B or a valid catastrophic Stage A.

## Resource And Safety Policy

Run a short risk assessment before downloads, GPU jobs, training, dataset setup, simulator setup, rollout, heavy imports, or cross-backbone execution. Proceed autonomously when:

- the source is official/documented/unambiguous;
- no token, secret, login, payment, or license click-through is needed;
- disk, RAM, VRAM, runtime, and dependency risks are inside the current budget;
- the action is local, reversible, and does not make an unsupported claim.

Stop before:

- token/secret/API-key access,
- paid services,
- license click-through,
- external upload/submission/publishing,
- deleting user files outside approved cache/repo cleanup,
- system-wide CUDA/PyTorch/driver changes,
- credentialed or system-driver setup,
- unsupported paper-level empirical claims.

OpenVLA-OFT INT4 is not universally prohibited. Quantized OpenVLA-OFT INT4 execution may be used only when a current risk assessment, local assets, and current governance allow it. Full-precision or fine-tuning paths require their own explicit risk budget.

## Prototype Governance

Stage A estimates direction, detects catastrophic harm, and validates the mechanism. It is not a permanent scientific kill for one- or two-episode differences. Permanent scientific kills require the statistical rules in `reports/current_research_governance.md`.

Future first serious prototypes should normally use exactly five policies: Base, closest external prior or faithful transparent proxy, Ours, key ablation, and one strongest simple reviewer-killer baseline. Additional internal controls require a concrete reviewer objection and must be cheaper than moving to the prior comparison.

Before expensive training or rollout, require a pre-experiment headroom and data audit, a bounded validation search budget, a mathematical objective audit, and an identity-preserving integration audit. The default validation search budget is no more than six total configurations, two seeds per lightweight configuration, two architecture choices, and three values for one critical coefficient. Save all tried configurations and negative results.

After a failed method:

1. archive the implementation and results;
2. record the exact failed assumption;
3. do not rescue the current formulation through cosmetic tuning;
4. pivot automatically to a method that changes enough core dimensions.

After three related non-GO methods, synthesize failures, increment the epoch, change at least two core dimensions, and continue. There is no finite global method-cycle limit.

## Required Validation

Use the explicit Python interpreter:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe
```

Before governance commits, run:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\check_current_research_governance.py
git diff --check
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
```

For method commits, add task-specific tests, compile checks, risk reports, rollout/training logs, and result artifacts as appropriate.

For future long-running WSL experiments, use detached durable execution. Save PID, heartbeat, stdout/stderr logs, partial result, final result, exact resume command, and a missing-key-only resume policy.
