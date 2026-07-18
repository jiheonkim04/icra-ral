# Strategic Pivot Epoch 1 Selection

Decision: `PIVOT_SELECTED`

Selected thesis: **VLA action-chunk reactivity under asynchronous inference delay** (`ASYNC_DELAY_REACTIVITY`). This is a research thesis, not a method acronym.

Execution type: `REPORT_ONLY`. No model forward, training, rollout, confirmatory access, Ours design, or physical experiment occurred during selection.

## Authoritative boundary

- Branch and pushed HEAD: `codex/epoch5-official-prior-first` at `f11ec2135f39dba3ccf315ee5271aa152247cff7`.
- Wrist-dropout method-development axis: `CLOSED`.
- RIFA, CVLR, and action-consistent missing-view outcomes remain unchanged.
- `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT` remains the only valid RL4IL label.
- Active worker at audit start: none.

The selection directive was read from `C:/Users/jiheo/Downloads/strategic_pivot_to_paper_autonomy_steer.md`, 961 lines, SHA-256 `FCFDE6371541CDB635F1B2D660A80379D227F2B0D32C14B38BBDD9BE7FFD68CC`.

## Exactly three candidate theses

### 1. Counterfactual instruction faithfulness

- Problem: a VLA executes the visually familiar LIBERO task instead of a plausible alternative instruction in the same scene.
- Closest prior: [LIBERO-CF and Counterfactual Action Guidance](https://arxiv.org/abs/2602.17659); [official code](https://github.com/yuffish/LIBERO-CF) at `8460457bfca6e0ef2e856bc104e2c60b023ef2a7`.
- Official protocol: condition-specific touch and task success across counterfactual spatial, object, long, focused, and OOD suites.
- Base/Prior: locally runnable X-VLA or OpenVLA-OFT INT4 versus training-free CAG.
- Residual hypothesis: CAG improves grounding but leaves task- and phase-dependent faithful-success residuals.
- Headroom: the official release reports OpenVLA-OFT faithful success `0.4%` versus biased success `78.6%` on average.
- Legal inference: current images, proprioception, instruction, a fixed empty-language branch, and policy outputs only.
- Local path: sequential two-branch inference; no physical robot.
- Novelty opportunity: only a mechanism beyond final-action CAG and archived pathwise LIFT could proceed.
- Reviewer objection: direct overlap with CAG, LIFT, LCG, and the historical language/target route.
- Stage 0 falsifier: no repeated Base failure, CAG saturation, or a residual explained by canonicalization/ordinary adaptation.
- Stage A path: paired official LIBERO-CF closed-loop evaluation after a verified CAG residual.
- Pareto/generalization: lower-cost single-branch deployment; held-out LIBERO-CF or LIBERO-Para; optional camera-only instruction sensitivity after positive simulation.
- Expected decisive time: `8` active hours.
- Status: `NOT_SELECTED_HIGH_ARCHIVED_OVERLAP`.

### 2. Asynchronous-delay action reactivity

- Problem: while a new action chunk is inferred, the policy executes actions generated from stale observations; delay and execution horizon degrade closed-loop reactivity.
- Closest prior: [A2C2](https://arxiv.org/abs/2509.23224); [official LIBERO code](https://github.com/k1000dai/a2c2-libero) at `54dd088302a0ef3f50c4add3ec927ab94d76a406`.
- Official path: the repository includes residual-dataset generation, a residual-transformer trainer, and LIBERO evaluation. It does not identify a released residual checkpoint, so local prior training is required and must be labeled accordingly.
- Protocol: official LIBERO Spatial success across fixed inference delays and execution horizons, with matched repeated resets.
- Base/Prior: official SmolVLA-LIBERO checkpoint versus official A2C2 residual correction.
- Residual hypothesis: A2C2 improves fixed-delay performance but leaves task- and delay-dependent residuals under the local consumer-GPU timing regime.
- Headroom: the paper reports `+23 pp` over RTC on dynamic Kinetix, `+7 pp` on LIBERO Spatial, and a remaining delayed-policy gap.
- Legal inference: current RGB/proprioception/instruction, stale base chunk, base hidden feature, chunk index, and measured delay.
- Local path: the existing 865 MiB SmolVLA checkpoint, 11 GiB official environment, cached LeRobot/LIBERO data, RTX 5080, and a lightweight residual transformer.
- Novelty opportunity: only after a verified residual, and never another queue scheduler, generic residual MLP, or renamed EAC.
- Reviewer objection: equivalence to A2C2, RTC, fixed short replanning, ordinary residual learning, or the closed EAC family.
- Stage 0 falsifier: no repeated Base delay gap, no matched A2C2 improvement, or no residual after A2C2.
- Stage A path: paired official LIBERO Spatial Base/Prior/Ours comparisons only after `VERIFIED_PRIOR_RESIDUAL`.
- Pareto/generalization: higher delayed success or matched success at lower correction latency/memory; unseen delay distribution as the second condition.
- Camera-only relevance: optional latency/action-stability measurement only after positive simulator evidence.
- Expected decisive time: `10` active hours.
- Status: `SELECTED`.

This is distinct from archived EAC: EAC changed queue commitment without injected inference delay or a trained current-observation correction head. It is also distinct from TL-ChunkRepair and phase retiming, which edited or retimed existing actions without an executable official A2C2 prior. Those routes remain closed.

### 3. Short-horizon intent aliasing

- Problem: frame-conditioned chunk policies can switch intent when visually similar observations require different actions because of recent context.
- Closest prior: [IntentVLA](https://arxiv.org/abs/2605.14712); [official AliasBench code](https://github.com/ZGC-EmbodyAI/IntentVLA) at `a4c611ee4938df82626e5a4e5c1b38f070bd9a82`.
- Protocol: twelve RoboTwin2 AliasBench tasks across back-and-forth, crossing-path, bimanual, and multi-goal families.
- Prior availability: the official repository releases task code only and states that model code, configs, and evaluation scripts are coming soon. Reported training uses 16 H100 GPUs, 30K steps, and batch 16 per GPU.
- Headroom: reported direct four-frame history `10.4%`, uniformly sampled history `28.1%`, IntentVLA `45.8%`.
- Local path: none that is currently fair and decisive; the official prior is not locally runnable.
- Reviewer objection: extensive overlap with archived PSE/MTF/CALA/NICE/HEST/HASTE/TSC history families.
- Expected decisive time: greater than the authorized local boundary.
- Status: `REJECTED_LOCAL_FEASIBILITY`.

## Frozen scoring

`TOTAL = 2N + 2R + 2H + 2F + 1.5P + 1.5C + G + A + D`

| Candidate | N | P | R | H | F | C | G | A | D | Total | Hard filters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Counterfactual instruction faithfulness | 3.0 | 5.0 | 5.0 | 5.0 | 4.0 | 5.0 | 5.0 | 5.0 | 1.5 | 60.5 | pass |
| Asynchronous-delay action reactivity | 4.0 | 5.0 | 4.5 | 5.0 | 4.5 | 5.0 | 4.5 | 5.0 | 3.5 | **64.0** | pass |
| Short-horizon intent aliasing | 4.0 | 4.0 | 4.0 | 5.0 | 1.5 | 4.0 | 5.0 | 5.0 | 2.0 | 53.0 | fail: local Base/Prior/Ours and 12-hour feasibility |

The numerical lead over the language thesis is `5.79%`, so the 10% rule is not met and the explicit tie-break applies: A2C2 has pinned executable official SmolVLA code and a directly measurable local delay surface that the closed EAC scheduler never tested. The language thesis reuses the exact CAG prior already anchoring archived LIFT and faces a materially tighter overlap boundary.

## Next authorized empirical action

Freeze and execute official-prior-first problem verification for `ASYNC_DELAY_REACTIVITY`:

1. verify synchronous SmolVLA competence;
2. verify a repeated delay-induced failure/gap on matched official resets;
3. train and execute the official A2C2 prior faithfully enough to test the same condition;
4. determine whether a residual remains;
5. return exactly one problem-verification decision.

Do not design or execute Ours before `VERIFIED_PRIOR_RESIDUAL`.
