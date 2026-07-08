# SafeTrace-VLA Kill Summary

Decision: kill SafeTrace-VLA as the current main RA-L route before STATE 2.

## Original Hypothesis

SafeTrace-VLA proposed that temporal safety monitors could generate preference pairs, such as safer trajectory/action chunk over unsafe trajectory/action chunk, and that temporal safety preference optimization would reduce temporal safety violations while preserving task utility better than safety filters or generic DPO.

The intended novelty was method-level temporal safety preference construction, not another local wrapper, runtime gate, or benchmark-only observation.

## Strongest Positive Evidence

- A bounded STATE 1 source audit identified official safety anchors: SafeManip, LIBERO-Safety, and ForesightSafety-VLA.
- Local standard LIBERO HDF5 proxy traces were available and produced temporal safety proxy metrics.
- The diagnostic observed nonzero temporal risk: risk exposure time `0.519444`, cumulative safety cost `939.0`, and temporal violation rate by step `0.519444` over `8` local proxy demos.
- Preference-pair generation worked: `800` valid preference pairs were produced, including `10` nontrivial pairs.
- No download, GPU job, simulator rollout, VLA model load, full fine-tuning, OpenVLA-OFT execution, or paper-grade claim occurred.

## Decisive Negative Evidence

- Safety-only/risk-only preference accuracy was `1.0`.
- Generic DPO proxy accuracy/loss was `1.0 / 0.052119`.
- SafeTrace proxy accuracy/loss was `1.0 / 0.052120`.
- SafeTrace did not beat safety-only, risk-only, or generic preference baselines.
- Official safety sources were identified but were not locally available for benchmark-backed reproduction in this run.
- Task-success labels were unavailable in the sampled local proxy traces, so utility preservation could not be established.

## Exact Kill Criterion Triggered

The baseline robustness gate failed: safety-only/risk-only and generic preference/DPO proxy matched the intended SafeTrace preference objective on generated pairs. Under the predeclared anti-baseline criteria, STATE 2 is blocked.

## Why Safety-Only And Generic DPO Kill The Method Claim

SafeTrace's claim requires more than labeling unsafe traces as worse than safe traces. If a risk-only score already achieves the same pair accuracy, then the method is just a safety monitor or safety filter written as preferences. If a generic DPO-style preference proxy matches the same labels and loss, then the temporal component has not shown method-level value beyond ordinary preference optimization.

Because both baselines matched SafeTrace on the local headroom smoke, any later training would test whether a bigger learner can exploit generic monitor labels, not whether SafeTrace contributes a distinct utility-preserving temporal safety objective.

## Reusable Artifacts

- `tca_map/safetrace_vla/diagnostic.py`: source audit, local HDF5 temporal monitor proxies, preference-pair headroom metrics, and safety-only/generic preference comparison.
- `scripts/220_safetrace_vla_diagnostic.ps1`: bounded runner that refuses downloads, GPU, rollout, simulator, heavy import, OpenVLA, and OpenVLA-OFT gates.
- `tests/test_safetrace_vla_diagnostic.py`: fixture-backed tests for metric generation and kill behavior.
- `reports/safetrace_vla_state1_result.md` and `reports/safetrace_vla_state1_result.json`: auditable STATE 1 result.
- SafeTrace source-audit notes for SafeManip, LIBERO-Safety, and ForesightSafety-VLA.

## Why Not RA-L-Stable

The route is not RA-L-stable because it failed before official benchmark reproduction, did not establish utility retention, and did not separate from safety-only or generic preference baselines. A publishable safety-preference method would need official benchmark-backed safe-success or temporal-risk metrics and a clear gain over risk-only filtering and generic DPO. SafeTrace currently has neither.

Execution boundary for this archive: documentation only. No new experiment, replay, rollout, training, loss computation, GPU job, download, heavy VLA import/model load, OpenVLA-OFT execution, new method topic, or paper claim occurred.

