# Epoch 10B terminal handoff

Terminal state: `EPOCH10B_ICAE_NO_PREDICTIVE_HEADROOM`

Branch: `codex/epoch10b-stage0-paper-continuation`

Stage 0 rollout checkpoint: `4318656822eb45591aa0490b14bd14dc1dbc5381`

## Decision

Epoch 10B reached the frozen Phase D scientific terminal after completing the certified fresh-controller mechanics route and the full development Stage 0 panel. Development checkpoint performance was identifiable, but ICAE did not demonstrate predictive headroom over the frozen equal-input baselines. Under the original decision order, distinguishable performance followed by ICAE gate failure requires `EPOCH10B_ICAE_NO_PREDICTIVE_HEADROOM`.

The decisive machine-readable result is `reports/epoch10b_stage0_result.json` (file SHA-256 `f1f79b7153bfac8fe8bf89ec625a233e40a6b9b0276aeaa1261f8eb6f19be8ab`; canonical payload SHA-256 `6a0fe0baa1271903b098d630ef82558e74c391762c902154929cd013c32c306d`).

## Honest denominators and execution integrity

- Whole-seed lineages: 8.
- Development checkpoints: 16, with steps 30 and 100 treated as nested observations rather than independent lineages.
- Tasks/suites: 4.
- Cached development states: 240 per checkpoint.
- Fresh scored branches: 11,520, comprising nominal, paired-candidate, and unpaired-candidate roles for 3,840 intervention pairs.
- Official closed-loop development episodes: 960 = 16 checkpoints × 4 tasks × 15 common reset seeds.
- Valid and executed official episodes: 960/960; unique episode keys: 960/960; successes: 582.
- Hierarchical bootstrap replicates: 10,000 with frozen seed 20260722.

All expected rollout keys were present exactly once. The completed rows matched the frozen checkpoint lineage, optimizer step, suite cap, task ID, and reset-seed ledger. The runner and host-guard hashes remained frozen. Two initial offline-cache preflight attempts failed before any scientific episode was written; both zero-row monitor records are preserved. The certified local model cache then loaded successfully, and no scientific rollout block required retry.

The final rollout host monitor contains 16 successful 60-row execution batches plus one zero-add completion audit. Peak host RAM was 83.7761%, the 80% soft warning was crossed, the 90% hard stop was not crossed, and WSL swap use was 0 bytes. WSL was shut down after execution.

## Frozen Stage 0 result

Checkpoint performance was distinguishable: the best-minus-worst macro-success point difference was 0.18333, with a frozen grouped-bootstrap 95% interval of [0.01667, 0.40000]. The best checkpoint was `epoch10_rank4_seed_909_step_0100`, and two development quality bands were occupied.

The primary ICAE result nevertheless failed every predictive-headroom threshold:

- ICAE equal-task cross-lineage concordance: 0.48824, below the required 0.60.
- Action-dimension-normalized MSE concordance: 0.56943.
- ICAE gain over normalized MSE: -0.08119, below the required +0.08.
- Grouped-bootstrap probability that the gain is positive: 0.4710, below the required 0.90.
- Total ICAE simulator steps: 980,322 versus a 331,200-step exhaustive rollout denominator, or 2.9599×, above the frozen 0.20× cap.

The negative controls did not reproduce a gain, and ICAE was not clearly dominated by every strong baseline. Those passes do not rescue the failed primary thresholds. Faithful CI-MSE remained `NOT_IMPLEMENTED_NO_PROXY_UNDER_FROZEN_ACTION_PROTOCOL`; no proxy result was substituted.

## Leakage and continuation boundary

Development success labels were opened only as authorized by the certified continuation. Held-out checkpoint actions queried: 0. Held-out prospective outcomes opened: false. Confirmation outcomes opened: false. No prospective freeze, held-out evaluation, confirmation, positive paper claim, or RA-L package was produced.

The terminal closes the named ICAE prospective thesis under the frozen Epoch 10B protocol. It does not claim that all VLA or robot-learning research is impossible. No new endpoint, state selector, branch design, or post-failure rescue is authorized within this campaign.

Campaign state: `reports/epoch10b_campaign_state.json`.

Evidence index: `reports/epoch10b_evidence_index.json`.
