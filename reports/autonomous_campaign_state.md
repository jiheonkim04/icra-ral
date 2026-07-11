# Autonomous RA-L Campaign State

Date: 2026-07-11 KST

Branch: `codex/autonomous-dual-review-ral-research`

Base commit verified on Windows Git: `8dc4de2fdbf576ace8bdf3699d190b761553c1fa`

## Repository Verification

- Windows Git branch at campaign start: `main`
- campaign branch created: `codex/autonomous-dual-review-ral-research`
- expected handoff commit present on `main`: yes
- Windows Git working tree before campaign edits: clean
- WSL runtime visibility after reboot: callable
- WSL Git caveat: WSL reports a massive line-ending/file-mode dirty tree while Windows Git remains clean. Repository status, staging, commit, and final status are therefore managed with Windows Git only.

## Required Handoff Files

The required handoff reports were checked. One name drift was found:

- requested: `reports/official_closed_loop_failure_summary.md`
- current equivalent used: `reports/official_libero_closed_loop_failure_summary.md`

Key evidence preserved from the handoff:

- official closed-loop SmolVLA scaleup: frozen base `74/100`, LoRA seeds `74/100`, `68/100`, `66/100`
- quantized OpenVLA-OFT INT4 hard slice: `20/20` successes
- matched SmolVLA hard slice: `11/20` successes
- ECHO final headroom: no oracle downstream improvement on `96` official policy candidates and `96` structured diagnostic candidates
- local ActionMap mini-anchor: oracle candidate headroom existed, but learned local candidate head lost to mean-action and cheap MLP and collapsed

## Fresh Literature Sources

Primary sources checked during this batch:

- CAC-VLA: https://arxiv.org/abs/2607.04816
- ACoT-VLA: https://arxiv.org/abs/2601.11404
- Set-Supervised Diffusion Policy: https://arxiv.org/abs/2606.01865
- BORA: https://arxiv.org/html/2605.30226
- Action-Effect Memory: https://arxiv.org/abs/2606.12499
- TORL-VLA: https://arxiv.org/html/2606.09337v3
- LaRA-VLA: https://arxiv.org/html/2602.01166v1
- LARA: https://arxiv.org/html/2606.07100v1
- LAWM: https://arxiv.org/html/2509.18428v2
- Pre-VLA: https://arxiv.org/abs/2605.22446
- ActionMap: https://arxiv.org/abs/2606.06904
- VLA-Corrector: https://www.alphaxiv.org/abs/2607.01804
- SEAM: https://arxiv.org/html/2607.04609v1
- VLA datasets/benchmarks survey: https://arxiv.org/html/2604.23001v1

## Resource Accounting

- new downloads: `0 GiB`
- active GPU time in this batch: `0 h`
- training in this batch: false
- simulator rollout in this batch: false
- code implementation in this batch: false

No implementation was launched because the three remaining distinct mechanism families were killed before prototype under the governance rule: when Reviewer B finds novelty collapse, simple-baseline dominance, or missing local feasibility, the correct action is KILL rather than another rescue loop.

## Current Stop Candidate

`NO_METHOD_AFTER_3_VALID_CYCLES`

This is a campaign stop candidate, not paper readiness. It means that three genuinely distinct method families were reviewed against current literature plus local evidence and each failed before a valid implementation gate.

