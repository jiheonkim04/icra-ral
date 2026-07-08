# TG-VLA Risk Register

Date: 2026-07-09 KST

| Risk | Severity | Evidence | Mitigation | Current status |
| --- | --- | --- | --- | --- |
| Novelty collapse into direct 3D target action-head injection | High | A June 2026 paper already lifts a grounded target point to 3D and injects it into the VLA action head via AdaLN with large LIBERO-PRO gains. | TG-VLA must emphasize instruction-resolved object priors, non-leaking target resolution, paraphrase consistency, and counterfactual sensitivity. Include single-point injection baseline. | Blocking for naive training. |
| Canonicalization-only dominates | High | Prior PRISM-VLA local diagnostic was killed because canonicalization-only beat the proposed method on held-out paraphrase and PRIDE metrics. | Canonicalization-only must be a primary baseline; do not train TG-VLA until metrics and split prevent lexical normalization from explaining the gain. | Blocking for naive consistency objective. |
| Standard LoRA or LoRA-SP explains gain | High | LoRA-SP reports that VLA transfer can need adaptive/higher-rank capacity and can outperform standard LoRA. | Compare against standard LoRA/action imitation and later LoRA-SP-style adaptive capacity if scaling. | Open. |
| PEFT/QLoRA tooling missing locally | Medium | Current environment has `peft=false` and `bitsandbytes=false`. | Either add a risk-assessed dependency task later or implement a tiny local LoRA wrapper with tests. | Blocks off-the-shelf LoRA/QLoRA. |
| Leakage through LIBERO metadata | High | Local BDDL and filenames expose targets. | Target signal must be derived from instruction text and visible/object names only; BDDL/filenames allowed only for training supervision or evaluation bookkeeping. | Must be audited before training. |
| Offline proxy mistaken for real VLA evidence | High | Many prior routes were killed for proxy-only evidence. | STATE 2 must use real SmolVLA model load/forward/adapter path and label any offline metric as smoke only. | Active constraint. |
| RTX 5080 16GB OOM | Medium | SmolVLA estimate is about 12GB load plus 2GB headroom; OpenVLA-OFT training exceeds local memory. | Batch size 1, rank 4, frozen backbone, stop on OOM, no OpenVLA-OFT. | SmolVLA feasible but tight; OpenVLA-OFT blocked. |
| No rollout/control evidence | Medium | STATE 1 only checks assets and feasibility. | Do not make RA-L claims before simulator or rollout evidence. | Not addressed in this run. |
| Target prior too weak without object detector | Medium | Instruction-only object extraction may not identify visible candidate binding. | Start with visible object names if available; do not use oracle eval targets except upper bound. | Needs implementation design. |
