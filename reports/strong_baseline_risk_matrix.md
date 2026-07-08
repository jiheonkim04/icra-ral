# Strong Baseline Risk Matrix

Purpose: block topics that can be killed by obvious controls before implementation.

| Candidate | Highest-risk simple baselines | First evidence required before implementation | RA-L stability |
| --- | --- | --- | --- |
| Constraint-validated spline VLA action interface | no-repair, raw chunk, fixed shift, linear time warp, gripper-only timing, diagonal affine, global scale, object-relative retargeting, mean action | Predeclare replay/control and controller-valid metrics; show spline representation can beat raw chunks and timing/retarget baselines on at least one real task family without weakening success. | Plausible if tested across LIBERO/RoboSuite and at least two policy backbones. |
| Early VLA failure detection plus stop/retry gate | no-repair, always-abstain, safety-only, clipping, nearest-demo, Mahalanobis-only, action-consistency-only | Need early detection AUPRC plus downstream utility: fewer catastrophic failures or better safe-success than no-repair/safety-only at matched intervention budget. | Plausible if evaluated on simulated and real or cross-dataset failure shifts. |
| Declarative/procedural disentanglement for language-conditioned control | canonicalization, nearest target, language template matching, no-language ablation, mean action | Need held-out compositional language failures with preserved object/target sensitivity and replay/control gains, not just paraphrase consistency. | Medium-low due PRISM-VLA canonicalization failure. |
| Phase-aware continual adaptation | uniform replay, nearest-demo replay, PHASER-like phase replay, random replay, no-replay | Must beat latest phase-aware replay literature and retain old tasks while learning new ones. | Medium, but novelty window is narrow. |
| Test-time latent adaptation | no-update, diagonal affine, global scale, safety-only, clipping, prompt-only heuristic, random probe | Need interaction-efficient deployment shift gains with explicit cost and no hidden reward leakage. | Medium if interaction budget is acceptable; risky for first 24-48 hour gate. |
| Semi-supervised/JEPA VLA pretraining | supervised-only, image augmentation, frozen visual encoder, nearest-demo, canonicalization for language | Requires heavy data/model training and multi-dataset evidence. | High scientific upside, low immediate feasibility. |
| One-step VLA generation or latency topic | high-noise schedule, ten-step decoding, OpenVLA-OFT throughput, action chunking | Must beat the newest one-step paper, not just reduce latency. | Low novelty unless paired with a distinct safety or control metric. |
| Contact-set or richer geometry injection | active single point, source-only, destination-only, source+destination, no-geometry, mean action | Already failed locally; require a fundamentally different literature gap. | Low. |
| ActionMap extension or heatmap tuning | mean action, linear/L1, cheap MLP, oracle candidate upper bound, candidate-collapse check | Local mini-gate failed; no extension until a stronger official-style anchor passes simple baselines. | Low for this repo right now. |

