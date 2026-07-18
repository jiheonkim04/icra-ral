# CVLR-XVLA Focused Overlap and Feasibility Decision

- Decision: `CVLR_XVLA_SCIENTIFICALLY_DISTINCT_AND_LOCALLY_EXECUTABLE`
- Contribution type: `PRIOR_EXTENSION`
- Replacement candidate generated: `false`
- CVLR training or rollout: none.

## Closest external prior

The closest prior is [WristWorld](https://arxiv.org/abs/2510.07313). Its direct overlap is the anchor-to-wrist problem, paired multi-view supervision, and use of recovered wrist information to improve a VLA. The paper's actual mechanism is nevertheless different: VGGT plus wrist-pose/point-cloud reconstruction and SPC loss condition a video generator that synthesizes wrist-view pixel sequences for data augmentation. It reports a `3.81%` CALVIN average task-completion-length increase and closure of `42.4%` of the anchor-wrist gap.

CVLR does not generate pixels or video, estimate wrist pose, reconstruct 4D geometry, or augment a VLA dataset. It predicts the existing X-VLA Florence2 wrist latent block from the current agent latent, language, and proprioception, and replaces only that auxiliary token block when the live wrist view is missing. The clean path returns the original auxiliary tensor unchanged. This is not WristWorld's exact claimed contribution.

Secondary checks against [MV-MWM](https://arxiv.org/abs/2302.02408), [RPT](https://arxiv.org/abs/2306.10007), and [ReconVLA](https://arxiv.org/abs/2508.10333) found related view masking, latent prediction, and reconstructive VLA supervision, but no exact missing-wrist-latent insertion into a frozen VLA action path.

## Local feasibility

Nine clean records spanning all three frozen tasks and demos `0/20/40` exposed finite, nonzero wrist targets of shape `[50, 1024]`. X-VLA's `aux_visual_inputs` is `[1, 100, 1024]`, and its first 50-token auxiliary block matched the wrist tokens exactly on every record. `model.transformer.forward` explicitly accepts this tensor, so the insertion point is accessible without modifying official X-VLA source.

The probe used `3520.39 MiB` peak VRAM. The planned `422,144`-parameter predictor and roughly `14.1 MiB` cached train/validation latents fit the RTX 5080 budget without downloads or model offload. The no-training probe and complete rows are in `reports/cvlr_xvla_feasibility_probe_result.json`.

CVLR is therefore eligible for exactly one preregistered Stage 0 protocol. No broad candidate search, RIFA reopening, training, checkpoint write, or rollout occurred during this decision.
