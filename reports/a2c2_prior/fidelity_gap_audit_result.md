# Focused A2C2 Fidelity-Gap Audit

Date: `2026-07-19 KST`

Primary decision: `A2C2_OBJECTIVE_FIDELITY_DEFECT_FOUND`

Authorized corrected label: `A2C2_FIDELITY_CORRECTED_LOCAL_PORT`

This was a source/config/checkpoint audit only. No training, rollout, Ours
design, or outcome inspection occurred during the audit. The completed v1
result remains unchanged: `MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`, Base standard
`10/15`, delayed Base `4/15`, delayed local Prior `3/15`, and
`A2C2_PRIOR_NO_LOCAL_IMPROVEMENT`.

## Finding

Three objective defects are established independently of the v1 outcomes:

1. v1 used `pepijn223/smolvla_libero`, not the author's paired
   `k1000dai/smolvla_libero_spatial_scratch` base.
2. v1 asserted that no official correction checkpoint was available and
   trained a 40-episode/40k/frozen-ResNet substitute. The author account in
   fact hosts a public six-layer, 960D-`vlm_hidden` checkpoint and its full
   432-episode/52,970-frame dataset.
3. the frozen official evaluator rotates both live RGB views 180 degrees;
   the installed local LeRobot wrapper used by v1 does not rotate either live
   view.

These are not post-hoc explanations for `3/15`. They follow directly from the
[paper](https://arxiv.org/abs/2509.23224), the frozen
[official repository](https://github.com/k1000dai/a2c2-libero/tree/54dd088302a0ef3f50c4add3ec927ab94d76a406),
the [pre-redaction source commit](https://github.com/k1000dai/a2c2-libero/commit/2275a8b3ae54141480fa8cca9a267296bdc56f82),
the author's [paired Spatial base](https://huggingface.co/k1000dai/smolvla_libero_spatial_scratch),
the [paper-structural-match residual checkpoint](https://huggingface.co/k1000dai/residual_transformer_libero_spatial_add_vlm_context),
and its [public residual dataset](https://huggingface.co/datasets/k1000dai/libero-spatial-smolvla-add-vlm-context).

## Row-by-row audit

| Area | Official/published behavior | v1 local behavior | Exact mismatch | Expected effect | Outcome-free repair | Rerun scope |
|---|---|---|---|---|---|---|
| Base identity | Author Spatial-scratch base | Third-party 40-task base | Different weights, corpus, config, latent width | Material base/prior incompatibility; sign unknown | Pin author base revision/hash | New matched Base/Prior panel |
| Base recipe | Spatial-only, 100k, batch 64 | General LIBERO, 25k, batch 32 | Different training path | Different chunk/latent distribution | Use uploaded base | New matched panel |
| Views/resolution | Top+wrist, 256×256 | Top+wrist, 256×256 | None | None | None | None |
| Live RGB orientation | Rotate both views 180° | No rotation | Wrong live observation orientation | Material visual distribution shift | Apply exact official rotation | New matched panel |
| State/action | 8D state, 7D relative action | 8D live state, 7D relative action | None | None | None | None |
| Language/base latent | SmolVLM language/base latent; public prior expects 960D | 576D latent from another base plus released task scalar | Wrong latent source/width | Core feature incompatibility | Use paired base/prior | New matched panel |
| Normalization | Full Spatial residual-dataset stats stored with checkpoint | Local 40-task data stats | Different normalization and base-action distribution | Changes residual scale/action | Load checkpoint buffers | New matched panel |
| Delay definition | `d` control steps; `d <= e <= H-d` | `d=10,e=40,H=50` | None | None | None | None |
| Queue | First `[0,e)`, retain `[e,e+d)`; later old `d` + new `[d,e)` | Same slices | None; no off-by-one found | None | Equivalence test | Tests only |
| Future actions | Retained already-generated base actions; no live expert | Same | None | None | None | None |
| Reset | Official init state + ten dummy steps | Wrapper performs same ten dummy steps | None | None | Reset-equivalence test | Tests only |
| Latest observation | Correction every control step | Same | None except RGB orientation | None beyond RGB defect | Forward-count test | Tests only |
| Base action/chunk | Selected action + source 50×7 chunk | Same | None | None | Tensor-identity test | Tests only |
| Phase | Released code uses `2πk/(H-1)` | Same | None versus code; paper writes `H` | None against released implementation | Preserve code path | None |
| Target | Normalized expert-minus-stale-base residual | Same | None | None | None | None |
| Residual integration | Add normalized residual, then unnormalize | Same | None | None | Equivalence test | Tests only |
| Loss/gripper | MSE; no extra clamp/clip/discretization | Same | None | None | None | None |
| Data coverage | 432 episodes, 52,970 frames, 10 tasks | 40 episodes, 2,438 cached rows | Large coverage reduction | Weaker phase/task/reset coverage | Use uploaded checkpoint | No retraining |
| Offset sampling | Every frame anchor; online valid offset | Stride-8 anchors; four fixed offsets | Different supervision weighting | Reduced temporal coverage | Use uploaded checkpoint | No retraining |
| Optimizer budget | Batch 64; paper 200k, upload 400k | Microbatch 4; 40k | Much smaller and source step-count conflict | Material optimization difference | Do not choose; use uploaded weights | No retraining |
| ResNet trainability | Uploaded paper-aligned config is unfrozen | v1 freezes ResNet | Different trainable graph | Different visual representation/capacity | Use uploaded weights | No retraining |
| Checkpoint availability | Public immutable author checkpoints exist | v1 recorded none available | False asset premise | Strongest executable author path untested | Hash-verify public pair | New matched panel |
| Module placement | Per-step additive plug-in; Base unchanged when off | Same separated Base/Prior arms | None | None | Base-equivalence test | Tests only |
| Evaluation scale | All 10 tasks; 10 or 50 rollouts/task | 3 tasks × 5 resets | Smaller local verification | More uncertainty; no sign inference | Keep local scope honest | New 3×5 identities |
| Condition/horizon | `e=10,d=0`; `e=40,d=10`; H=50; 220 steps | Same | None | None | None | None |
| Success/seed | Official `check_success`, official states, seed 7 stream | Same success/states; paired deterministic chunk noise | RNG stream differs, comparator pairing valid | Individual trajectories may differ | Freeze one paired schedule | New identities |
| Published result | Table: 89.2 vs 81.8 clean; 84.2 vs 64.4 delayed | 10/15, 4/15, 3/15 locally | Scale/result differ | Descriptive only | Do not tune old outcomes | New matched panel |

The machine-readable report records the exact source hashes, artifact
revisions, checkpoint LFS SHA-256 values, and evidence references.

## Release inconsistencies and bounded resolution

The public release is not internally exact enough to justify a retraining
claim: the paper says 200k correction updates while the uploaded config says
400k; the pre-redaction evaluator names a 10-layer no-`vlm_hidden` checkpoint
while the paper specifies six layers with base latent/language input; and the
paper describes a 512-wide MLP while released code uses 1024-wide hidden
layers.

The correction is nevertheless unique without choosing a training recipe:
do not retrain. Pin the author's paired Spatial base and the only inspected
public residual artifact matching the paper's six-layer plus 960D-base-latent
structure, load their immutable weights/statistics, and use the official live
image and queue path. Because released code and prose still differ, the result
must remain `A2C2_FIDELITY_CORRECTED_LOCAL_PORT`, never
`OFFICIAL_A2C2_REPRODUCTION`.

## Gate consequence

One and only one corrected A2C2 execution is authorized. It must use new
development and verification identities, preserve the old 45 rows, freeze
focused equivalence tests before rollout, and compare corrected Prior against
its matched corrected Base. A second fidelity-correction iteration is
forbidden. Additional-Prior search and Ours remain unauthorized until the
corrected decision is known.
