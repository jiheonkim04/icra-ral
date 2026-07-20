# Epoch 7 focused method-overlap and supervision audit

Decision date: 2026-07-20

Status: `INCREMENTAL_BUT_DEFENSIBLE_WITH_STRONG_EVIDENCE`.

Selected mechanism: **equivalence-selective action-energy ranking**. This decision authorizes a frozen Stage-0 diagnostic and, only if that diagnostic exposes nontrivial headroom, a bounded real X-VLA training audit. It does not authorize confirmatory identities or a paper claim.

## Executable mechanism

For a real demonstration tuple containing observation `o`, factual instruction `l+`, and real X-VLA-format action chunk `a`, sample (i) a verified LIBERO-Para equivalent `l~` for the same task and (ii) a distinct, feasible LIBERO-Goal instruction `l-` from the same declared world. Let X-VLA's released clean-action denoiser be `f_theta`. With a fixed bank of diffusion times and noises,

```text
x_t = t epsilon + (1 - t) a
E_theta(o, l, a) = mean_(t,epsilon) L_XVLA(f_theta(x_t, o, l, t), a)
L_XVLA = 500 MSE(position) + 10 MSE(rotation-6D) + BCE(gripper)
E_pos = 0.5 E_theta(o, l+, a) + 0.5 E_theta(o, l~, a)
L_rank = softplus((margin + E_pos - E_theta(o, l-, a)) / temperature)
L_total = E_pos + lambda_rank L_rank
```

This matches the official X-VLA `ee6d` implementation: its training forward constructs `x_t = t*noise + (1-t)*action`, predicts the clean action, and applies position MSE scaled by 500, rotation-6D MSE scaled by 10, and gripper BCE over the padded 20D single-arm representation. It is not described as a velocity-field loss or a generic uniform MSE.

The negative instruction is never paired with a fabricated correct action and is never supervised as the target of `a`. It only requires the factual real action to have lower denoising energy under its factual/equivalent intent than under a different feasible intent. Negative ranking is restricted to a frozen early, pre-interaction window where all ten Goal intents remain feasible in the shared world. At deployment the trained X-VLA receives the ordinary images, proprioception, and one instruction and executes its ordinary single conditional denoising branch; there is no router, retrieval model, symbolic planner, vision-only branch, or privileged simulator input.

The causal path is direct: the ranking term changes gradients through X-VLA's instruction-conditioned action denoiser; the changed denoiser changes generated action chunks; those chunks control the official simulator and can change official task success.

## Real-supervision legality

The retained raw LIBERO corpus is complete enough for the supervision-legality audit: 130 HDF5 files, 100,442,942,572 bytes, including all ten LIBERO-Goal task files. The Goal suite contains 500 demonstrations and 63,728 frames. Each task has 50 demonstrations; every inspected numeric array is finite; all ten aggregate action arrays are noncollapsed. The ten Goal BDDLs share an exact world signature over regions, fixtures, objects, and initialization, while declaring ten distinct goals. Consequently, each factual sample has nine distinct declared Goal instructions available as feasible early-window negatives. Actual X-VLA training/energy execution instead uses the authors' separately released `2toINF/Libero-XVLA-format` Goal subset at revision `27ddd36538ee4812bd31fd8b494f8d7c6a11ef9d` (428 converted demonstrations, 1,899,116,312 bytes), eliminating local action-conversion ambiguity.

LIBERO-Para supplies 4,092 declared equivalent instructions: 870 action, 2,963 compositional, and 259 object variants. Equivalent positives reuse the real demonstrated action chunk. Pair construction is deterministic and outcome-free. No simulator success, Base/Prior/Ours result, or future checkpoint performance may select the pairs.

Legality boundary:

- Allowed: factual and declared-equivalent text with the same real action target; a different feasible instruction used only as an incompatibility ranking negative.
- Prohibited: inventing the action that should satisfy `l-`; treating `a` as correct for `l-`; selecting negatives or early-window length using Ours outcomes; simulator state at inference.
- Required before training: use the authors' exact released X-VLA-format demonstrations, verify their loader semantics, and freeze hashes for train/validation/confirmatory partitions.

The machine-readable source audit is `selectivity_supervision_audit.json`; the final LIBERO-CF runtime preflights are `libero_cf_artifact_preflight.json` and `libero_cf_ood_artifact_preflight.json`.

## Focused primary-work overlap matrix

| Work | Positive supervision | Equivalence mechanism | Between-intent mechanism | Inference | Difference from selected mechanism |
|---|---|---|---|---|---|
| RobustVLA ([paper](https://arxiv.org/abs/2510.00037)) | real VLA training targets | semantics-preserving lexical/syntactic perturbations and action consistency | no explicit factual-action rejection across feasible intents | ordinary or adversarially trained policy | selected method adds feasible-intent ranking and uses denoising energy, not only within-intent consistency |
| RoVLA ([paper](https://arxiv.org/abs/2605.19678)) | real demonstrations | samples synonymous instructions during training; instructional/flow/observation consistency | no explicit alternative-intent ranking | ordinary policy | selected method's irreducible term is between-intent real-action compatibility; `lambda_rank=0` is the matched local paraphrase-only ablation |
| CAST ([paper](https://arxiv.org/abs/2508.13446), [code](https://github.com/catglossop/CAST)) | real data plus generated branches | not the main claim | creates counterfactual instructions and synthetic counterfactual actions with a VLM and atomic policy | trained policy | closest conceptual overlap, but selected method fabricates no counterfactual action target, ranks only a real factual action, and needs no generation teacher |
| CAG ([paper](https://arxiv.org/abs/2602.17659), [code/LIBERO-CF](https://github.com/yuffish/LIBERO-CF)) | none for guidance | none | contrasts language-conditioned and language-unconditioned action predictions | two action branches plus guidance | selected method is training-time single-policy discrimination with ordinary one-branch inference; it is not action guidance |
| RSS ([paper](https://aclanthology.org/2026.acl-long.190/)) | no new action targets for the guidance rule | dense paraphrase neighborhood integration | residual subtraction of visual affordance prior, not explicit feasible-intent labels | multiple language/visual-affordance evaluations | selected method uses real demonstration gradients and explicit intent negatives, with no inference ensemble |
| ProGAL-VLA ([paper](https://arxiv.org/abs/2604.09824)) | imitation plus planner/3D grounding supervision | not paraphrase-equivalence training | entity-level contrastive alignment and verified symbolic goal | planner, 3D entity graph, and learned policy | selected method has no planner, 3D graph, entity labels, or symbolic verification and contrasts action compatibility rather than entity embeddings |

The selected method differs from every closest primary prior by at least two major dimensions, but its ingredients—paraphrase positives, contrastive ranking, and action reconstruction energy—are individually established. The honest classification is therefore `INCREMENTAL_BUT_DEFENSIBLE_WITH_STRONG_EVIDENCE`, not `DISTINCT_METHOD_CONTRIBUTION`.

## Candidate adjudication

### M1 — equivalence-selective action-energy ranking: selected

- Base-preserving path: zero-initialized LoRA on the released X-VLA checkpoint; no deployment-time auxiliary component. The exact trainable module set must be frozen after a one-batch gradient audit.
- Key ablation: identical positive canonical/paraphrase reconstruction with `lambda_rank=0`; this is a local RoVLA-IC-style proxy, not an official RoVLA reproduction.
- Simple control: the frozen 22.7M-parameter all-MiniLM-L6-v2 canonicalizer already measured at 25/30.
- External Prior: the corrected, mechanism-faithful CAG-TF port, retained as a negative relevant comparator because it is action-connected but not competent locally.
- Generalization: official LIBERO-CF counterfactual tasks, whose author artifact passed serial spatial and OOD environment preflights. LIBERO-CF releases no X-VLA integration, so any X-VLA evaluation must be labeled a matched adaptation.
- Cheap falsifier: frozen Base action-energy discrimination on held-out early real chunks. Kill if paired energy accuracy is at chance/no better than instruction-independent scoring, if action energies collapse, or if the ranking gradient does not reach trainable instruction-to-action parameters.
- Main risk: early actions can be shared across feasible goals, allowing no honest discriminatory signal. A positive offline diagnostic is necessary but not sufficient; official closed loop remains mandatory.

### M2 — learned semantic router/canonical prompt: rejected

This is functionally indistinguishable from the already executed MiniLM canonicalizer and generic retrieval/prompt normalization. It cannot support a method claim even if trained. Classification: `TOO_OVERLAPPING_FOR_RAL_METHOD_CLAIM`.

### M3 — learned language amplification or dual-branch guidance: rejected

Conditional/unconditional amplification is directly occupied by CAG; dense-neighborhood residual guidance is occupied by RSS; token/attention recalibration is close to IGAR and ProGAL-VLA. Adding a learned gate would not create two major distinctions and risks the canonical-retention failure already observed for CAG-TF. Classification: `TOO_OVERLAPPING_FOR_RAL_METHOD_CLAIM`.

## Stage-0 authorization boundary

The next authorized empirical sequence is:

1. Freeze deterministic demonstration, frame-window, paraphrase, negative-intent, time, noise, and model partitions without reading Ours outcomes.
2. Verify raw LIBERO observations/actions can be converted exactly into the official X-VLA LIBERO training/inference convention.
3. Run the unmodified Base energy falsifier on frozen validation chunks.
4. Only if discrimination headroom is nontrivial, run one real CUDA microbatch with Full and `lambda_rank=0`, proving nonzero gradients, changed weights, reload identity, finite legal actions, Full-versus-Ablation difference, and canonical retention.

No closed-loop Ours or confirmatory identity is authorized until Stage 0 returns `STAGE0_METHOD_GO` or a genuinely underpowered fixed expansion under the frozen rule.

## Internal review

### Executor

Ran an offline metadata/action audit over ten Goal HDF5 files and all 4,092 LIBERO-Para rows; cloned and preflighted the official MIT-licensed LIBERO-CF artifact at revision `8460457bfca6e0ef2e856bc104e2c60b023ef2a7`; constructed and reset one spatial and one OOD author-task environment; executed ten finite dummy simulator actions in each; inspected no success, reward, or done outcome; loaded no VLA; performed no training.

### Skeptical reviewer

CAST is conceptually close because it also seeks observation-conditioned instruction/action selectivity. The proposed loss is a synthesis of known tools, and the five-pair residual over the semantic control is small. Early-window action compatibility may be unidentifiable when several goals begin with similar reaches. LIBERO-Para equivalence labels are released metadata rather than human re-verification in this campaign. LIBERO-CF is an author artifact but lacks an official X-VLA adapter. Simulation-only findings cannot establish physical-robot robustness.

### Adjudicator

- Novelty: positioning issue and high evidence burden, not yet fatal.
- Supervision legality: pass, conditional on the frozen early-window restriction and no fabricated negative action.
- Local executability: pass for Stage-0 preflight; training still depends on exact X-VLA-format materialization.
- Leakage risk: repairable by a hash-bound partition manifest before Ours.
- Comparator risk: semantic canonicalizer is strong; CAG remains a correctly implemented but empirically weak external Prior.
- Decision: authorize only the selected mechanism's frozen Stage-0 path. Kill this mechanism if the Base energy falsifier or real-gradient audit fails; do not rescue it as augmentation alone.
