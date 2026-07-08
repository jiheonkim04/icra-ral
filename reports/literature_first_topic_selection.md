# Literature-First Topic Selection Memo

Status: memo only. No experiment, training, rollout, download, GPU use, OpenVLA-OFT, new diagnostic, or new method implementation occurred for this memo.

## Steering Change

The process was drifting toward implementation-first topic testing. Stop that pattern. Future topics must be selected from literature gaps and simple-baseline risk first, then converted into an evidence contract before any code.

ActionMap status: the mini-gate implementation had already been merged into local `main` before the stop steer arrived. It has not been pushed. No further ActionMap code, diagnostics, reproduction, extension work, or failure mining should start from this point.

## Selection Criteria

Each candidate is scored against:

- novelty against newest relevant papers/preprints,
- what those papers still do not solve,
- methodological novelty rather than benchmark-only observation,
- ability to support strong experiments across multiple datasets/models,
- SOTA or defensible sub-SOTA axis,
- risk of simple-baseline kill,
- whether a first executable experiment is possible within 24-48 hours,
- realistic RA-L stability.

## Candidate Scores

| Candidate | Novelty | Unsolved gap | Experiment strength | Baseline-kill risk | 24-48 hour first gate | RA-L stability | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constraint-validated spline VLA action interface | Medium-high | Controller-valid, temporally editable VLA trajectory outputs under contact and replay | Could use LIBERO/RoboSuite first, then multiple backbones | High but testable | Plausible if scoped to replay/control contract | Medium-high | Top candidate |
| Early failure detection with evidence-calibrated stop/retry | Medium | Detection must translate to downstream safe utility | Strong if cross-perturbation and cross-model | High due safety-only/no-repair | Plausible if existing failure traces are available | Medium-high | Runner-up |
| Declarative/procedural disentanglement | Medium | Language/action disentanglement beyond canonicalization | Needs careful held-out language and control | Very high due canonicalization | Unclear | Medium-low | Park |
| Phase-aware continual replay | Low-medium | Latest PHASER already covers phase-aware replay | Needs continual VLA suite | High due PHASER/uniform/nearest-demo | Unclear | Medium | Park |
| Semi-supervised or JEPA VLA learning | Medium | Heavy data/model pretraining | Strong only with large-scale training | Medium | No | High if resourced | Park |
| One-step VLA generation | Low | Latency axis now has a strong high-noise schedule baseline | Could be strong but likely derivative | High | Maybe, but novelty weak | Low-medium | Reject |
| Contact-set geometry injection | Low | Local full contact-set lost to single-point/destination-only | Weak from current evidence | Already killed | No | Low | Reject |
| ActionMap extension | Low for this repo now | Local mini-gate failed simple baselines and collapsed candidates | Full reproduction forbidden | Already killed locally | Stop | Low | Reject |

## Top 2

1. Constraint-validated spline VLA action interface.
2. Early failure detection with evidence-calibrated stop/retry.

## Recommended Topic

Recommended for next literature deep dive, not implementation: `Constraint-Validated Spline VLA Action Interface`.

Reason: it has a recent strong anchor, a real methodological axis, a direct robotics metric path, and a clear baseline gauntlet. It is different from killed routes because the first claim would be controller-valid structured trajectory execution, not an offline action-head proxy, contact geometry injection, paraphrase consistency, symbolic repair, or ActionMap heatmap tuning.

## Evidence Required Before Any Implementation

- A one-page evidence contract naming exact task family, datasets/models, and primary metric.
- Direct replay/control metric first, not only offline action loss.
- Controller-valid action rate and clip-step rate.
- Baselines: mean action, no-repair/raw chunk, clipping, safety-only, diagonal affine, global scale, gripper-only, fixed shift, linear time warp, nearest demo, object-relative retargeting.
- Predeclared kill rule if any simple baseline matches the proposed representation on primary utility or validity.
- Minimum cross-scope plan: one local 24-48 hour gate, then at least two datasets or policy backbones before RA-L claims.

