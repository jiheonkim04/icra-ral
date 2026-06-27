# Reviewer #2 Risk Report: No Large OpenVLA-OFT Reproduction

## Can This Publish Without Direct OpenVLA-OFT Large Reproduction?

Possibly, but only with a narrower claim.

A paper can still be viable if it clearly presents TCA-Map as a low-compute target-grounding/action-decoding improvement over native SmolVLA heads, ActionMap, and ActionMap plus counterfactual augmentation. It should not claim broad SOTA over OpenVLA-OFT unless OpenVLA-OFT is directly reproduced under comparable benchmark conditions.

## What Claim Is Still Valid?

Valid claim:

> TCA-Map improves target-conditioned action decoding under strict compute constraints, preserving standard performance while improving counterfactual target grounding over ActionMap and native heads.

This requires evidence that:

- TCA-Map beats native SmolVLA head.
- TCA-Map beats ActionMap.
- TCA-Map beats ActionMap + counterfactual augmentation.
- Offline robustness gains transfer to at least a small simulator rollout.
- No privileged simulator state is used at default inference.

## What Claim Is Invalid?

Invalid claims without direct evidence:

- OpenVLA-OFT SOTA.
- Full VLA SOTA.
- Real-world deployability.
- Standard manipulation success based only on offline proxy metrics.
- Language grounding is solved.
- OpenVLA-OFT training or rollout performance inferred from frozen smoke.

## What Baseline Will Kill Us?

The strongest paper-killing baseline is:

- ActionMap + counterfactual augmentation on the same SmolVLA frozen features and same tiny/rollout splits.

If that baseline matches TCA-Map on target accuracy, wrong-target rate, and counterfactual success while preserving standard performance, the novelty collapses into data augmentation plus heatmaps.

Other dangerous baselines:

- Native SmolVLA head with carefully tuned counterfactual data augmentation.
- Target-head-only variant without target-conditioned action heatmaps.
- Oracle target-conditioned head, if the gap between oracle and TCA-Map is too large.

## What Result Is Needed To Justify Publication?

Minimum convincing result:

- Standard rollout metric degradation no worse than 1-2 percentage points versus ActionMap or native head.
- Counterfactual/robust target grounding improvement at least +10 percentage points.
- Wrong-target rate reduction at least 20 percent relative.
- TCA-Map beats ActionMap + counterfactual augmentation and target-head-only ablations.
- Latency/VRAM overhead is reported and reasonable.
- Training/test counterfactual templates are disjoint.
- Default inference uses no privileged target labels or simulator state.

## What Must Move To WSL2/Cloud/Remote If Local Fails?

Move to WSL2/Linux:

- LIBERO/RoboSuite simulator setup.
- Small rollout validation.
- Rendering/import checks.
- Reproducible rollout scripts.

Move to cloud/remote GPU:

- OpenVLA-OFT full baseline.
- Multi-seed LIBERO/RoboCasa sweeps.
- Large ActionMap/TCA-Map benchmark matrix.
- Full ablations across seeds, datasets, and backbones.
- Any experiment needing more than 14 GB local VRAM or less than 2 GB VRAM headroom.

## Reviewer #2 Bottom Line

The low-compute path is publishable only if it is honest and narrow. The paper must sell TCA-Map as a compute-efficient grounding/action-head contribution, not as a general OpenVLA-OFT SOTA claim. The decisive comparison is not OpenVLA-OFT frozen smoke; it is TCA-Map versus ActionMap and ActionMap + counterfactual augmentation under the same SmolVLA frozen-feature budget.
