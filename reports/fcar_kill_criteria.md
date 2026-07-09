# FCAR Kill Criteria

Date: 2026-07-09 KST

Kill or pivot FCAR if any of these are true.

## Evidence Gate

- regenerated base/LoRA predictions differ materially from the official routing-gate evidence without a logged reason;
- frame oracle no longer clears `0.005` absolute and `5%` relative headroom over frozen/base;
- task/instruction oracle becomes the only headroom source, because MoIRA-style routing covers that.

## Performance Gate

- FCAR fails to beat frozen/base by `0.005` absolute or `5%` relative action L2;
- FCAR fails to beat standard rank-4 LoRA;
- FCAR fails to beat mean-action prior;
- FCAR fails to beat MoIRA-style instruction/task router;
- FCAR fails to beat static mixture / adapter soup.

## Novelty Gate

- FCAR reduces to task routing;
- FCAR reduces to instruction routing;
- FCAR reduces to generic LoRA expert MoE;
- FCAR can be matched by adapter soup/static merge;
- FCAR's only benefit is action-dimension oracle behavior that cannot be realized fairly.

## Measurement Gate

- official data cannot support a fair train/val/test split;
- gate requires ground-truth action, reward, success label, future frames, or custom metadata at inference;
- offline action L2 gains are not stable across task/phase groups;
- simulator rollout remains unavailable and offline gains are too weak for a paper direction.

## Practical Gate

- runtime exceeds 30 minutes for the tiny gate;
- report lacks reproducible seed/split;
- any OpenVLA-OFT, custom `LIBERO_7D`, full benchmark, or simulator rollout dependency appears.
