# ActionMap Anchor Risk Register

## Anchor Approximation Risk

Risk: the local CPU NumPy heatmap diagnostic could be mistaken for a full ActionMap reproduction.

Mitigation: label STATE 1 as exploratory local HDF5 action-head reproduction only. Keep the official ActionMap paper and code as the anchor, but do not claim comparable training, backbone, benchmark, or success-rate evidence.

## Simple Baseline Risk

Risk: ActionMap-style heatmaps may look interesting while mean, linear/L1, or cheap MLP baselines are stronger.

Mitigation: make those baselines mandatory and kill or reframe if any match or beat the heatmap head on held-out 7D action L2.

## Candidate Grid Risk

Risk: a coarse candidate grid may cap performance so the learned head cannot win.

Mitigation: report an oracle nearest candidate upper bound and treat weak oracle headroom as a kill/reframe condition, not as method evidence.

Observed STATE 1 outcome: the oracle upper bound had strong headroom (`0.065653208` action L2), so the grid itself was not the decisive blocker. The learned ActionMap-style head still lost to mean-action and cheap MLP and collapsed to one rotation bin.

## Reframe Risk

Risk: treating a failed local anchor approximation as a reason to invent a method anyway would repeat the method-first pattern that killed previous routes.

Mitigation: block STATE 2 failure mining and extension work unless a re-run first beats mean-action, linear/L1, and cheap MLP baselines without candidate collapse on real HDF5-backed metrics.
