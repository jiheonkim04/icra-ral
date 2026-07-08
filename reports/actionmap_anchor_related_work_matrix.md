# ActionMap Anchor Related Work Matrix

| Work | Evidence | Reproduction relevance | Local gap |
| --- | --- | --- | --- |
| ActionMap (2026), https://arxiv.org/abs/2606.06904 | Introduces a voxel heatmap action head for VLA action decoding and reports LIBERO and real-world Franka gains over native decoders. | Main anchor. STATE 1 approximates the translation, rotation, and gripper heatmap idea locally. | Local diagnostic is CPU NumPy over HDF5 observations, not full VLA training. |
| Official ActionMap pre-release code, https://github.com/showlab/ActionMap | Releases a core heatmap action head preview with translation grid, rotation grid, and gripper branch. | Guides the required diagnostic shape: heatmap/candidate scoring rather than direct regression. | The repo does not yet provide a complete training stack; local reproduction remains minimal. |
| Native L1/regression action heads | Common VLA action decoder baseline. | Required baseline through `linear_l1_action_head`. | If direct regression wins locally, ActionMap-style reproduction is not useful yet. |
| Mean-action baseline | Strong trivial local baseline that killed earlier action-decoder routes. | Required baseline. | If mean action wins, the local feature/action setup is too weak for anchor claims. |
| Simple MLP action head | Cheap nonlinear baseline. | Required anti-baseline before proposing extensions. | If MLP wins, ActionMap-style heatmap novelty is not isolated. |

## Positioning

This route first asks whether the ActionMap-style anchor is locally reproducible enough to beat simple action-head baselines. Only after that should failure mining or a narrow extension be considered.

