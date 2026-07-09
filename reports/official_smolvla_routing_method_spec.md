# Official SmolVLA Routing Method Spec

Date: 2026-07-09 KST

Final decision: `GO_DESIGN_FRAME_CONDITIONAL_ROUTING`

Method name: `Frame-Conditional Adapter Retention`

Problem statement: Low-data rank-4 LoRA on official SmolVLA-LIBERO creates negative transfer: some frames improve while others are worse than frozen/base.

Precise failure gap: Routing oracle can avoid LoRA-hurt frames, but task-only routing has weak or MoIRA-covered headroom.

Model components:

- frozen/base SmolVLA expert
- rank-4 LoRA adapted expert
- small router/gate
- retention regularizer

Router input signals:

- instruction/task embedding
- current state
- visual embeddings if cached
- base-vs-LoRA action disagreement
- chunk/eval-loss proxy or uncertainty

Router level: `frame-level or hybrid`
Frozen/base explicit expert: `True`

Training objective: supervise gate to select LoRA only when it improves held-out action proxy while retaining base otherwise; no method training should start before a fixed protocol.

Retention loss: penalize selecting LoRA when frozen/base has lower action L2 or lower normalized chunk loss on diagnostic supervision.

Metrics:

- held-out action L2
- translation L2
- rotation L2
- gripper error/sign
- normalized chunk eval loss
- negative-transfer frame rate

Required baselines:

- frozen/base SmolVLA
- standard rank-4 LoRA
- mean-action prior
- task oracle
- frame oracle
- MoIRA-style instruction routing

Ablations:

- no frozen/base expert
- instruction-only router
- frame-state-only router
- no retention loss
- weighted adapter merge/soup

Exact first experiment: planning-only next; then, if approved, train a tiny gate on the same official diagnostic split with frozen/base fallback and compare to oracle bounds.

Kill criteria:

- frame oracle below 5%/0.005 headroom
- gate fails to beat frozen/base
- MoIRA-style instruction router matches it
- mean-action or trivial prior explains gains

Novelty vs MoIRA: must be frame/state/action-disagreement aware and base-retentive; instruction-to-adapter routing alone is killed by MoIRA.
Novelty vs standard LoRA: explicitly avoids LoRA negative-transfer frames instead of always applying the adapter.
Novelty vs AAC: not an action-chunk length scheduler; AAC is an adjacent temporal-stability baseline, not the routing mechanism.
Expected RA-L strength: `low-medium until official rollout exists`
Expected kill risk: `high`