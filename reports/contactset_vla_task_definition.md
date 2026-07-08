# ContactSet-VLA Task Definition

Long title: Contact-Set Grounded Action-Head Injection for Vision-Language-Action Robot Manipulation.

Anchor paper: Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization, arXiv:2606.27663v1, 26 Jun 2026, https://arxiv.org/abs/2606.27663.

Motivation: the anchor paper shows that a 3D grounded point injected directly into a VLA action head can dramatically improve spatial and task generalization on LIBERO-PRO. The open question for this branch is whether one point is too weak for contact-rich and multi-stage manipulation where the policy needs source-object contact, placement target, support surface, orientation/contact normal, and safety/avoidance cues.

Core hypothesis: injecting a structured contact set into the action head improves action prediction and later replay/progress under spatial and task perturbations beyond active single-point injection and simple source-only or destination-only baselines.

STATE 1 scope: build a bounded offline action-head diagnostic over local LIBERO HDF5 action chunks. Do not start full VLA fine-tuning, OpenVLA-OFT, GPU jobs, downloads, simulator rollouts, or paper-grade claims.

Non-goals:

- no Target-Prior TCA-Map continuation,
- no CSS-Shield continuation,
- no ExecSpec-Repair continuation,
- no AMP-GD continuation,
- no ResetSpec-Retarget continuation,
- no Phase-Locked Retiming continuation,
- no TL-ChunkRepair continuation,
- no ContactTube-Aug continuation,
- no full VLA fine-tuning before a bounded diagnostic beats simple baselines.

Evidence labels allowed in STATE 1:

- exploratory offline action-head loss,
- local HDF5 action-chunk proxy,
- geometry-observability audit,
- setup readiness for a later exact-init replay/progress diagnostic.

Evidence labels not allowed in STATE 1:

- standard LIBERO success,
- simulator rollout success,
- SOTA,
- paper-grade result,
- real-robot claim.

