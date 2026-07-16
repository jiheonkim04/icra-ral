# Epoch 4 Cycle 35 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `BRID-VLA`

Previous decision: `BRID_STAGE_0_NO_RESIDUAL_HEADROOM`

Previous result: `reports/brid_vla/stage_0_result.json`

BRID is closed without rescue. Its fixed Stage 0 result completed
`46080 / 46080` model rows with zero exceptions and exact manifest/partial key
equality, but the raw Diffusion Policy action-chunk proxy had negative
residual headroom relative to Base. This is a development-only no-headroom stop,
not a closed-loop scientific kill.

Cycle 35 must use one genuinely new mechanism. LoRA may be used only as
implementation infrastructure. The closest positive prior must enter the first
serious comparison. The selected direction should avoid the residual diffusion
family and be reproducible from existing LIBERO demonstrations without
privileged inference inputs.

## Primary-Source Anchors

### MTIL

Sources:

- https://arxiv.org/abs/2505.12410
- https://arxiv.org/html/2505.12410v3
- https://github.com/yulinzhouZYL/MTIL

AUTHOR_STATED: MTIL encodes full trajectory history with Mamba state-space
recurrence, conditions action prediction on that history plus the current
observation, and reports superior performance against ACT and Diffusion Policy
on ACT, Robomimic, LIBERO, and real-world tasks.

INDEPENDENTLY_INFERRED: The actual mechanism is not generic adaptation. It is a
compressed recurrent history state used as a belief-like latent for
non-Markovian action disambiguation. The strongest local proxy can hold SmolVLA
Base fixed and learn only a history-state residual/gate around Base action
chunks.

CROSS_PAPER_SYNTHESIZED: BRID showed that raw diffusion residual modeling around
Base lacks local prior headroom, while MTIL's positive axis is history
disambiguation rather than action-density generation. A Base-preserving
history-state method changes representation and failure condition at the same
time.

Mechanism map:

- observation/input: current LIBERO image/proprio/instruction, previous
  deployment-observable observations/actions, and frozen SmolVLA Base chunk;
- learned representation: recurrent state-space history state;
- supervision: demonstration action chunks and Base residuals on discovery and
  validation identities only;
- objective: residual chunk prediction plus history-state activation and clean
  Base-retention losses;
- policy component changed: bounded residual/gate after the frozen Base action
  chunk;
- inference intervention: default Base passthrough, nonzero correction only
  when history state predicts ambiguity-relevant residual structure;
- primary metric: validation proxy success/headroom, action validity, bounded
  delta, clean retention, and later paired closed-loop success;
- demonstrated causal link in prior: full-history state improves temporal
  ambiguity handling in imitation learning;
- untested local causal link: the history state can improve SmolVLA Base only
  when Base's current-frame action chunk is history-ambiguous.

Local relevance: existing LIBERO demonstrations contain ordered trajectories,
actions, instructions, and observations; cached SmolVLA Base chunks provide the
identity anchor. No rewards, success flags, object poses, future frames, or
held-out reset identities are needed at inference.

### Behavior Transformer

Sources:

- https://mahis.life/bet/
- https://github.com/notmahi/bet
- https://arxiv.org/abs/2206.11251

AUTHOR_STATED: Behavior Transformer models multimodal continuous behavior from
unlabeled demonstrations by clustering actions into discrete bins and learning a
continuous offset corrector; it reports improved demonstrated-task performance
while capturing major behavior modes.

INDEPENDENTLY_INFERRED: The core mechanism is discrete action-mode
classification plus offset prediction, not a generic transformer. A local proxy
can quantize Base residual chunks, predict residual mode tokens, and reconstruct
bounded offsets.

CROSS_PAPER_SYNTHESIZED: BRID's diffusion prior failed as an action-density
proxy, but the residual oracle still showed nonzero diagnostic reduction over
Base. A discrete residual-mode prior tests whether the residual structure is
clusterable even when denoising scores are not useful.

Local relevance: LIBERO action chunks and cached Base chunks are sufficient for
residual codebook construction. The risk is action-mode collapse or a simple
mean-residual/code-frequency baseline explaining the gain.

### Implicit Behavioral Cloning

Sources:

- https://implicitbc.github.io/
- https://github.com/google-research/ibc
- https://arxiv.org/abs/2109.00137

AUTHOR_STATED: IBC treats behavior cloning as implicit energy modeling over
observation-action pairs and reports that EBMs often outperform explicit MSE or
mixture-density policies on robotic tasks, including discontinuous,
multivalued, and contact-rich settings.

INDEPENDENTLY_INFERRED: The mechanism is energy-based action scoring and
candidate selection, not direct action regression. A local proxy can score
bounded candidates around the frozen Base chunk and choose the minimum-energy
candidate only when its margin over Base is sufficient.

CROSS_PAPER_SYNTHESIZED: BRID failed as a generative residual sampler. IBC tests
a different decision rule: compare candidate action chunks by learned implicit
compatibility instead of sampling a full action distribution.

Local relevance: existing LIBERO demonstration chunks provide positive
observation-action pairs. Negative candidates can be sampled from bounded Base
perturbations and other discovery chunks. The risk is that the energy model
collapses to explicit L2 residual regression or selects globally destructive
perturbations.

## Selection Implications

The strongest next direction is MTIL-anchored full-history state integration.
It has the most direct positive prior on LIBERO-like sequential manipulation,
changes the representation rather than the residual action-density estimator,
and can preserve SmolVLA identity by construction. Behavior Transformer and IBC
remain viable backup anchors because they test discrete action-mode structure
and implicit action compatibility respectively, but both are less directly VLA
and LIBERO aligned.
