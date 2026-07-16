# Epoch 4 Cycle 36 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `MHS-VLA`

Previous decision: `MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE`

Previous result: `reports/mhs_vla/stage_0_result.json`

MHS is closed without rescue. Its fixed Stage 0 result completed
`5193 / 5193` model rows with zero exceptions, exact manifest/partial key
equality, and no duplicate, missing, extra, or split-overlap keys. The
development-only label/contrast gate failed because validation labels collapsed
to `0` positive and `114` negative labels. This is not a closed-loop
scientific kill.

Cycle 36 must generate exactly three candidates. The selected method must use
one genuinely new scientific mechanism. LoRA may be used only as
implementation infrastructure. The closest positive external prior must enter
the first serious comparison. The selected direction should avoid changing MHS
labels, thresholds, history construction, tasks, identities, or interpretation.

## Primary-Source Anchors

### ACG

Sources:

- https://arxiv.org/abs/2510.22201
- https://arxiv.org/html/2510.22201v2
- https://github.com/DAVIAN-Robotics/ACG
- https://davian-robotics.github.io/ACG/

AUTHOR_STATED: ACG targets action incoherence in diffusion and flow-matching
VLA action heads. It constructs an intentionally incoherent action-generation
vector by disrupting temporal communication in self-attention, then guides
sampling away from that vector. The paper reports improved action coherence and
success on RoboCasa, DexMimicGen, and real-world SO-101 tasks, and states that
the method applies to flow-based VLA models including SmolVLA.

INDEPENDENTLY_INFERRED: The actual mechanism is perturbation guidance over the
action-generation vector field, not LoRA, generic smoothing, or action
regression. A local extension can keep the same flow-guidance axis while
replacing the hand-constructed incoherence direction with a
LIBERO-demonstration-calibrated coherence direction.

CROSS_PAPER_SYNTHESIZED: MHS failed because a binary history label collapsed.
ACG suggests a continuous action-coherence target that can be audited from
existing action chunks and does not require success labels, object states, or
privileged inference inputs. The strongest local experiment should compare
against ACG itself and a simple smoothing reviewer-killer.

Mechanism map:

- observation/input: ordinary SmolVLA observation, language, proprioception,
  and generated action-flow/chunk tensors;
- learned representation: demonstration-calibrated action-coherence geometry
  over within-chunk velocity, acceleration, jerk, and protected gripper events;
- supervision: existing LIBERO demonstration action chunks and cached Base
  action chunks on discovery/validation identities only;
- objective: fit a task/phase-local continuous coherence score and use it to
  define a bounded guidance direction at generation time;
- policy component changed: test-time action-generation guidance, not the VLA
  backbone and not a replacement action head;
- inference intervention: zero guidance is exact Base; nonzero guidance is
  bounded and only changes action generation when the sampled chunk violates
  the frozen coherence manifold;
- primary metric: validation action validity, clean retention, coherence
  improvement, ACG separation, simple-smoothing separation, and later paired
  closed-loop success;
- demonstrated causal link in prior: steering flow generation away from an
  incoherent vector field improves coherence and manipulation success;
- untested local causal link: a LIBERO-calibrated coherence direction improves
  SmolVLA more than ACG's architecture-perturbation direction under matched
  action semantics and budget.

Local relevance: existing LIBERO demonstrations contain full 7D action chunks,
episode order, gripper transitions, task labels, and ordinary SmolVLA inputs.
No simulator state, object pose, reward, success, done, future observation, or
confirmatory identity is required at inference.

### VLA-Corrector, A2C2, AAC, and RTC

Sources:

- https://arxiv.org/abs/2607.01804
- https://arxiv.org/html/2607.01804v1
- https://github.com/ZJU-OmniAI/vla-corrector
- https://arxiv.org/abs/2509.23224
- https://arxiv.org/html/2509.23224v1
- https://github.com/k1000dai/a2c2-libero
- https://arxiv.org/abs/2604.04161
- https://lance-lot.github.io/adaptive-chunking.github.io/
- https://arxiv.org/abs/2512.05964

AUTHOR_STATED: VLA-Corrector monitors predicted versus observed visual latent
evolution, truncates stale chunks, and applies Online Gradient Guidance during
replanning. A2C2 runs a small correction head at every control step using the
latest observation, base action, and chunk index. AAC selects chunk size from
action entropy. Training-time RTC conditions on action prefixes during
training to avoid inference-time inpainting overhead.

INDEPENDENTLY_INFERRED: These priors strongly cover adaptive horizon,
detect-and-correct, and stepwise action correction. They are not suitable as
the selected Cycle 36 mechanism unless the technical object is clearly outside
latent-drift monitoring, chunk truncation, entropy chunk sizing, and residual
step correction.

CROSS_PAPER_SYNTHESIZED: Cycle 20 NICE already extended VLA-Corrector through
normalized innovation and closed as a data failure. Cycle 36 therefore records
these priors as important comparison pressure, but does not select a new
latent-drift monitor.

Local relevance: these methods remain important baselines or reviewer
objections for any action-chunk method. They should not be used to rescue MHS
or to retune a closed NICE-style monitor.

### GEAR-VLA and AFI

Sources:

- https://arxiv.org/abs/2606.08530
- https://arxiv.org/html/2606.08530v2
- https://arxiv.org/abs/2512.07472
- https://arxiv.org/html/2512.07472v1

AUTHOR_STATED: GEAR-VLA uses coarse-to-fine action learning, semantic-aligned
3D integration, and embodiment canonicalization, reporting state-of-the-art
performance on LIBERO, zero-shot LIBERO-Plus, RoboTwin 2.0, and real-world
generalization. AFI addresses memory traps by constructing spatial affordance
fields, detecting stuck behavior, and using affordance-guided intervention;
it reports gains on OOD real-world scenarios and LIBERO-Pro.

INDEPENDENTLY_INFERRED: The shared positive axis is geometry- or
affordance-grounded action selection. A local proxy can use demonstration
image/action trajectories to form 2D or feature-space interaction fields, but
it cannot claim official 3D SAF or full GEAR reproduction without depth,
multi-view 3D reconstruction, or released checkpoints.

CROSS_PAPER_SYNTHESIZED: Geometry/affordance remains an important alternative
after MHS, but the local data path has higher source-risk than coherence
guidance because existing LIBERO demonstrations do not directly provide the
3D affordance fields used by AFI or GEAR.

### Robustness and Invariance Priors

Sources:

- https://arxiv.org/abs/2604.10055
- https://arxiv.org/abs/2510.00037
- https://arxiv.org/abs/2510.03827
- https://arxiv.org/html/2510.03827v2
- https://arxiv.org/abs/2510.13626
- https://sylvestf.github.io/LIBERO-plus/

AUTHOR_STATED: STRONG-VLA decouples robustness acquisition from clean
task-aligned refinement and reports LIBERO gains under seen and unseen
perturbations across OpenVLA, OpenVLA-OFT, and pi0. RobustVLA combines input
consistency and output robustness against multi-modal perturbations on LIBERO.
LIBERO-PRO and LIBERO-Plus document severe robustness failures under object,
layout, camera, initial-state, instruction, lighting, background, and sensor
perturbations.

INDEPENDENTLY_INFERRED: These priors support a robustness candidate, but prior
campaign cycles already touched perturbation replay, language contrast,
occlusion/complementary-view adaptation, and clean-retention projection. A new
candidate would need a narrow mechanism and a clear non-overlap argument.

Local relevance: existing LIBERO demonstrations can support synthetic
perturbation invariance and clean-retention diagnostics, but the link to the
current MHS data failure is weaker than the continuous action-coherence route.

## Selection Implications

The strongest Cycle 36 direction is ACG-anchored demonstration-calibrated
coherence guidance. It offers a positive prior with official code and a
flow-based VLA mechanism, avoids binary label collapse, preserves SmolVLA by
default, and can be audited from existing LIBERO action chunks before any
rollout. Geometry/affordance and robustness/invariance remain viable backup
directions but have higher local supervision or overlap risk.
