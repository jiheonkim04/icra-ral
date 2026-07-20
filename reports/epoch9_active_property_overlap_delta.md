# Epoch 9 Active-Property Overlap Delta

Date: 2026-07-20 KST
Scope: focused delta from the Epoch 8 literature matrix; primary sources checked through 2026-07-20.
Decision: `PROCEED_WITH_NARROWED_VLA_SENSOR_MINIMAL_CLAIM`

## Exact proposed scope

The working contribution is not the first language-conditioned robot system to
reason about hidden weight. It is a controlled VLA benchmark and method in
which visually matched candidates have independently balanced latent physical
properties, the instruction uses a relative operator such as *heaviest* or
*lightest*, and the policy must physically probe candidates, infer their
ordering from ordinary RGB/proprioceptive action effects, select the intended
target, and finish the manipulation. At inference it receives no segmentation,
object pose, simulator property, explicit force/torque, tactile signal, reward,
success flag, or oracle target identity.

## Closest collision

[CLIER / SHOP-VRB2 (ICRA 2025)](https://arxiv.org/abs/2404.15194) is the
closest prior and changes the novelty position. It already introduces a MuJoCo
benchmark with non-visual weight/stiffness, relative-property instructions
(including picking, placing, stacking, and ordering by heaviest/lightest),
interactive measurement, target selection, and closed-loop simulated and real
manipulation. It therefore precludes claims that Epoch 9 is the first active
physical-property language benchmark or the first closed-loop robot to execute
relative-weight instructions.

The remaining exact difference is substantial but narrower:

| Dimension | CLIER / SHOP-VRB2 | Epoch 9 target |
|---|---|---|
| Sensing | RGB scene parsing plus Mask R-CNN, CosyPose, explicit object geometry, and a measured weight/stiffness field; real weight is measured through lifting and joint-torque differences | only ordinary RGB and proprioceptive/controller-response history; no explicit property/force field, segmentation, or object pose |
| Intervention | learned symbolic planner invokes hand-coded grasp/lift/weigh/squeeze primitives | standardized non-destructive physical probes whose raw temporal effects form the learned evidence |
| Language role | seq2seq instruction-to-symbolic-program generation over a scene graph | relative language operator conditions a VLA-side relational target selector |
| Belief | measured non-visual property is written into the scene graph and then sorted | temporal learned pairwise ordering from action/RGB/proprioception trajectories |
| Low-level policy | neuro-symbolic keyframe planner and pre-coded primitive controllers | retained VLA completes the selected canonical task after legal routing/conditioning |
| Evaluation | heterogeneous objects, visible attributes, and explicit property measurements; strong task breadth and sim-to-real evidence | causal matched appearance/geometry with property-position swaps, static/no-probe controls, property-identification/first-contact/completion decomposition, VLA retention |
| Hardware | RGB camera and robot joint-torque measurement; pose models and explicit scene graph | RGB cameras and ordinary proprioception only; simulator first, no physical robot in Epoch 9 |

Paper positioning must lead with this delta and compare against CLIER directly.
A paper-scale result should show why sensor-minimal VLA grounding is not
explained by explicit measurement plus symbolic sorting.

## Required comparisons

| Work | Exact overlap | Exact non-collision |
|---|---|---|
| [Position: VLA Models Cannot Be Verified to Perform Physical Reasoning](https://arxiv.org/abs/2606.30686) | motivates controlled interventions that separate semantic mapping from physical action decisions | a position/identifiability argument, not an implemented physical probe, hidden-property language benchmark, target selector, or completing policy |
| [Physically Grounded VLMs / PhysObjects](https://arxiv.org/abs/2309.02561) | language reasoning about mass, fragility, material, and other physical concepts; planner-level robot demonstrations include heaviest-object tasks | predicts human visual priors from static appearance; does not make visually identical counterfactual candidates identifiable through interaction or learn a probe-response belief |
| [SaPaVe (CVPR 2026)](https://arxiv.org/abs/2603.12193) | jointly learns active perception and manipulation in a VLA | intervention is camera motion/viewpoint selection and 3-D geometry-aware execution, not contact probes for latent mass; no relative-property target benchmark |
| [ActiveVLA](https://arxiv.org/abs/2601.08325) | active information gathering improves VLA manipulation | chooses 3-D viewpoints and zoom regions to resolve spatial/occlusion uncertainty; it neither changes object dynamics nor identifies a hidden property from action effects |
| [RoboSemanticBench](https://arxiv.org/abs/2606.02277) | diagnoses whether VLAs use instruction semantics to choose the correct candidate before manipulation | candidates encode visible answers to math/knowledge questions; no hidden physical property, active probe, temporal belief, or causal property swap |
| [ForceVLA (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/file/8633b46e12cc5f2ee1f05a6ca2c65b38-Paper-Conference.pdf) | fuses language, vision, proprioception, and interaction dynamics for closed-loop VLA action | consumes a dedicated 6-axis force/torque sensor and addresses contact-rich execution, not sensor-minimal relative-property identification and target selection |
| [Tactile-VLA](https://arxiv.org/abs/2507.09160) | grounds force-related language and adapts manipulation from contact feedback | adds tactile sensing and hybrid position-force control; evaluates how to interact, not which visually matched candidate satisfies an unknown relative property |
| [AT-VLA (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Li_AT-VLA_Adaptive_Tactile_Injection_for_Enhanced_Feedback_Reaction_in_Vision-Language-Action_CVPR_2026_paper.html) | selectively injects high-rate tactile feedback while preserving a VLA stream | requires tactile hardware and targets fast contact reaction; no active property-ranking benchmark or probe-to-target chain |
| [TaF-VLA](https://arxiv.org/abs/2601.20321) | uses temporal tactile observations aligned to physical force and shows force-aware VLA gains | requires tactile images, 6-axis force/torque, and a force-map dataset; no RGB/proprio-only candidate ranking |
| [FD-VLA](https://arxiv.org/abs/2602.02142) | predicts a force token from vision and robot state so inference can omit a physical force sensor | force distillation is supervised by force sensing and supports contact-rich control; it does not actively compare visually matched candidates named by relative language |
| [Learning Object Properties Using Robot Proprioception via Differentiable Robot-Object Interaction](https://arxiv.org/abs/2410.03920) | identifies mass and elasticity from robot reactions without vision or external measurement tools | estimates numerical properties of a manipulated object through a differentiable digital twin; no language, candidate comparison, VLA target selection, or task-completion benchmark |
| [Predictive Visuo-Tactile Interactive Perception](https://doi.org/10.1109/TRO.2025.3531816) | actively explores mass, friction, stiffness, and related object properties and uses estimates downstream | relies on tactile/visual tracking and property-estimation machinery rather than a VLA, relative-property language, and RGB/proprio-only inference |

## Audit conclusion and claim constraints

No checked primary paper implements the full conjunction of (1) visually
matched and position-balanced candidate objects, (2) relative latent-property
language, (3) a physical contact probe, (4) learned temporal relational belief
from RGB/proprioception without explicit force, tactile, pose, segmentation, or
property input, (5) VLA target selection, and (6) closed-loop completion with
ordinary-task retention.

The route is therefore not directly duplicated. The novelty claim must avoid
"first active physical reasoning benchmark" and "first robot to follow
heaviest/lightest instructions." A defensible claim, if supported
empirically, is the first controlled sensor-minimal VLA study of relative
hidden-property grounding through learned physical probe effects, with causal
property-position swaps and decomposed closed-loop evidence. CLIER is the
mandatory closest prior; static PhysObjects routing, explicit-measurement
sorting, endpoint proprioception, and force/tactile systems are mandatory
controls or discussion comparators where executable comparison is impossible.
