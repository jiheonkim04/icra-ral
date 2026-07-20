# Epoch 8 Active Latent-Property Boundary

Audit date: 2026-07-20

## Bounded current boundary

The June 2026 physical-reasoning position paper explicitly motivates
conditional tests in which an instruction refers to a nonvisual physical
property. RoboSemanticBench evaluates knowledge-grounded target choice but not
active identification of a hidden physical property. The inspected VLA
robustness and counterfactual-language benchmarks do not ship a probe-to-belief-
to-target manipulation task with ordinary-task retention. This leaves an
implementation opportunity, not a universal novelty claim: interactive
perception and system identification are mature adjacent fields, and a toy
hidden-mass task alone would not support a RA-L paper.

## Local artifact audit

The retained `libero_90` Kitchen Scene 2 supplies three visually identical
`akita_black_bowl` instances and separate official tasks for putting the front,
middle, or back instance on the same plate. Official init states and 50-demo
HDF5 files exist for all three tasks. X-VLA previously completed the ordinary
positional tasks in the retained 90-task scan, giving a competent low-level
Base rather than an untrained toy policy.

LIBERO has no hidden-mass language predicate, active-probe demonstrations, or
trained probe-selection checkpoint. Those are costs of a new benchmark and
method, not empirical closures. The new Stage -1 task therefore changes no
source asset: it cross-installs one frozen scene state into the three
goal-equivalent BDDL models, multiplies only the intended bowl's mass and
inertia by eight, and changes only the policy instruction between paired
conditions.

A retained discovery replay provides initial observability evidence. Under the
same legal grasp/lift actions, standard and 8x-mass bowl conditions both reached
official success, while the object trajectories differed by roughly 9--10 mm
late in the episode and the wrist-image mean absolute pixel difference exceeded
8 gray levels. Thus hidden mass is not automatically unobservable through the
legal RGB/proprioception stream on this controller stack.

## Stage -1 claim limit

The frozen six-pair screen asks only whether an already competent X-VLA has a
reproducible active-property grounding gap. A positive gap would authorize
method/data design; it would not show that a probe-to-belief mechanism works.
A negative gap would close this exact task construction, not active physical
property grounding in general.
