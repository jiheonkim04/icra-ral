# TG-VLA Task Definition

Date: 2026-07-09 KST

Long title: Target-Grounded VLA Adaptation for Language-Robust Robot Manipulation

## Core Claim Under Test

TG-VLA is valid only if the research novelty is target/object grounding injected into the VLA action pathway, plus consistency and sensitivity objectives that separate target-preserving paraphrases from true target changes.

LoRA, QLoRA, frozen feature heads, and low-rank adapters are training tools only. They are not the contribution.

## Hypothesis

VLA policies fail under paraphrase and object lexical variation partly because the instruction-resolved target or object is not tied strongly enough to the action pathway. A lightweight adapter should improve wrong-target, paraphrase, and object-lexical robustness if it conditions the action representation or action head on a non-leaking target/object prior while preserving sensitivity when the target actually changes.

## Method Sketch

Preferred local path:

1. Use SmolVLA as the first real VLA backbone.
2. Freeze the backbone or use tiny LoRA/adapter updates only.
3. Resolve a target/object grounding signal from instruction text plus visible/object names, not from eval labels, BDDL task IDs, filenames, or success metadata.
4. Inject that signal into the action pathway through a target-conditioned residual adapter, FiLM/AdaLN gate, or action-head conditioning.
5. Train with supervised action loss plus:
   - same-target paraphrase consistency,
   - counterfactual target sensitivity,
   - clean-retention loss.

## Required Baselines

- frozen/base SmolVLA or frozen feature baseline,
- standard LoRA or action imitation adapter,
- canonicalization-only,
- prompt-only target insertion or canonical target wording if cheap,
- generic paraphrase augmentation,
- no-adaptation baseline,
- mean action, linear/L1 action head, and simple MLP action head where a head-only comparison is used,
- direct single 3D point or destination-only point baseline if any 3D/point signal is used,
- oracle target upper bound only when clearly labeled oracle.

## Non-Leakage Rule

Allowed:

- instruction text,
- observation images and robot proprioception,
- object names or visible candidates available to the model at inference,
- official train split metadata used only for training supervision.

Forbidden at inference:

- BDDL target labels,
- task IDs,
- filenames,
- eval labels,
- success/failure labels,
- simulator privileged state,
- oracle object targets except in a separately labeled upper-bound ablation.

## Valid Novelty

TG-VLA remains novel only if it shows that explicit target/object grounding in the action pathway is necessary beyond:

- standard LoRA,
- canonicalization,
- prompt engineering,
- generic paraphrase augmentation,
- generic DPO/ORPO,
- single-point injection,
- old TCA heads or local ActionMap approximations.

## Current Boundary

This run is a STATE 0-1 alignment and feasibility gate. No training, rollout, download, OpenVLA-OFT execution, or paper-grade claim is authorized by this file.
