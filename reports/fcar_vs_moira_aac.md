# FCAR Vs MoIRA And AAC

Date: 2026-07-09 KST

## MoIRA

MoIRA is a close routing baseline. It uses modular instruction routing, external text-based routing options, low-rank adapter experts, and evaluates on LIBERO-style benchmarks.

Source: https://arxiv.org/abs/2507.01843

FCAR survives MoIRA only if it is not text/task-only routing.

Required difference:

- frozen/base is an explicit expert;
- routing is frame/state/action-disagreement-aware;
- the objective is retention against frozen/base negative transfer;
- evaluation focuses on negative-transfer frames, not only task-average routing.

Killed variants:

- route by task;
- route by instruction;
- external LLM router;
- multiple LoRA experts without frame signal;
- generic MoE for robotics.

## AAC

AAC adapts action chunking at inference time using action uncertainty/entropy, without training or architecture changes.

Source: https://arxiv.org/abs/2604.04161

FCAR is different because it does not choose chunk size. It chooses between base and LoRA experts per frame. AAC remains an adjacent temporal-stability baseline, especially if FCAR later claims control stability.

## Standard LoRA

Standard rank-4 LoRA is already measured and is worse than frozen/base on aggregate:

- frozen/base action L2: `0.106514960`
- rank-4 LoRA action L2: `0.118024259`

FCAR must show that selective retention avoids this negative transfer.

## Adapter Soup / Static Merge

Adapter soup or static prediction mixing could remove some negative transfer without a learned frame gate.

FCAR must beat a static mixture grid before it is method-worthy.

## Task-Specific Experts

Task oracle barely improves over frozen/base:

- frozen/base action L2: `0.106514960`
- task oracle action L2: `0.106079976`
- absolute gain: `0.000434984`

Task-level specialization alone is not sufficient under current evidence.

## Frozen/Base Fallback

Frozen/base is the strongest realistic aggregate expert and must be selectable at inference. FCAR must justify every deviation from base.
