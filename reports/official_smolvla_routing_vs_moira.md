# Official SmolVLA Routing vs MoIRA

Date: 2026-07-09 KST

## MoIRA Comparison

- MoIRA does modular instruction routing: `True`
- MoIRA uses external text-based routing: `True`
- MoIRA uses low-rank adapter experts: `True`
- MoIRA evaluates on LIBERO: `True`
- instruction/task-only routing killed by MoIRA: `True`

Sources:

- MoIRA arXiv/html: https://arxiv.org/html/2507.01843v2
- SmolVLA paper: https://arxiv.org/abs/2506.01844
- SmolVLA blog: https://huggingface.co/blog/smolvla
- AAC paper: https://arxiv.org/abs/2604.04161

## Surviving Differentiator

Base-retentive frame/state/action-disagreement-aware gating that can select frozen/base per frame to avoid negative transfer.

## Rejected Novelty Claims

- use LoRA
- route by task
- route by instruction
- use multiple adapters
- external LLM router
- generic MoE for robotics

## Required Comparison Baselines

- frozen/base fallback
- standard rank-4 LoRA
- task-specific LoRA experts
- task oracle
- simple instruction embedding router
- MoIRA-style text router
- adapter soup / weighted LoRA merge
- AAC as adjacent temporal-stability baseline

## Baseline-By-Baseline Comparison

Standard rank-4 LoRA:

- observed aggregate action L2 is worse than frozen/base: `0.118024259` vs `0.106514960`;
- any proposed method must not be "use LoRA"; it must avoid LoRA negative-transfer frames.

Frozen/base fallback:

- frozen/base is the strongest realistic aggregate expert in the current evidence;
- the surviving design must include frozen/base as an explicit selectable expert.

Task-specific LoRA experts:

- task oracle selects LoRA for only `80` / `200` frames and improves action L2 over frozen/base by only `0.000434984`;
- task-specific experts alone are unlikely to justify method work unless future task samples show larger headroom.

Oracle task router:

- task oracle action L2 is `0.106079976`, only `0.004083783` relative improvement over frozen/base;
- this fails the `0.005` absolute / `5%` relative headroom gate.

Simple instruction embedding router:

- with the current labels it is equivalent to task/instruction oracle;
- because task-oracle headroom is tiny and MoIRA already covers instruction routing, this is not a surviving novelty.

Adapter soup / weighted LoRA merge:

- not tested in this run;
- it must be a required future baseline because a static or weighted merge could remove negative transfer without a learned frame router.

AAC:

- AAC is adjacent because it adapts action chunking at inference time;
- it is not a direct adapter-routing killer, but any control-stability claim must compare against AAC-style inference-time temporal adaptation.
