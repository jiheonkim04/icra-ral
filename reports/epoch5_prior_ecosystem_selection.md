# Epoch 5 Official-Prior Ecosystem Selection

Date: 2026-07-17 KST
Branch: `codex/epoch5-official-prior-first`
Audit anchor: `reports/autonomous_research_full_history_audit.md`
Audit commit: `b0ecb6ea5f6eba2953b5bd842883c0474d634dff`

## Decision

Selected ecosystem: **OpenVLA-OFT on LIBERO**.

This is not an Ours method and not a rescue of MCI-VLA. Epoch 5 begins by validating an official external prior, then finding a residual failure condition before any new method proposal.

Cycle 39 ordinary exact-three local-method search is superseded. The supersession is a strategy reset, not a scientific kill.

## Selection Criteria

Weights from the user instruction:

| Criterion | Weight |
|---|---:|
| Official artifact completeness | 25 |
| Positive closed-loop evidence | 20 |
| Local reproducibility | 20 |
| Meaningful residual gap | 15 |
| Novelty-extension opportunity | 10 |
| Second-backbone/condition path | 10 |

## Exactly Three Prior Ecosystems

| Rank | Ecosystem | Primary paper | Official artifacts | Positive closed-loop evidence | Benchmark/action semantics | Local feasibility | Residual gap | Second path | Score |
|---:|---|---|---|---|---|---|---|---|---:|
| 1 | OpenVLA-OFT on LIBERO | "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success", arXiv 2502.19645 | project page, `moojink/openvla-oft`, MIT code, Hugging Face LIBERO checkpoints | reports 97.1% average success across four LIBERO suites and 26x faster action generation than base OpenVLA | LIBERO, 7D actions, action chunk size 8, optional wrist/proprio | already has local repo, local 15G combined checkpoint, validated INT4 hard-slice run | hard-slice condition saturated; likely residual requires perturbation/robustness condition | same prior can be compared to SmolVLA Base and later Quantized OpenVLA-OFT + Ours | 89 |
| 2 | pi0.5 / OpenPI LIBERO | pi0 and OpenPI ecosystem; pi0.5-LIBERO checkpoint in official repo | `Physical-Intelligence/openpi`, official GCS checkpoints, PyTorch/JAX paths, LIBERO configs | official README says pi0.5-LIBERO gets state-of-the-art performance; LIBERO-PRO evaluates pi0/pi0.5 with single official checkpoint | LIBERO via OpenPI inputs/outputs, policy-server workflow | feasible only after new OpenPI/uv/JAX or PyTorch stack; likely Docker/GCS friction | meaningful: official issue reports reproduction/version risk, LIBERO-PRO perturbations expose robustness gap | second condition via LIBERO-PRO perturbations | 79 |
| 3 | Policy Contrastive Decoding (PCD) on open robot policies | "Policy Contrastive Decoding for Robotic Foundation Models", arXiv 2505.13255 / OpenReview | `pcd-robot/PCD`, `PCD-real`, `PCD-LeRobot`; Apache-2.0 for LeRobot repo | paper/repo report improvements on OpenVLA, Octo, and pi0 in simulation and real world; training-free | SimPLER and real robot settings; LeRobot path exists | likely feasible after extra dependencies and checkpoint downloads; not matched LIBERO out of the box | meaningful: object-relevance/spurious-correlation failures | can apply as prior-method baseline to SmolVLA/OpenVLA if action-logit interface is available | 75 |

## Source Evidence

OpenVLA-OFT primary sources:

- Project page: https://openvla-oft.github.io/
- Paper: https://arxiv.org/abs/2502.19645
- Official code: https://github.com/moojink/openvla-oft
- LIBERO instructions: https://github.com/moojink/openvla-oft/blob/main/LIBERO.md
- Example checkpoint: https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial
- Combined checkpoint used locally: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`

OpenPI/pi0.5 primary sources:

- Official repo: https://github.com/Physical-Intelligence/openpi
- Open-sourcing post: https://www.pi.website/blog/openpi
- pi0 paper: https://arxiv.org/abs/2410.24164
- FAST paper: https://arxiv.org/abs/2501.09747
- LIBERO-PRO paper: https://arxiv.org/abs/2510.03827

PCD primary sources:

- Paper: https://arxiv.org/abs/2505.13255
- Official repo: https://github.com/pcd-robot/PCD
- LeRobot implementation: https://github.com/pcd-robot/PCD-LeRobot
- Real-world implementation: https://github.com/pcd-robot/PCD-real

## Why OpenVLA-OFT Is Selected

OpenVLA-OFT is selected because it is the only ecosystem among the three with all of the following already true locally:

- official LIBERO benchmark match;
- public code and checkpoints;
- local official-prior checkout at `C:\assets\repos\openvla-oft`;
- local combined checkpoint at `/home/jiheon/assets/checkpoints/openvla-oft/moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10`;
- prior local INT4 execution artifacts with 20/20 successful OpenVLA-OFT hard-slice episodes;
- focused test validation in this epoch: `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_openvla_oft_int4_gate.py -q`, 4 passed.

The main weakness is residual-gap risk: the recovered hard-slice comparison is saturated by OpenVLA-OFT INT4. Therefore, the next scientific action after reproduction validation is residual-condition selection, not Ours design.

## Rejected-For-Now Ecosystems

pi0.5/OpenPI is strong but needs a new OpenPI stack, official GCS checkpoint acquisition, and likely Dockerized LIBERO evaluation. It is the first fallback if OpenVLA-OFT cannot expose a residual.

PCD is scientifically attractive because it attacks object-relevance and spurious correlations across OpenVLA, Octo, and pi0, but it is not as directly matched to the existing LIBERO/SmolVLA/OpenVLA-OFT local stack and may require extra checkpoints or segmentation/object-mask assets.

Octo and RDT were inspected but not selected into the exact-three set because local LIBERO matching is weaker for Octo and local RTX 5080/24GB RAM feasibility is weaker for RDT-style 1B diffusion foundation models. TinyVLA was not selected because official checkpoint/reproducibility evidence was weaker than the selected three for this repo's immediate LIBERO path.

## Post-OpenVLA Fallback Preflight

Date: 2026-07-17 KST.

After `R2R-OFT` failed its offline selection gate and the no-training
short-requery control was not selected, the two preselected fallback ecosystems
were checked for immediate local readiness before any download, install, or
rollout.

| Ecosystem | Official source state | Local state | Blocking issue | Classification |
|---|---|---|---|---|
| pi0.5 / OpenPI LIBERO | `Physical-Intelligence/openpi` cloned at main `15a9616a00943ada6c20a0f158e3adb39df2ccac`; official README lists `gs://openpi-assets/checkpoints/pi05_libero`, `>8 GB` inference, `>22.5 GB` LoRA fine-tuning, and Docker as recommended LIBERO workflow | source at `C:\assets\repos\openpi`; isolated env `/home/jiheon/venvs/openpi-uv` created with Python 3.11/JAX/Torch/OpenPI; official checkpoint downloaded to `/home/jiheon/assets/checkpoints/openpi`; one random-input policy-load smoke exited `137` during restore/inference under current WSL memory | official source/env/checkpoint are present, but policy instantiation is blocked by local memory/resource kill before JSON result; no closed-loop rollout happened | `OPENPI_PI05_LOCAL_POLICY_LOAD_EXIT_137_NOT_SCIENTIFIC_KILL` |
| PCD / PCD-LeRobot | `pcd-robot/PCD` cloned at main `cec18b820daeadfdaf080c030a1b5eb080ff75cd`; `pcd-robot/PCD-LeRobot` object database inspected at main `519b4a814e85bf9b786677d90b0ff07218729bb2`; official READMEs require SAM2, GroundingDINO, Inpaint-Anything/big-lama, and extra pretrained checkpoints | source checkouts now exist under `C:\assets\repos`; checked OpenVLA WSL env has LeRobot, OpenCV, and diffusers, but no `sam2`, `groundingdino`, or `openpi`; no matching local vision/inpainting checkpoints found | official method still requires segmentation/inpainting dependency setup and manual/extra checkpoint downloads before a fair prior run | `FALLBACK_REQUIRES_EXTERNAL_SETUP_NOT_SCIENTIFIC_KILL` |

Decision: `OPENPI_PI05_LOCAL_POLICY_LOAD_EXIT_137_NOT_SCIENTIFIC_KILL`.

This is not a paper-method result and not a kill of either external prior.
OpenPI progressed from source preflight to local source/env/checkpoint
availability, but current local WSL memory killed policy restore/inference
before a usable prior rollout. PCD remains source-inspected but blocked before
fair execution by dependency and checkpoint requirements.
