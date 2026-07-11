# OpenVLA-OFT Local Feasibility

Date: 2026-07-11 KST

Selected second VLA: `OpenVLA-OFT`

Selected released checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`

## Source And Checkpoint

- official repository: `https://github.com/moojink/openvla-oft`
- LIBERO evaluation instructions: `https://raw.githubusercontent.com/moojink/openvla-oft/main/LIBERO.md`
- setup instructions: `https://raw.githubusercontent.com/moojink/openvla-oft/main/SETUP.md`
- paper: `https://arxiv.org/abs/2502.19645`
- checkpoint API audit: Hugging Face model is public, non-gated, MIT, SHA `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- checkpoint size from Hugging Face API with blob metadata: `15,939,168,050` bytes (`14.845` GiB)

The selected combined checkpoint is preferred over suite-specific checkpoints because this gate needs one second backbone/checkpoint that can evaluate both `libero_spatial` and `libero_10`.

## Compatibility

OpenVLA-OFT has official LIBERO support and released checkpoints for `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, plus one combined checkpoint trained on all four suites. The official recipe uses LIBERO simulation, third-person and wrist images, proprio state, and relative actions.

The current local SmolVLA route uses LeRobot's official LIBERO wrapper rather than OpenVLA-OFT's runner, so cross-model evaluation would require a separate OpenVLA-OFT environment or a careful adapter around its official runner. That adapter is protocol work, not a method contribution.

## Hardware Feasibility

Local RTX 5080 16GB:

- feasible for metadata audit: yes
- feasible for official-size checkpoint download: storage likely yes, but approval required
- feasible for faithful inference: not proven; the 14.845 GiB checkpoint leaves little VRAM margin
- feasible for fine-tuning: no; fine-tuning is also forbidden by objective

8x RTX 3090 lab:

- feasible for inference: likely yes using one 24GB card per rollout worker or one card total
- feasible for official fine-tuning: not needed and not approved
- best hardware path: lab GPU if the user wants real cross-backbone rollout evidence

## Local Asset Check

Current local state:

- `C:\assets\checkpoints\openvla`: absent
- `C:\assets\checkpoints\openvla_oft`: absent
- `C:\assets\repos\openvla-oft`: absent
- `C:\assets\repos\LIBERO-PRO`: absent

No OpenVLA-OFT asset was downloaded in this pass.

## Feasibility Decision

OpenVLA-OFT is a valid second-backbone choice, but it is not ready to run locally without download approval and likely a lab-GPU path.

Decision: `SECOND_BACKBONE_DOWNLOAD_APPROVAL_REQUIRED`
