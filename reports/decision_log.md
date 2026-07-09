# Decision Log

## 2026-07-08: Research Reset And Target-Grounded ActionMap Scout

Decision: `NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

Reason: the only salvageable family was Target-Prior TCA reframed as Target-Grounded ActionMap / Language-Grounded Action Heatmap, but the local ActionMap substrate had not cleared mean-action, linear/L1, and cheap-MLP gates.

## 2026-07-08: ActionMap Mini-Anchor Gate

Decision: `KILL_ACTIONMAP_ANCHOR`

Reason: the bounded LIBERO/HDF5 mini-anchor produced real metrics but failed the hard gate.

Key metrics:

- dataset/split: `8` local LIBERO HDF5 demos, `deterministic_per_demo_time_holdout`
- train/eval records: `1008 / 432`
- mean-action action L2: `0.466767673`
- linear/L1 action L2: `0.812610317`
- simple MLP action L2: `0.501926707`
- ActionMap-style action L2: `0.529931357`
- oracle candidate action L2: `0.065653208`
- candidate top1: `0.018518519`
- candidate collapse: yes, unique translation/rotation/gripper bins `5 / 1 / 2`

Triggered kill criteria:

- mean-action baseline matched or beat the ActionMap-style heatmap head;
- cheap MLP action head matched or beat the ActionMap-style heatmap head;
- ActionMap-style head collapsed to too few candidates.

Consequence: do not proceed to Target-Grounded ActionMap from this local anchor result.

Interpretation: this kills the local minimal ActionMap approximation, not the official ActionMap paper. The low oracle candidate upper bound suggests candidate-space headroom, but the learned local head collapsed and did not exploit it.

## 2026-07-08: Official Anchor Required

Decision: `OFFICIAL_ANCHOR_REQUIRED`

Reason: local proxy and minimal approximations have repeatedly produced plausible auxiliary evidence while failing simple-baseline gates. No new VLA method should be started without an official anchor reproduction.

Only viable next steps:

A. Official ActionMap reproduction with official code/assets.

B. Official LIBERO-Safety/SafeManip benchmark reproduction.

C. Stop VLA method search under current constraints.

Execution boundary for this archive pass:

- experiments happened: no;
- training happened: no;
- rollout/replay happened: no;
- downloads/GPU/OpenVLA-OFT happened: no / no / no;
- Target-Grounded ActionMap implementation happened: no;
- new method implementation happened: no.

## 2026-07-09: PatchGuard-VLA STATE 1B Decision

Decision: `KILL_BASELINE_DOMINATED`

Reason: STATE 1B resolved the prior installable environment blocker and proved that local PEFT/bitsandbytes/CUDA/SmolVLA LoRA can run, but PatchGuard did not beat the predeclared baselines.

Key positive evidence:

- patch effect measured in STATE 1: max attacked policy-action L1 `0.181765`;
- max attacked translation-action L2 `0.213965`;
- kinematic/proprioceptive signal available;
- PEFT `0.19.1` installed and worked;
- bitsandbytes `0.49.2` installed and 4-bit/8-bit CUDA smokes passed;
- CUDA/PyTorch on RTX 5080 worked;
- SmolVLA LoRA injection worked;
- tiny training smoke ran;
- loss computed;
- VRAM peak `2224.845` MB;
- runtime `57.438` sec.

Decisive negative evidence:

- standard LoRA metric `0.144186`;
- generic adversarial LoRA metric `0.142803`;
- PatchGuard metric `0.13356`;
- cutout/random-erasing metric `0.02973`;
- PatchGuard did not beat generic adversarial LoRA under the archive decision criterion;
- PatchGuard did not beat cutout/random-erasing;
- PatchGuard did not beat both generic adversarial LoRA and cutout/random-erasing.

Consequence: kill PatchGuard-VLA as the current RA-L method route. Do not proceed to PatchGuard STATE 2 or more PatchGuard training.

Interpretation: this kills the PatchGuard method claim, not the LoRA environment. The next valid step is standard SmolVLA LoRA baseline reproduction on an official or standard task split.

Execution boundary for this archive pass:

- experiments happened: no;
- training happened: no;
- GPU job happened: no;
- rollout/replay happened: no;
- downloads happened: no;
- OpenVLA-OFT happened: no;
- new method implementation happened: no;
- paper claims happened: no.
