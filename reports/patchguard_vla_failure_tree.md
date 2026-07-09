# PatchGuard-VLA Failure Tree

Date: 2026-07-09 KST

## Root Outcome

`KILL_BASELINE_DOMINATED`

PatchGuard-VLA is archived as a main RA-L route because the method-specific LoRA variant did not beat the required simple baselines after the local environment was unblocked.

## Failure Tree

1. Was the physical-patch effect measurable?
   - Yes.
   - Evidence: STATE 1 max attacked policy-action L1 vs clean `0.181765`; max attacked translation-action L2 vs clean `0.213965`.
   - Result: continue.

2. Was a non-leaking kinematic/proprioceptive signal available?
   - Yes.
   - Evidence: local LIBERO HDF5 exposed EEF/joint/proprioceptive state, and STATE 1 recorded kinematic signal available.
   - Result: continue.

3. Was the prior `TOO_HEAVY_LOCAL` result a method kill?
   - No.
   - Evidence: STATE 1B installed and validated PEFT and bitsandbytes, then ran local SmolVLA LoRA injection and tiny training.
   - Interpretation: STATE 1 was environment-blocked, not method-killed.

4. Did PEFT/bitsandbytes/CUDA work locally?
   - Yes.
   - Evidence: PEFT `0.19.1`; bitsandbytes `0.49.2`; 4-bit and 8-bit CUDA smokes passed; PyTorch `2.10.0+cu128`; CUDA runtime `12.8`; RTX 5080.
   - Result: continue.

5. Did local SmolVLA LoRA injection work?
   - Yes.
   - Evidence: LoRA injected into `state_proj`, `action_in_proj`, and `action_out_proj` with `9984` trainable params.
   - Result: continue.

6. Did the tiny training smoke run and compute loss?
   - Yes.
   - Evidence: batch size 1, rank 4, 10 steps per variant; loss computed; VRAM peak `2224.845` MB; runtime `57.438` sec.
   - Result: continue to baseline gate.

7. Did PatchGuard beat generic adversarial LoRA?
   - No, under the archive decision criterion.
   - Evidence: generic adversarial LoRA metric `0.142803`; PatchGuard metric `0.13356`.
   - Interpretation: PatchGuard was not clearly separated from a generic adversarial augmentation LoRA control and did not earn a baseline-resistant method claim.

8. Did PatchGuard beat cutout/random-erasing?
   - No.
   - Evidence: cutout/random-erasing metric `0.02973`; PatchGuard metric `0.13356`.
   - Consequence: hard kill criterion triggered.

9. Can PatchGuard proceed to STATE 2?
   - No.
   - Reason: the environment works, so the remaining failure is method-level baseline dominance.

## Failure Diagnosis

PatchGuard found a real vulnerability signal and a real adapter path, but the kinematic consistency objective did not produce a baseline-resistant method signal in the smallest allowed LoRA smoke. The strongest surviving result is infrastructure, not a defense.

## Stop Rule

Do not run PatchGuard STATE 2, more PatchGuard training, rollout, OpenVLA-OFT, or a renamed PatchGuard-like defense route from this evidence. A future defense route would need a new predeclared official benchmark anchor and must first beat cutout/random-erasing and generic adversarial LoRA.
