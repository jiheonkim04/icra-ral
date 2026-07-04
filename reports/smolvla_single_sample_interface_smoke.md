# SmolVLA Single-Sample Interface Smoke

## Purpose

This is a bounded engineering smoke after the SmolVLA load-only smoke has passed. It validates that one synthetic observation can be converted into the policy input interface and produce one action tensor locally.

It is not a benchmark, not a rollout, not training, not a dataset evaluation, and not paper-grade evidence.

## Scope

Allowed inside this task only:

- local SmolVLA checkpoint load,
- local tokenizer/config files,
- one synthetic state/image/text batch,
- one CPU `select_action` call,
- runtime and memory logging.

Forbidden:

- downloads,
- OpenVLA-OFT execution,
- LIBERO/RoboSuite/RoboCasa/dataset access,
- simulator execution,
- rollouts,
- training,
- token or secret access,
- paper-level claims.

## Command

Run only inside the bounded single-sample task:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_SINGLE_SAMPLE_INFERENCE="1"
powershell -ExecutionPolicy Bypass -File scripts\28_smolvla_single_sample_interface_smoke.ps1
Remove-Item Env:\ALLOW_SINGLE_SAMPLE_INFERENCE -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
```

The script writes the ignored runtime report:

```text
reports\smolvla_single_sample_interface_report.json
```

## Pass Criteria

- local files are complete,
- runtime dependencies are present,
- output action shape matches the configured action dimension,
- output values are finite,
- adapter metadata is recorded for synthetic state/image/action interface mapping,
- no download, training, rollout, simulator, OpenVLA-OFT, token, or dataset behavior occurs,
- runtime stays under 10 minutes,
- measured CUDA allocation stays under 14GB.

## Next Step

If this passes, continue to tiny feature-cache/interface validation. Still do not train, rollout, use real benchmark data, or make paper claims.

## Latest Local Result

The bounded smoke passed on CPU with one synthetic observation:

```text
load_and_interface_elapsed_sec=29.484
single_sample_inference_elapsed_sec=1.719
action_shape=[1, 6]
action_finite=true
adapter_metadata_recorded=true
cuda_max_allocated_mb=0.0
downloads_performed=false
training_performed=false
real_rollouts_performed=false
openvla_oft_executed=false
```

Adapter metadata recorded:

- state adapter: `diagnostic_eef_pos_quat_xyz_6d_state_adapter`,
- image adapter sources: `agentview_image`, `robot0_eye_in_hand_image`, `agentview_image`,
- action adapter: `policy_6d_delta_pose_plus_gripper_zero_hold`,
- diagnostic adapted action dim: `7`,
- implicit padding performed: `false`,
- truncation performed: `false`.
