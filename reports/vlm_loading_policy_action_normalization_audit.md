# VLM Loading Policy and Action-Normalization Audit

This report defines the next conservative diagnostic after weak one-sample offline SmolVLA action decoding.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\108_plan_vlm_loading_policy_action_normalization_audit.ps1
```

The audit reads local SmolVLA config/processor metadata and existing diagnostic reports only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Expected interpretation:

- `decision=no_go_rollout_scaling`: learned-policy rollout scaling remains blocked.
- `ready_for_repeated_offline_decoding_plan=true`: the next safe step is a tiny repeated offline action-decoding plan over a few local LIBERO HDF5 timesteps.
- VLM-enabled loading, full SmolVLM2 weight acquisition, simulator rollout scaling, and paper-grade claims remain separate risk-assessed tasks.

The audit should explicitly record:

- whether `config.json` requests `load_vlm_weights=true`,
- whether the observed local diagnostic used `load_vlm_weights=false`,
- whether the external SmolVLM dependency is tokenizer/config-only,
- ACTION/STATE normalization policy,
- processor safetensor files used for normalization/unnormalization,
- policy 6D action versus LIBERO 7D expert action convention,
- clipping in the 6D-to-7D action adapter,
- camera aliasing and image resize metadata.

If the audit reports weak alignment with disabled VLM loading and action clipping, the next action is not another rollout. The next action is a bounded offline diagnostic plan that compares a few HDF5 expert actions against decoded actions while logging load policy, normalization, clipping, gripper strategy, and image aliases.
