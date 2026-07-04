# Offline Demonstration Action Decoding

This bounded diagnostic reads one local LIBERO HDF5 observation/action pair and runs exactly one local CPU SmolVLA action decode. It compares the decoded action to the expert action without creating a simulator environment.

Command:

```powershell
$env:ALLOW_OFFLINE_DEMO_ACTION_DECODING="1"
powershell -ExecutionPolicy Bypass -File scripts\106_bounded_offline_demo_action_decoding.ps1
Remove-Item Env:\ALLOW_OFFLINE_DEMO_ACTION_DECODING -ErrorAction SilentlyContinue
```

This diagnostic must not download, install, train, create simulator environments, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims. It is only a one-sample offline action-decoding check.
