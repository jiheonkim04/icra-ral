# ContactTube-Aug Autopilot State

Branch: `codex/contacttube-aug-state0-state1`.

Start point: local `main` at `43cb59b` (`Archive TL-ChunkRepair route`).

STATE 0 status: initialized concise docs and immediately continued to STATE 1.

STATE 1 status: bounded replay smoke completed and killed before training.

Safety boundary:

- downloads: no,
- GPU: no,
- training: no,
- loss computation: no,
- OpenVLA-OFT: no,
- VLA model loading/inference: no,
- simulator replay: only with `ALLOW_CONTACTTUBE_AUG_STATE1=1` after risk assessment.

Current STATE 1 command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\180_contacttube_aug_state1_smoke.ps1
```

Bounded replay command:

```powershell
$env:ALLOW_CONTACTTUBE_AUG_STATE1="1"
powershell -ExecutionPolicy Bypass -File scripts\180_contacttube_aug_state1_smoke.ps1
Remove-Item Env:\ALLOW_CONTACTTUBE_AUG_STATE1 -ErrorAction SilentlyContinue
```

## STATE 1 Result

- replay happened: yes, bounded LIBERO/RoboSuite diagnostic,
- total simulator steps: `1621`,
- variants: `6`,
- exact-init no-op upper bound success: `true`,
- HDF5 object pose available: `false`,
- runtime object pose available: `true`,
- ContactTube-Aug controller-valid action rate: `0.849265`,
- ContactTube-Aug clip-step rate: `0.150735`,
- ContactTube-Aug tube score: `0.015226`,
- simple object-relative tube score: `0.009154`,
- ContactTube-Aug beats random action jitter: `true`,
- ContactTube-Aug beats random pose jitter: `true`,
- ContactTube-Aug beats simple object-relative retargeting: `false`,
- decision: `kill`,
- next state: `archive_or_reframe_contacttube_aug_before_training`.

The failed controller-invalid replay is preserved as the canonical STATE 1 result in:

- `reports/contacttube_aug_state1_result.json`,
- `reports/contacttube_aug_state1_result.md`.
