# A2C2 fidelity-corrected actual-path smoke: attempt 2

Date: `2026-07-19 KST`

Classification: `PRIOR_ACTUAL_PATH_PREFLIGHT`

Fidelity label: `A2C2_FIDELITY_CORRECTED_LOCAL_PORT`

Decision: `A2C2_CORRECTED_ACTUAL_PATH_SMOKE_FAIL`

This attempt proved that both the paired Base and public prior strict-load on
CUDA from the checkpoint-compatible author source. It then stopped before
environment reset because the historical author config is a dataclass without
the later `to_dict()` reporting API. No scientific outcome was persisted or
counted.

The reporting fix is a continuation of the same serializer compatibility
repair: use `dataclasses.asdict`, as the historical author code itself does.
It changes no model tensor, input, queue, task, reset, delay, action, metric, or
decision rule.

The complete attempt is preserved under
`runs/a2c2_fidelity_corrected/a2c2_corrected_smoke_repair1_20260719t1826`.
