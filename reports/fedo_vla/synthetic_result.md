# FEDO-VLA Prototype Result

Date: `2026-07-12`

Final decision: `SYNTHETIC_MECHANISM_PASS`

- mode: `synthetic`
- training happened: `True`
- closed-loop experiment happened: `False`
- full checkpoint: `reports\fedo_vla\checkpoints\fedo_synthetic_full.pt`
- full checkpoint sha256: `87ce5c89c6d03591da89d281c9c49f0b2a0370bf88de4d13aebda869929de7d0`
- no-feedback checkpoint: `reports\fedo_vla\checkpoints\fedo_synthetic_no_feedback.pt`
- no-feedback checkpoint sha256: `bf1414cb0161b10282545f341c9be547efa06f9e8735e6526b4070e59c2f5db4`
- summary: `{'mean_realized_error_norm': {'apex_feedback_proxy': 0.097144, 'faulted_frozen_smolvla': 0.126658, 'fedo_full': 0.058561, 'static_inverse_gain': 0.051621}, 'synthetic_passed': True}`
- elapsed seconds: `1.696`

Next step: Run real trace training.
