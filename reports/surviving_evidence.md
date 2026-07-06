# Surviving Evidence

This file separates evidence that remains valid from claims that did not survive.

## Evidence That Survives

### Offline Fixed-Prior TCA

Fixed-prior TCA repeatedly outperformed ActionMap on bounded offline proxy splits when the target prior was correct.

This evidence is valid as offline proxy evidence only. It is not paper-grade rollout evidence.

### Prior-Source Audit

The prior-source audit passed and found no inference-time leakage for the fixed instruction-text target prior. The fixed prior did not use BDDL metadata, dataset target labels, eval labels, task IDs, filenames, or manifest target fields at inference.

### 7D Bridge

The raw LIBERO 7D action interface and bridge were validated. The project learned that silent 4D-to-7D padding is invalid, and the later diagnostics used explicit 7D actions.

### Expert Replay

Matched-init expert replay succeeded, validating the environment/action bridge under exact demonstration initial states. Default reset remained a separate mismatch and should not be confused with matched-init replay.

### Negative Online Action-Quality Result

The online 7D diagnostic head route was tested honestly. The best redesigned head improved over ActionMap but failed to beat a mean-action baseline. This is a useful negative result.

## Evidence That Does Not Survive As A Main Claim

### TCA-Select

TCA-Select showed no meaningful gain in the current diagnostics. It should be killed or de-emphasized as a core contribution.

### Representation Collapse

The representation-collapse claim is unsupported. The representation-sensitivity audit did not justify a hidden-collapse claim.

### Rollout-Level TCA Support

Closed-loop rollout support is not established. Fixed-prior TCA has no valid rollout-level support in the current low-compute path.

### RA-L Main Paper Claim

The current evidence package is insufficient for a main RA-L robotics-control submission.

