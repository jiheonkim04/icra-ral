# FEDO-VLA Reviewer Attack

Date: `2026-07-12 KST`

## Verdict

Decision: `APPROVE_ONLY_WITH_APEX_AND_STATIC_BASELINE_KILL_GATES`

FEDO-VLA is allowed as Cycle 2 only because it changes the core problem and mechanism relative to DICD-VLA and ECHO. It is high-risk because APEX is a very close direct prior.

## Closest Papers

1. APEX, `https://arxiv.org/html/2606.16504`: plug-and-play adaptive policy execution for policy/controller tracking error.
2. RobustVLA, `https://arxiv.org/html/2510.00037v4`: robustness against multi-modal perturbations, including action noise.
3. VLA-Corrector, `https://arxiv.org/abs/2607.01804`: online visual-drift detection and corrective replanning for action chunks.
4. Real-time Correction for VLA Action Chunks, `https://arxiv.org/html/2509.23224v1`: correction of action chunks under real-time observation mismatch.
5. APEX-adjacent low-level residual and adaptive-control policies cited by APEX.

## Main Attacks

1. APEX may already own the policy-controller execution-gap claim.
2. A static inverse-gain baseline may explain all improvement under a simple synthetic fault.
3. A generic integral-error feedback controller may match any learned disturbance observer.
4. If the fault is hand-designed to favor phase features, the result may be benchmark engineering rather than a method.
5. If measured realized action is computed from simulator internals rather than low-level feedback, the method is privileged.

## Mandatory Prototype Baselines

- faulted frozen SmolVLA;
- static inverse-gain compensation;
- APEX-style error-feedback proxy;
- FEDO no-phase/error-feedback ablation;
- full FEDO.

## Required Evidence

The prototype must report:

- closed-loop task success;
- task-balanced success;
- per-task success;
- clean no-fault retention;
- command/residual magnitudes;
- whether the measured realized action is available without simulator state.

## Kill Decision Rules

FEDO is killed if static inverse gain, APEX-style feedback, or no-phase/error ablation matches or beats full FEDO. A narrow positive over only faulted frozen SmolVLA is insufficient.
