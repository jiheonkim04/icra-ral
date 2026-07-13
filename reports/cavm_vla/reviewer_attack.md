# Reviewer B Attack: CAVM-VLA

Date: 2026-07-13 KST

Proposal hash under review: `849A98B2F137FC43EAA68C7B7D7DB246FEF58DD2EDBBD1F8869C4BA092DE68F2`

## Closest Three Current Papers

### 1. Retrieve-then-Steer / Online Success Memory

URL: https://arxiv.org/html/2605.10094v1

Overlap:

- frozen generative VLA;
- deployment memory;
- retrieval of state-relevant action chunks;
- memory as non-parametric action prior;
- no parameter update.

Difference claimed by CAVM:

- Retrieve-then-Steer builds success memory and filters consistency among successful candidates.
- CAVM explicitly retrieves both successful and failed traces and estimates a local success-minus-failure action direction.

Reviewer risk:

If the failed-trace term does not beat success-only memory under the same retrieval kernel, CAVM is not a method-level contribution.

### 2. HELM

URL: https://arxiv.org/abs/2604.18791

Overlap:

- uses episodic memory;
- uses success/failure evidence;
- targets long-horizon VLA execution failures;
- adds an external control layer around a frozen VLA.

Difference claimed by CAVM:

- HELM centers a memory-conditioned state verifier and rollback/replanning harness.
- CAVM must not verify or replan; it modifies the action vector through an outcome-contrastive memory prior.

Reviewer risk:

If CAVM's contrast gate becomes a failure detector or a replan trigger, it collapses back into HELM/RCV-style verification.

### 3. Harness VLA / LaMem-VLA Memory Line

URLs:

- https://arxiv.org/abs/2607.08448
- https://arxiv.org/abs/2607.07608

Overlap:

- frozen or memory-augmented VLA;
- historical trace evidence;
- memory influences action generation under long-horizon or perturbed deployment.

Difference claimed by CAVM:

- Harness VLA composes analytic primitives and learns operating ranges.
- LaMem-VLA weaves memory into native latent tokens.
- CAVM is a lightweight external 7D action-prior field, feasible for the current frozen SmolVLA runner.

Reviewer risk:

The local implementation may be less expressive than these priors and may not deserve a paper claim unless the contrastive negative-memory component produces a clear closed-loop gain.

## Simplest Equivalent Method

Success-only nearest-neighbor action blending:

`a'_t = (1 - lambda) a_t + lambda mu_success(z_t)`.

This is the most dangerous equivalent baseline. It uses the same memory, same distance metric, same held-out identities, and same action clipping but removes the failure memory. If this matches CAVM, the proposed contrastive mechanism is dead.

## Strongest Simple Killer Baseline

`nearest_success_replay`: retrieve the single nearest successful same-task record and blend directly toward its action when within the same density gate.

This baseline can win if the task/reset structure makes successful traces almost deterministic. If it matches CAVM, the extra contrastive machinery is unnecessary.

## Mathematical Risks

1. Terminal success labels are coarse. A failed episode can contain locally good actions before the failure point. CAVM must not assume every action in a failed episode is bad.
2. State/action distance may retrieve phase-mismatched records. The chunk fraction helps but may be insufficient.
3. The failure mean can repel away from useful but common approach actions if success/failure diverge only near the end.
4. The contrast magnitude `||mu+ - mu-||` could reflect task identity or reset geometry rather than a local action-value direction.
5. There is no probability distribution in the proposal, so no KL/entropy objective is justified.

## Leakage Risks

- Do not use simulator object poses, rewards before terminal success, or privileged task progress at inference.
- Do not use Stage 2 held-out identities in memory construction, standardization, bandwidth selection, or margin selection.
- Do not select tasks or resets from observed CAVM outcomes.
- Do not use successful traces from the exact held-out test identity.

## Required Reviewer Conditions Before Implementation

Proceed only if preregistration includes:

1. fresh identity base distinct from RCV and PSE;
2. Stage 0 hard kill for no success/failure mixture or no local action separation;
3. fixed variants: frozen, success-only proxy, nearest-success replay, no-failure ablation, full CAVM;
4. exact split of acquisition, calibration, Stage 2A, Stage 2B identities;
5. no privileged inference features;
6. leak tests for identity overlap and feature schema;
7. a direct rule that CAVM is killed if no-failure ablation or nearest-success replay matches it.

## Reviewer Verdict

`CONDITIONAL_PROCEED`.

The proposal is not an exact duplicate because the closest success-memory prior discards failed traces instead of using them as local negative action evidence. The formulation is mathematically simple and locally feasible. It remains high-risk: if the failure-memory term does not matter, the method must be killed without rescue.
