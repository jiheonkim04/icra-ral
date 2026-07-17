# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Branch base: audit commit `b0ecb6ea5f6eba2953b5bd842883c0474d634dff`
- Current epoch: 5
- Current cycle: 0
- Current stage: `epoch_5_official_prior_ecosystem_selection`
- Current decision: `STRATEGIC_NEW_EPOCH_OFFICIAL_PRIOR_FIRST`
- Previous method: `MCI-VLA`
- Previous decision: `MCI_STAGE_0_IMPLEMENTATION_FAILURE`
- MCI rescue/retune: prohibited and not performed
- Cycle 39 ordinary local-method search: superseded by strategy reset, not a scientific kill

## Audit Anchor

- Full audit: `reports/autonomous_research_full_history_audit.md`
- Audit branch: `codex/full-history-audit-before-resume`
- Audit commit: `b0ecb6ea5f6eba2953b5bd842883c0474d634dff`
- Audit totals accepted by user: 73 routes, 47 formal methods, 31 trained/checkpointed routes, 17 Stage A, 10 Stage B, 0 GO, 0 second-backbone Ours, 0 official prior reproductions.

## Epoch 5 Artifacts

- Ecosystem selection: `reports/epoch5_prior_ecosystem_selection.md`
- Reproduction plan: `reports/epoch5_prior_reproduction_plan.md`
- Reproduction result: `reports/epoch5_prior_reproduction_result.md`
- Reproduction result JSON: `reports/epoch5_prior_reproduction_result.json`

## Selected Prior Ecosystem

Selected: OpenVLA-OFT on LIBERO.

Why: public primary paper, MIT official code, official LIBERO checkpoints, local checkout, local 15G checkpoint, and existing validated INT4 hard-slice run.

Focused validation command passed:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_openvla_oft_int4_gate.py -q
```

Observed: `4 passed in 0.10s`.

## Current Scientific Meaning

OpenVLA-OFT INT4 prior evidence is positive on the recovered hard-slice condition:

- OpenVLA-OFT INT4: 20/20.
- SmolVLA frozen-base exact-init: 11/20.

But this condition is saturated by the prior. There is no residual gap on that condition, so Ours design is still blocked.

## Next Action

Preregister and run a bounded residual-gap diagnostic for OpenVLA-OFT. Candidate residuals must come from official-prior limits or benchmark stressors, not a new acronym:

- LIBERO-PRO-style perturbation if locally accessible;
- official LIBERO task subset where OpenVLA-OFT is not saturated;
- language-grounding or visual-feedback stressor motivated by OpenVLA-OFT's paper discussion.

If no residual remains, move to second-ranked ecosystem: pi0.5/OpenPI.

## Prohibitions

- Do not design Ours yet.
- Do not generate three local method candidates.
- Do not reopen CAVM, CALA, RAR, MCI, CSPR, or governance-closed prior routes.
- Do not create cached-feature residuals, frozen-policy gates, history heads, verifiers, visual canonicalizers, memory lookups, or proxy-only prior methods.
- Do not claim INT4 OpenVLA-OFT is full-precision reproduction.
