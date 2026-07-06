# CSS-Shield Risk Register

## Trivial Clipping Risk

Risk: CSS-Shield may be no better than action clipping.

Mitigation: clipping-only is a required baseline. Continue only if full shield beats clipping-only on semantic or safety metrics.

## Safe-Stop Collapse Risk

Risk: the shield may reject nearly every action and preserve no utility.

Mitigation: track intervention rate, utility preservation, reward/success when available, and target-directed movement.

## Synthetic Failure Overclaim Risk

Risk: randomized unsafe proposals may make the shield look useful while native VLA proposals are not realistic.

Mitigation: label synthetic proposal diagnostics separately and move quickly to native SmolVLA or simulator action sources.

## Object-Key Proxy Risk

Risk: object position keys may be missing or may not reflect task success.

Mitigation: report missing keys explicitly and treat object-distance movement as a diagnostic proxy, not standard success.

Current status: materialized in State 1. The intended object was available, but the counterfactual object was not present as an observation object key, so wrong-target semantic intervention was not exercised.

Next mitigation: choose or construct the smallest bounded diagnostic where both intended and counterfactual object keys are observable, or label semantic shielding blocked.

## Safety-Only Equivalence Risk

Risk: full CSS-Shield may reduce unsafe actions only because of safety damping, not because of semantic counterfactual reasoning.

Current status: active. In State 1, full CSS-Shield beat clipping-only on unsafe rate but did not beat safety-only.

Mitigation: State 2 must include a semantic wrong-target setting where safety-only is insufficient.

## Old TCA Route Revival Risk

Risk: CSS-Shield could drift back into target-prior TCA-Map claims.

Mitigation: keep TCA route killed for RA-L-stable submission and frame CSS-Shield as runtime intervention.

