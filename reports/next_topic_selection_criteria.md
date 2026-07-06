# Next Topic Selection Criteria

Any new topic must pass these gates before it becomes the main RA-L route.

## Required Early Gates

- Produce a rollout metric within 48 hours.
- Show a nontrivial baseline gap within 72 hours.
- Beat a simple baseline, not only no-method or no-shield.
- Avoid reliance on offline-only proxy metrics.
- Avoid reliance on native policy competence unless competence is verified first.
- Define kill criteria before implementation.

## Novelty Requirements

- Clear novelty against recent VLA, robot safety, action-decoder, and runtime intervention papers.
- Clear robotics evidence path for RA-L.
- A baseline list that includes simple alternatives such as mean-action, clipping-only, safety-only, and native-only where relevant.

## Execution Requirements

- Run the smallest simulator or rollout diagnostic first.
- Keep planner/report work bounded until a metric, loss, rollout result, or concrete blocker exists.
- Log weak and negative results.
- Do not change metrics, splits, or baselines after seeing results unless the change is marked exploratory.

## Preferred Topic Shape

The next topic should be rollout-first and baseline-first. It should fail quickly if it cannot beat a simple baseline.

