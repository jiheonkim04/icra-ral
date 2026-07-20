# Latent-Dynamics Attribution Closure

Decision: `NO_LEGAL_HEADROOM` at the pre-policy expert-feasibility gate.

The outcome-free simulator preflight passed on all four frozen tasks. The repaired protocol reused the exact cached post-settle observation, preserved qpos/qvel and render-defining model hashes, changed only the named dynamics arrays, and queried no policy, reward, or success endpoint.

The subsequent feasibility oracle followed the frozen standard-only selection rule: demonstrations were tested in numeric order under standard dynamics, and the lowest-index successful standard replay was selected before its intervention outcome was known. All four selected trajectories succeeded under standard dynamics and contacted the intended target. Under the frozen intervention, the same trajectories succeeded for drawer opening and bowl placement, but failed for plate pushing and stove-button activation despite target contact.

Therefore only two tasks spanning two collapsed families (`articulated` and `pick_transport_place`) were eligible. The paperability contract required at least three tasks spanning `articulated`, `planar_push`, and `pick_transport_place`. This feasibility and coverage failure is a kill condition, and bounded expansion is illegal because expansion required every validity, competence, feasibility, and coverage gate to pass.

No X-VLA dynamics rollout occurred, no policy was loaded or queried, and no Ours design or training was authorized. The result does not show that the altered push or button tasks are physically impossible; it shows only that the frozen legal headroom oracle did not verify them. Selecting alternate demonstrations based on altered outcomes, weakening intervention severity, deleting the missing family, or recasting the benchmark as a dynamics method are prohibited rescues.
