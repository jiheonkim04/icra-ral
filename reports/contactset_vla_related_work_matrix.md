# ContactSet-VLA Related Work Matrix

| Work | Evidence | ContactSet-VLA relevance | Gap targeted |
| --- | --- | --- | --- |
| Direct Action-Head Injection of A Grounded 3D Point (2026), https://arxiv.org/abs/2606.27663 | Reports large LIBERO-PRO gains from lifting a grounded point into 3D and injecting it into the action head with AdaLN; GR00T-N1.6 average success improves from 31.2 to 77.5 under task perturbation and 28.1 to 60.2 under position perturbation. | Primary anchor. ContactSet-VLA keeps the action-head injection idea but expands the geometry from one point to a structured set. | A single target point may be insufficient for contact-rich pick/place stages, support surfaces, normals, and safety/avoidance. |
| LIBERO-PRO (2025/2026), https://arxiv.org/abs/2510.03827 | Introduces perturbations over manipulated objects, initial states, instructions, and environments to expose memorization and brittle generalization. | Defines the kind of spatial/task perturbation pressure that makes action-head geometry injection meaningful. | STATE 1 is not a LIBERO-PRO benchmark; it only prepares a local action-head diagnostic. |
| 2D language or visual prompting for target coordinates | The anchor paper reports these as weaker than 3D action-head injection under the same oracle point. | Required negative control for future scale-up, but not implemented in STATE 1. | ContactSet-VLA first asks whether richer 3D geometry beats the single 3D point before adding 2D prompt comparisons. |
| Dense 3D VLA/action-head methods | Prior work cited by the anchor paper injects richer 3D information into action heads, often with larger encoders. | Supports the idea that action prediction benefits from 3D action-head conditioning. | ContactSet-VLA aims for a lightweight set encoder rather than a dense scene encoder. |
| Simple object-center and destination-only baselines | Local killed-route history shows many methods collapse when simple geometry/action baselines are included. | These baselines are mandatory variants in STATE 1. | Full contact set is only useful if it beats source-only, destination-only, and source+destination injection. |

## Positioning

ContactSet-VLA is not a wrapper, shield, replay repair, retiming method, or data augmentation route. It is an action-head conditioning hypothesis: structured 3D contact geometry should be injected where actions are decoded, and it must beat single-point and simple point-set baselines before any full VLA training is justified.

