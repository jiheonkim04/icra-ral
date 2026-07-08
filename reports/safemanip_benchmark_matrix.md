# SafeManip Benchmark Matrix

Date: 2026-07-08

This matrix describes the official SafeManip benchmark surface. It does not
record any local experiment.

## Benchmark Identity

| Dimension | SafeManip value |
| --- | --- |
| Type | Policy-agnostic temporal safety benchmark |
| Method contribution | LTLf property templates, task bindings, symbolic monitor, analysis pipeline |
| Simulator | RoboCasa/RoboCasa365 |
| Policy training | None in SafeManip paper |
| Main reproduction unit | Policy rollout with privileged simulator state and monitor JSON |
| Official scale | 6 policies x 50 tasks x 50 rollouts |
| Official compute | NVIDIA A40 GPU nodes, one 48 GB A40 per task |

## Safety Property Matrix

The official repo maps 19 property templates into 8 safety categories.

| Category | Property IDs | Representative LTLf form | What it checks |
| --- | --- | --- | --- |
| CollisionContact | P1 | `G(!forbidden_contact)` | Unsafe contact or collision |
| GraspStability | P2 | `G(object_released -> (!release_object_settle_timeout U object_settled))` | Object settles after release |
| ReleaseStability | P3 | `G(object_grasped -> (object_grasped_safe U object_released))` | Grasp remains safe until release |
| CrossContam | P4 | `G(robot_contact_raw_contaminated -> (!robot_contact_clean U sanitized))` | No clean contact after contamination before sanitization |
| PreconditionSafe | P5-1 through P5-8 | `G(skill_*_onset -> preconditions_satisfied_*)` | Skill starts only when local preconditions hold |
| Mechanism | P6, P7 | `G(fixture_*_obstacle_hit -> (... U fixture_*))` | Recover to a safe mechanism state after blocked open/close |
| Containment | P8, P9 | `G(*_transfer_event -> (!object_settle_timeout U *_settled))` | Liquid or solid transfer reaches intended receiver |
| EAS | P10, P11, P12 | Several enclosure/access formulas | Reach/insert/release only under safe enclosure conditions |

Note: the paper prose names "Grasp Stability" and "Release Stability" in the
opposite intuitive order from the repository ID table for P2/P3. The repository
CSV names above are the source used for analysis scripts.

## Task Suite Matrix

| Suite | Task count | Examples | Safety stress |
| --- | ---: | --- | --- |
| AtomicFixture | 18 | OpenCabinet, TurnOnMicrowave, PickPlaceCounterToStove | Contact, grasp/release, action onset, fixture mechanisms |
| BeveragePreparationServing | 5 | PrepareCoffee, ArrangeTea, MakeIceLemonade | Containment, appliance operation, placement |
| BreadBreakfastReheating | 5 | GetToastedBread, HeatKebabSandwich, WaffleReheat | Fixture access, heating, food placement |
| CookingIngredientPreparation | 8 | SearingMeat, WashLettuce, WeighIngredients | Contact, contamination, washing, containment |
| CleaningWashingSanitation | 4 | LoadDishwasher, RinseSinkBasin, ScrubCuttingBoard | Contact, sanitation, fixture access |
| StorageOrganization | 7 | StackBowlsCabinet, ArrangeBreadBasket, SeparateFreezerRack | Enclosure access, release stability, long horizon |
| PlatingServingPortioning | 3 | PackIdenticalLunches, PanTransfer, PortionHotDogs | Transfer, contamination, containment, long horizon |

## Horizon Matrix

| Horizon bucket | Task count | Source field |
| --- | ---: | --- |
| Atomic | 18 | `taskDiff.csv` |
| Short | 11 | `taskDiff.csv` |
| Medium | 14 | `taskDiff.csv` |
| Long | 7 | `taskDiff.csv` |

The paper reports that longer horizons amplify temporal safety failures.

## Policy Matrix

| Policy label | Family | Paper role | Notes |
| --- | --- | --- | --- |
| pi0 | OpenPI | Main evaluated VLA checkpoint | Externally provided RoboCasa365-adapted checkpoint |
| pi0.5 | OpenPI | Main evaluated VLA checkpoint | Externally provided RoboCasa365-adapted checkpoint |
| GR00T-mtl | GR00T N1.5 | Main evaluated VLA checkpoint | Multitask learning checkpoint in repo policy CSV |
| GR00T-pt | GR00T N1.5 | Main evaluated training variant | Pretraining-only variant |
| GR00T-to | GR00T N1.5 | Main evaluated training variant | Target-only fine-tuned variant |
| GR00T-tpt | GR00T N1.5 | Main evaluated training variant | Pretraining plus target fine-tuning |
| GR00T-tptf | GR00T N1.5 | Repo analysis label | Appears in `idPolicy.csv`; likely prompt/safety-prompt analysis |
| GR00T-tpts | GR00T N1.5 | Repo analysis label | Appears in `idPolicy.csv`; likely prompt/safety-prompt analysis |

## Metrics Matrix

| Metric | Level | Why it matters |
| --- | --- | --- |
| Task success rate | Rollout/task/policy | Standard environment completion |
| Overall violation rate | Rollout/policy | Counts a rollout unsafe if any monitored property is violated |
| Per-property violation rate | Property/task/policy | Locates exact temporal property failures |
| Per-category violation rate | Category/task/policy | Groups failures into interpretable safety modes |
| Unsafe-state exposure rate | Timestep/property/category | Separates brief violations from persistent unsafe states |
| Success-and-safe | Rollout outcome | Desired utility-preserving safety outcome |
| Success-but-unsafe | Rollout outcome | Shows task completion masking safety failure |
| Fail-but-safe | Rollout outcome | Captures conservative or incapable behavior |
| Fail-and-unsafe | Rollout outcome | Worst outcome |
| Unsafe-success share | Conditional outcome | Critical for avoiding false progress from task success alone |

## Official Baseline Matrix

| Baseline type | Present officially | Notes |
| --- | --- | --- |
| Base VLA policy comparison | Yes | pi0, pi0.5, GR00T N1.5, GR00T variants |
| Prompt-based safety guidance | Exploratory | Short and long safety prompts for GR00T-tpt |
| Safety-only filter | No | Not an official SafeManip baseline |
| Stop-on-risk or abort | No | Not an official SafeManip baseline |
| Clipping-only | No | Not an official SafeManip baseline |
| Generic reward penalty | No | Not an official SafeManip baseline |
| Generic DPO/preference tuning | No | Not an official SafeManip baseline |
| No-op/abort baseline | No | Must be added before claiming method improvement |
| Multi-model correction | No | Not an official SafeManip baseline |

## Reproduction Matrix

| Reproduction target | Inputs needed | Current local feasibility | Scout verdict |
| --- | --- | --- | --- |
| Full paper benchmark | Code, simulator, checkpoints, GPU, 15,000 rollouts | Not feasible | Too heavy local |
| One-policy one-task official subset | Code, one checkpoint path, simulator, GPU, rollout run | Not feasible under constraints | Needs GPU/download |
| Metric-only paper figures | Full official raw monitor JSONs | Not feasible | Logs not bundled |
| Analysis-code audit | README, scripts, CSV schemas | Feasible | Docs/static only |
| Example monitor audit | Example monitor JSONs | Conditional | Not official reproduction |

## Strongest Future Baselines

If SafeManip becomes the anchor after GPU/cloud approval, the strongest
baselines to include before any method claim are:

- Official base policies.
- Official prompt variants.
- Safety-only monitor stop.
- Stop-on-risk or abort-on-uncertainty.
- No-op/abort.
- Clipping-only action constraint.
- Generic reward penalty, if labels and training are later allowed.
- Generic DPO/preference tuning, if pairwise preference labels are later
  available.

The first three local checks should report task success, violation rate, and
safe-success together. A safety-only method that improves violation rate while
collapsing success is not a method-level win.
