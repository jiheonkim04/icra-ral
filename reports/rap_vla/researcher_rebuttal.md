# RAP-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `RAP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Researcher A accepts Reviewer B's conditional pass in full. RAP remains a
single-mechanism method: retrieved legal action anchors plus bounded
residualized action-flow learning. It does not rescue VDR, KITE, RAR, LIFT,
EAC, HEST, HASTE, or COVI.

## Accepted Constraints

### OptimusVLA Proximity

RAP will not claim to invent retrieval memory for VLA policies. The novelty
claim is narrower: RAP residualizes the trainable action-flow path around a
retrieved legal action anchor and tests whether that learned residual is useful
beyond a transparent OptimusVLA memory-prior proxy and direct anchor-only
retrieval.

The first serious comparison remains exactly:

1. `smolvla_base`;
2. `optimusvla_memory_prior_proxy`;
3. `rap_full`;
4. `rap_anchor_only_no_residual`;
5. `standard_lora`.

### Memory Separation

Discovery/training demonstrations only may enter the RAP memory used by the
candidate. Validation rows may be queried for scoring but may not be inserted
into the candidate memory. Confirmatory task/reset identities may not be
embedded, indexed, inspected, or used to tune memory construction.

Before any training, RAP will persist:

- memory row keys;
- source HDF5 paths and demo ids;
- split labels;
- feature normalization hash;
- retrieval metric, top-k, task/phase filters, and distance calibration;
- duplicate, missing, extra, frame-overlap, and split-overlap audit.

### Headroom And Direct-Retrieval Explanation

RAP accepts that the method stops before rollout if retrieved anchors fail to
beat task/phase means or if anchor-only retrieval explains the available signal.

Stage 0 must show:

- retrieved anchors beat task/phase mean chunks by the preregistered margin;
- residual targets retain positive variance after anchor subtraction;
- a legal deployment-input residual probe beats zero residual;
- RAP full differs from anchor-only/no-residual before rollout.

### Action Validity Unit System

RAP will report both normalized chunk validity and postprocessed 7D LIBERO
validity before rollout. The hard no-go gate is postprocessed 7D action
validity, because that is the deployment action sent to the environment.
Normalized chunk validity remains a diagnostic and scale-audit metric.

No clipping, threshold widening, unit-system switch, or post-hoc validity
reinterpretation is allowed after Stage 0 begins.

For mechanism smoke, RAP will report:

- Base action;
- retrieved anchor action;
- RAP action;
- residual norm;
- gate value;
- retrieval confidence;
- top-k diversity;
- dimensions changed;
- validity context.

### OptimusVLA Proxy Fidelity

Before freezing the executable Stage 0 protocol, RAP will check whether the
official OptimusVLA released assets can be installed and used within local
budget. If not, policy 2 remains explicitly named a transparent proxy, not an
official reproduction.

The proxy will match RAP's memory sources, retrieval features, action
postprocessing, task/reset manifest, and inference budget as far as locally
possible. Any deviation from official OptimusVLA must be listed before Stage 0
execution and cannot be changed after validation outcomes are seen.

### Standard LoRA

Matched standard LoRA remains mandatory because RAP updates policy weights. If
standard LoRA matches or beats RAP in the first serious comparison, RAP does
not become a paper candidate.

The standard LoRA policy must match RAP's demonstrations, optimizer steps,
rank, target modules, ordinary flow objective, clean-retention policy where
applicable, and checkpoint-selection budget.

### Memory Overhead

RAP will report:

- index row count;
- feature dimension and dtype;
- memory-action bytes;
- index bytes;
- retrieval latency;
- total policy latency;
- peak allocated GPU memory;
- retrieval frequency.

Timing, throughput, resource-utilization, and latency measurements overlapping
or unresolved against resource-contention intervals remain ineligible for final
paper evidence.

## Rebuttal To Reviewer Concerns

RAP accepts that a direct nearest-action memory can be a strong simple
explanation. This is why anchor-only/no-residual is not optional. RAP also
accepts that OptimusVLA's official result is strong enough that a weak local
proxy would be unfair. The proposal therefore uses official assets when
locally feasible, otherwise it labels the comparison as a transparent proxy
and freezes all deviations before development results are interpreted.

RAP's residual path remains meaningful only if the anchor explains coarse
task/phase action mode and the learned residual explains current-state
deviation. If either side collapses, Stage 0 must stop with the correct
failure class rather than proceeding to rollout.

## Decision

RAP proceeds to mathematical mechanism audit and preregistration. The audit must
formalize:

- retrieval feature variables and memory partitions;
- anchor distribution construction;
- residual target formula and tensor shapes;
- residual gate initialization and bounds;
- objective scales and gradient paths;
- OptimusVLA proxy and anchor-only ablation formulas;
- postprocessed action-validity gate;
- required Stage 0 ablations and stop classes.

No training, validation search, rollout, simulator access, or confirmatory-test
access is authorized by this rebuttal.
