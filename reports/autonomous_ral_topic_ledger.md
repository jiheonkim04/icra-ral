# Autonomous RA-L Topic Ledger

## Cycle 1

Topic: delay-indexed action-chunk deployment

Method: `DICD-VLA`

Status: `KILLED_VALID_PROTOTYPE`

Distinctness:

- core problem: controlled execution delay
- representation: delay-indexed chunk features plus executed-action history
- action-generation mechanism: adapter changes deployed action instead of ranking existing candidates

Forbidden rescues still excluded:

- ECHO candidate ranking or verifier rescue
- PhaseBarrier threshold tuning or rename
- CensorCredit relabeling or hold-strength tuning
- ISAC intervention-set fine-tuning without real paired intervention/correction data

Cycle 1 Stage A kill:

- decision: `SIMPLE_BASELINE_EXPLAINS_METHOD`
- result file: `reports/dicd_vla/stage_a_result.json`
- full DICD: `1 / 10`
- direct chunk-index delay: `2 / 10`
- frozen delay-only baseline: `2 / 10`
- no-history ablation: `1 / 10`
- exceptions: `0`
- mechanism active: `true`

The delay-indexed history-conditioned adapter must not be revived through threshold tuning, longer training, or cosmetic changes. A new cycle must change at least two of the core problem, representation, training signal, objective, action-generation mechanism, and closed-loop intervention.

## Cycle 2

Topic: feedback execution-disturbance observer

Method: `FEDO-VLA`

Status: `KILLED_VALID_PROTOTYPE`

Distinctness versus Cycle 1:

- core problem changes from execution delay to low-level action-realization disturbance
- representation changes from delay-indexed chunk features to command/realized-action feedback
- action-generation mechanism changes from delayed chunk adapter to residual command compensation
- reviewer-killer direct prior changes from chunk-index delay to APEX-style feedback

Required kill gates:

- static inverse gain matches or beats full: `SIMPLE_BASELINE_EXPLAINS_METHOD`
- APEX-style feedback matches or beats full: `DIRECT_PRIOR_EXPLAINS_METHOD`
- no-feedback ablation matches or beats full: `KEY_COMPONENT_NOT_USEFUL`

Cycle 2 Stage A kill:

- decision: `CLEAN_RETENTION_FAILURE`
- result file: `reports/fedo_vla/stage_a_result.json`
- faulted frozen SmolVLA: `0 / 10`
- static inverse gain: `2 / 10`
- APEX-style feedback proxy: `2 / 10`
- FEDO no-feedback ablation: `2 / 10`
- FEDO full under faults: `1 / 10`
- clean frozen SmolVLA: `4 / 10`
- clean FEDO full: `0 / 10`
- exceptions: `0`

FEDO-VLA must not be revived through more epochs, gain retuning, alternate thresholds, or another residual command-compensation wrapper. A new Cycle 3 method must change at least two of the core problem, representation, training signal, objective, action-generation mechanism, and closed-loop intervention.

## Cycle 3

Topic: geometric-continuity anchored perception

Method: `GCAP-VLA`

Status: `STAGE_A_READY_FINAL_ALLOWED_CYCLE`

Constraint:

Cycle 3 is the last allowed distinct method cycle under the governance correction. It must not be a renamed DICD, FEDO, ECHO, candidate-ranking, adaptive-chunk, or residual-command-compensation method.

Distinctness versus prior cycles:

- core problem changes to partial camera occlusion and missing interaction-region geometry
- representation changes to image validity masks, temporal repaired camera tensors, and Sobel geometric anchors
- action-generation mechanism is unchanged; the intervention is perception-path repair before the frozen VLA
- training signal is absent in the prototype; the hypothesis is deployment-time geometric continuity under visual corruption

Required kill gates:

- full-frame hold-last matches or beats full: `SIMPLE_TEMPORAL_BASELINE_EXPLAINS_METHOD`
- no-temporal ablation matches or beats full: `TEMPORAL_COMPONENT_NOT_USEFUL`
- full fails to beat occluded frozen: `NO_OCCLUSION_ROBUSTNESS_GAIN`
- clean drop exceeds 0.02: `CLEAN_RETENTION_FAILURE`

Stage A planned result files:

- `reports/gcap_vla/stage_a_partial_result.json`
- `reports/gcap_vla/stage_a_result.json`
- `reports/gcap_vla/stage_a_result.md`
