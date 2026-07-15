# FAMR-VLA Stage 0A Result

Date: 2026-07-15 KST

Decision: `FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED`

## Audit Summary

- target/pretraining exact intersection: `0`
- target demonstrations: `150`
- source terminal successes/failures: `150 / 0`
- discovery/validation/test model decodes: `{'train': 24, 'validation': 0, 'test': 0}`
- duplicate episode/frame identity hashes: `0 / 0`
- adapter identity postprocessed max error: `0.0`
- micro-fit optimizer steps: `20 / 20`
- fixed-subset loss before/after: `0.7321685557253659 / 0.6487047945459684`
- fixed-subset relative reduction: `0.11399528227036353`
- checkpoint reload max error: `0.0`
- scaling identity passed: `True`
- peak CUDA allocation GiB: `1.0808053016662598`
- confirmatory observations/actions: `0 / 0`
- exception count: `0`

## Adjudication

All frozen provenance, semantic, identity, gradient, fit, checkpoint, grouping, scaling, and memory gates passed. The fixed 300-step endpoint stage is authorized.

Next command: `/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_famr_vla_stage0.py --mode train-endpoint --checkpoint /mnt/c/assets/checkpoints/smolvla_libero --libero-data-root /mnt/c/assets/data/libero --stable-artifact /mnt/c/Users/jiheo/tca_map/reports/official_smolvla_stable_prediction_artifact.json --run-root /mnt/c/Users/jiheo/tca_map/runs/famr_vla/stage0a --report-root /mnt/c/Users/jiheo/tca_map/reports/famr_vla`
