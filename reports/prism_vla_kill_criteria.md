# PRISM-VLA Kill Criteria

## Continue Criteria

Continue past State 1 only if all are true:

- paraphrasing causes measurable degradation in the base/no-paraphrase-training arm,
- PRISM improves over simple paraphrase augmentation on paraphrase proxy, PRIDE, consistency, object-lexical robustness, or action divergence,
- clean retention stays at or above 80 percent of the base clean proxy,
- counterfactual sensitivity remains within 80 percent of simple augmentation and does not collapse to same-target predictions,
- at least one real local dataset-backed proxy metric is produced.

## Kill Or Block Criteria

Kill or block scale-up if any are true:

- simple paraphrase augmentation matches PRISM on robustness metrics,
- paraphrase degradation is not measurable,
- PRISM improves consistency only by ignoring instruction differences,
- clean proxy drops substantially,
- counterfactual object/target sensitivity collapses,
- no real VLA/model/dataset metric can be produced,
- the result depends on OpenVLA-OFT or heavy local training,
- the result is only planner text with no executable diagnostic, loss, metric, or test.

## Evidence Labels

State 1 can only support:

- exploratory offline proxy,
- CPU surrogate-policy result,
- local LIBERO action-chunk proxy,
- setup readiness for a later real VLA adapter diagnostic.

State 1 cannot support:

- standard LIBERO success,
- simulator rollout success,
- SOTA or paper-grade claims,
- real robot robustness claims.
