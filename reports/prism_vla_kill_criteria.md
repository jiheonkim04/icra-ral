# PRISM-VLA Kill Criteria

## State 2 Decision

Decision: kill PRISM-VLA as the current main route unless a later, separately gated real SmolVLA adapter diagnostic overturns the held-out proxy result.

Reasons:

- canonicalization-only matched or beat every PRISM variant on primary held-out paraphrase proxy, PRIDE, and difficulty-weighted robustness,
- the best PRISM variant, `prism_vla_plus_canonicalization`, improved over simple augmentation but not over canonicalization-only on the primary held-out metrics,
- counterfactual sensitivity was not preserved versus canonicalization-only under the proxy gate,
- no real VLA checkpoint PRISM-vs-canonicalization metric was produced in State 2.

Allowed follow-up: one separate risk-assessed real SmolVLA paraphrase feature/adapter diagnostic may be proposed only if it directly compares canonicalization-only, PRISM, and PRISM+canonicalization without full fine-tuning, rollout, GPU, downloads, OpenVLA-OFT, or paper-grade claims.

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
