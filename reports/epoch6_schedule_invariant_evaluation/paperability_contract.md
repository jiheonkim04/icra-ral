# Paperability Contract: Schedule-Invariant Stochastic VLA Evaluation

Status: operationally resource-blocked before problem adjudication; retained as a resumable thesis, not scientifically closed.
Empirical outcomes used: none.
Paper generation: unauthorized.

## One-sentence claim

Episode-addressed policy randomness can remove evaluation-schedule dependence
from stochastic VLA action sampling without serializing inference, making paired
robot-policy comparisons reproducible on a fixed execution stack.

## Claim-to-mechanism causal graph

`shard/batch request order -> process-global RNG draw assignment -> sampled`
`action chunk -> physical trajectory -> episode outcome/comparison`

The proposed contract, only after the problem gate, replaces the first edge
with:

`(model, task, episode, policy-call, root seed) -> dedicated random sample`

It must preserve the model's sampling distribution and must not use outcomes.

## Closest-three-prior difference table

| Prior | What it already provides | Required distinct contribution |
|---|---|---|
| vla-eval | Standard model/benchmark interface, provenance, episode sharding, batch inference, and reproducibility audit | A framework-wide stochastic-sampling identity contract and evidence that schedule changes otherwise alter paired closed-loop conclusions. |
| What Are We Actually Benchmarking? | Protocol/statistical analysis of closed-loop nondeterminism | Isolation and removal of policy-randomness reassignment caused specifically by evaluation scheduling. |
| vLLM batch invariance | Non-robotics precedent for batch-invariant stochastic generation | Robot episode/call addressing, continuous-action trace invariance, closed-loop materiality, and policy-comparison evidence. |

## Expected contributions, contingent on evidence

1. An empirical characterization of schedule-to-sample coupling in stochastic
   VLA evaluation across multiple policy families.
2. A precise episode-addressed randomness contract and compatible adapter path
   that preserves sampling semantics while making per-episode traces invariant
   to request order on a fixed stack.
3. Paired closed-loop evidence showing when the contract changes reproducibility,
   inference, and throughput tradeoffs—and when it does not.

## Primary table shell

| Policy | Benchmark/task family | Schedule | Episodes | Trace discordance | Success | Paired disagreement | 95% interval | Throughput |
|---|---|---:|---:|---:|---:|---:|---|---:|
| X-VLA | TBD frozen manifest | serial/global RNG | -- | -- | -- | -- | -- | -- |
| X-VLA | same | sharded/global RNG | -- | -- | -- | -- | -- | -- |
| X-VLA | same | sharded/addressed RNG | -- | -- | -- | -- | -- | -- |
| SmolVLA | same/compatible | matched rows | -- | -- | -- | -- | -- | -- |
| OpenVLA-OFT | deterministic control | matched rows | -- | -- | -- | -- | -- | -- |

## Key ablation table shell

| Randomness scheme | Episode addressed | Call addressed | Order-invariant trace | Marginal-law check | Overhead |
|---|---:|---:|---:|---:|---:|
| Global seed | no | no | -- | reference | -- |
| Per-shard seed | no | no | -- | -- | -- |
| Serialized inference | no | no | -- | -- | -- |
| Episode only | yes | no | -- | -- | -- |
| Full proposed contract | yes | yes | -- | -- | -- |

## Main figure shell

Left: two schedules assign process-global random draws to different episode
keys. Middle: episode-addressed keys remove this edge. Right: paired trajectory
and outcome discordance before and after the contract, with throughput.

## Simulation-only defense

The claim concerns simulator evaluation semantics, not physical-robot
performance. Evidence must span competent stochastic policies, multiple task
families, paired fixed reset identities, raw and processed actions, closed-loop
outcomes, a deterministic control, and throughput. Claims remain stack-local.

## Strongest likely reject reason and required answer

Reject reason: this is deterministic-seeding software hygiene with no robotics
consequence. Required answer: repeated schedule-induced paired trajectory and
outcome disagreement that changes a realistic policy comparison, plus a general
contract across at least two stochastic policy families and negligible enough
overhead to retain parallel evaluation value.

## Six-page allocation sketch

- 0.6 page: evaluation problem and exact claim.
- 0.7 page: closest work and why ordinary seeds are insufficient.
- 1.2 pages: stochastic identity contract and adapter semantics.
- 1.8 pages: multi-policy/task closed-loop results and ablations.
- 0.7 page: throughput, reproducibility, and generalization.
- 0.5 page: limitations/failure analysis.
- 0.5 page: references and compressed reproducibility material.

This contract is automatically archived if either frozen problem gate fails.

## Operational disposition

The outcome-suppressed problem audit did not reach its scientific sequences.
Early smokes crossed the frozen 82% physical-memory ceiling; after the host
baseline later improved, two complete forwards stayed below the ceiling but
failed frozen pagefile/teardown qualification. The bounded semantically null
repair budget is exhausted. See `operational_blocker.json`. This is not evidence
for or against schedule dependence. A fresh immutable run may resume only
after the recorded human-controlled clean-host condition is established.
