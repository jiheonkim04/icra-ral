# OCFN-VLA Researcher Rebuttal

Date: 2026-07-12 KST

Reviewer B's attack is accepted.

OCFN will not claim generic VLA guidance, action correction, verification, or online candidate selection. The only Stage A claim is that closed-loop train outcomes can identify a task-conditioned initial flow-noise prior for a frozen SmolVLA flow sampler.

Protocol changes accepted before implementation:

- keep `global_success_noise_prior` as the simple killer baseline;
- keep `task_shuffled_noise_prior` as the key ablation;
- log exact selected noise identities for every task and variant;
- treat full/global equality across all tasks as trivial equivalence unless the held-out action traces still differ by a predeclared implementation reason;
- disclose second-backbone risk instead of deferring it.

No threshold, task, identity, noise-bank size, or tie-break rule will be changed after inspecting Stage A.
