# Official SmolVLA Rollout Pilot Plan

Date: 2026-07-10 KST

- plan status: `not_executed`
- reason: Official LIBERO rollout dependencies are missing in the active native environment: ['hf-libero', 'libero', 'robosuite']

Predeclared pilot if WSL/Linux official stack is available:

- tasks: minimum 4 fixed LIBERO task IDs across available suites.
- reset/evaluation seeds: fixed before execution.
- episodes: 5 per selected task per policy if runtime allows.
- policies: frozen_base, rank4_lora seeds 11/22/33, validation-selected action-space static mixes for each seed.
- no policy or seed selection after rollout outcomes.
