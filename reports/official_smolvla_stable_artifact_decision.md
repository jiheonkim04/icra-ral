# Official SmolVLA Stable Artifact Decision

Date: 2026-07-10 KST

Final decision: `NEEDS_LONGER_LORA_BASELINE_REPRO`

Reason: The artifact works, but the single rank-4 LoRA regeneration seed is now the main unresolved robustness issue.

Exact next step: Run independent standard rank-4 LoRA seeds under the fixed manifest; do not design a new method yet.
