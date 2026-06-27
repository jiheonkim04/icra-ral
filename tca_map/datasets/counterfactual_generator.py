"""Counterfactual target-swap generation for dummy smoke tests."""

from __future__ import annotations


def make_counterfactual_pairs(samples: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for sample in samples:
        target = sample["target"]
        distractor = sample["distractor"]
        pairs.append(
            {
                "sample_id": sample["sample_id"],
                "positive_instruction": sample["instruction"],
                "positive_target": target,
                "negative_instruction": f"pick up the {distractor['name']}",
                "negative_target": distractor,
                "perturbation_type": "object_identity_swap",
                "template_id": "dummy_target_swap_v0",
                "valid_negative": True,
            }
        )
    return pairs
