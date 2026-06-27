"""Small LIBERO-style dummy samples for interface validation."""

from __future__ import annotations


def make_dummy_samples(count: int = 4) -> list[dict]:
    samples: list[dict] = []
    objects = ["red mug", "blue bowl", "green cube", "yellow cup"]
    for idx in range(count):
        target_idx = idx % len(objects)
        distractor_idx = (target_idx + 1) % len(objects)
        action = [round(0.1 * (idx + 1), 3), round(-0.05 * idx, 3), round(0.02 * idx, 3), 1.0]
        samples.append(
            {
                "sample_id": f"dummy_{idx:03d}",
                "observation": {
                    "rgb_shape": [64, 64, 3],
                    "proprio": [0.0, 0.0, 0.0, 1.0],
                    "candidate_objects": objects,
                },
                "instruction": f"pick up the {objects[target_idx]}",
                "target": {"object_id": target_idx, "name": objects[target_idx]},
                "distractor": {"object_id": distractor_idx, "name": objects[distractor_idx]},
                "expert_action": action,
                "dataset_version": "dummy-libero-v0",
            }
        )
    return samples
