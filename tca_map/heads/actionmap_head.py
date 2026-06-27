"""Small ActionMap-style heatmap head for smoke validation."""

from __future__ import annotations

import math


class ActionMapHead:
    def __init__(self, grid_size: int = 8):
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        self.grid_size = grid_size

    def action_to_voxel(self, action: list[float]) -> tuple[int, int, int]:
        coords = []
        for value in action[:3]:
            clipped = max(-1.0, min(1.0, float(value)))
            idx = round((clipped + 1.0) * 0.5 * (self.grid_size - 1))
            coords.append(int(idx))
        return tuple(coords)  # type: ignore[return-value]

    def predict_heatmap(self, hidden_tokens: list[float]) -> dict:
        total = sum(hidden_tokens) if hidden_tokens else 0.0
        center = int(abs(total) * 997) % self.grid_size
        voxel = (center, (center + 1) % self.grid_size, (center + 2) % self.grid_size)
        entropy = math.log(self.grid_size ** 3)
        return {"top_voxel": voxel, "entropy": entropy, "grid_size": self.grid_size}

    def expected_action(self, heatmap: dict) -> list[float]:
        scale = 2.0 / (self.grid_size - 1)
        return [round(v * scale - 1.0, 4) for v in heatmap["top_voxel"]] + [1.0]
