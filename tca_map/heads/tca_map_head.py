"""Target-conditioned ActionMap smoke head."""

from __future__ import annotations

from .actionmap_head import ActionMapHead
from .target_heatmap_head import TargetHeatmapHead


class TCAMapHead:
    def __init__(self, grid_size: int = 8):
        self.target_head = TargetHeatmapHead()
        self.action_head = ActionMapHead(grid_size=grid_size)

    def predict(self, hidden_tokens: list[float], candidate_objects: list[str]) -> dict:
        target = self.target_head.predict(hidden_tokens, candidate_objects)
        conditioned_tokens = hidden_tokens + [float(target["top_index"])]
        action_heatmap = self.action_head.predict_heatmap(conditioned_tokens)
        action = self.action_head.expected_action(action_heatmap)
        return {"target": target, "action_heatmap": action_heatmap, "action": action}
