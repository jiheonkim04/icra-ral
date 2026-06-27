"""Target heatmap over candidate object slots."""

from __future__ import annotations


class TargetHeatmapHead:
    def predict(self, hidden_tokens: list[float], candidate_objects: list[str]) -> dict:
        if not candidate_objects:
            return {"top_index": -1, "scores": []}
        token_sum = sum(hidden_tokens)
        scores = []
        for idx, name in enumerate(candidate_objects):
            lexical_bonus = sum(ord(ch) for ch in name) % 17
            scores.append(float(token_sum + lexical_bonus - idx))
        top_index = max(range(len(scores)), key=lambda idx: scores[idx])
        return {"top_index": top_index, "scores": scores}
