"""Dependency-free dummy adapter used by preflight and smoke tests."""

from __future__ import annotations

import hashlib
import math


class DummyAdapter:
    name = "dummy"

    def encode(self, observation: dict, instruction: str) -> dict:
        tokens = self.get_hidden_tokens(observation, instruction)
        return {"tokens": tokens, "instruction": instruction}

    def get_hidden_tokens(self, observation: dict, instruction: str) -> list[float]:
        digest = hashlib.sha256(instruction.encode("utf-8")).digest()
        base = [b / 255.0 for b in digest[:8]]
        proprio = observation.get("proprio", [0.0, 0.0, 0.0, 1.0])
        return base + [float(x) for x in proprio]

    def predict_action(self, observation: dict, instruction: str) -> list[float]:
        tokens = self.get_hidden_tokens(observation, instruction)
        return [round(math.tanh(sum(tokens[i::4]) / 4.0), 4) for i in range(4)]

    def train_step(self, batch: dict, loss_config: dict) -> dict:
        pred = self.predict_action(batch["observation"], batch["instruction"])
        target = batch["expert_action"]
        mse = sum((p - t) ** 2 for p, t in zip(pred, target)) / len(target)
        return {"loss": mse, "action_mse": mse}

    def action_normalize(self, action: list[float]) -> list[float]:
        return [float(x) for x in action]

    def action_denormalize(self, action: list[float]) -> list[float]:
        return [float(x) for x in action]
