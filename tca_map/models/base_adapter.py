"""Adapter protocol for VLA backbones."""

from __future__ import annotations

from typing import Protocol


class BaseAdapter(Protocol):
    def encode(self, observation: dict, instruction: str) -> dict:
        ...

    def get_hidden_tokens(self, observation: dict, instruction: str) -> list[float]:
        ...

    def predict_action(self, observation: dict, instruction: str) -> list[float]:
        ...

    def train_step(self, batch: dict, loss_config: dict) -> dict:
        ...

    def action_normalize(self, action: list[float]) -> list[float]:
        ...

    def action_denormalize(self, action: list[float]) -> list[float]:
        ...
