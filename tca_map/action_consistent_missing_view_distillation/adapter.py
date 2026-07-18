"""Frozen lightweight module for action-consistent missing-view distillation.

The reconstruction decoder shares a trunk with the action adapter, but its
prediction is never accepted as an input to the action residual.  This makes
cross-view reconstruction training-only auxiliary supervision rather than a
reconstructed-token insertion mechanism.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class ActionConsistentMissingViewAdapter(nn.Module):
    """Zero-initialized residual on X-VLA action hidden states.

    The attachment tensor is the output of ``model.transformer.norm`` with
    shape ``[batch, action_horizon, hidden_size]``.  A homogeneous clean batch
    takes an early return and therefore preserves the original tensor object.
    """

    RECONSTRUCTION_PREFIXES = ("wrist_token_position", "reconstruction_core", "reconstruction_output")

    def __init__(
        self,
        *,
        hidden_size: int = 1024,
        bottleneck_dim: int = 128,
        wrist_token_count: int = 50,
        wrist_token_dim: int = 1024,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.wrist_token_count = int(wrist_token_count)
        self.wrist_token_dim = int(wrist_token_dim)
        self.residual_scale = float(residual_scale)

        self.shared_down = nn.Linear(self.hidden_size, self.bottleneck_dim)
        self.shared_core = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.action_residual_output = nn.Linear(self.bottleneck_dim, self.hidden_size)

        self.wrist_token_position = nn.Parameter(
            torch.zeros(self.wrist_token_count, self.bottleneck_dim)
        )
        self.reconstruction_core = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.reconstruction_output = nn.Linear(self.bottleneck_dim, self.wrist_token_dim)

        nn.init.normal_(self.wrist_token_position, mean=0.0, std=0.02)
        nn.init.zeros_(self.action_residual_output.weight)
        nn.init.zeros_(self.action_residual_output.bias)
        nn.init.zeros_(self.reconstruction_output.weight)
        nn.init.zeros_(self.reconstruction_output.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        missing_indicator: torch.Tensor,
        *,
        compute_reconstruction: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden must be [B,T,{self.hidden_size}], got {tuple(hidden.shape)}"
            )
        if missing_indicator.ndim != 2 or tuple(missing_indicator.shape) != (hidden.shape[0], 1):
            raise ValueError("missing_indicator must be [B,1]")
        if not bool(torch.isfinite(missing_indicator).all().item()):
            raise ValueError("missing_indicator must be finite")
        if bool(((missing_indicator < 0) | (missing_indicator > 1)).any().item()):
            raise ValueError("missing_indicator must be in [0,1]")

        # Deployment clean-view execution disables the hook.  This explicit
        # guard additionally guarantees exact object/value identity if called.
        if not bool((missing_indicator != 0).any().item()):
            zeros = hidden.new_zeros((hidden.shape[0], 1))
            return hidden, None, {"residual_l2": zeros, "missing_fraction": zeros}

        shared = F.gelu(self.shared_down(hidden))
        shared = F.gelu(self.shared_core(shared))
        residual = self.residual_scale * torch.tanh(self.action_residual_output(shared))
        mask = missing_indicator.to(dtype=torch.bool).unsqueeze(-1)
        adapted = torch.where(mask, hidden + residual, hidden)

        reconstruction = None
        if compute_reconstruction:
            pooled = shared.mean(dim=1)
            reconstruction_hidden = pooled.unsqueeze(1) + self.wrist_token_position.unsqueeze(0)
            reconstruction_hidden = F.gelu(self.reconstruction_core(reconstruction_hidden))
            reconstruction = self.reconstruction_output(reconstruction_hidden)

        residual_l2 = torch.linalg.vector_norm(residual.detach().float(), dim=(-2, -1), keepdim=False)
        return adapted, reconstruction, {
            "residual_l2": residual_l2.unsqueeze(-1),
            "missing_fraction": missing_indicator.detach().float().mean(dim=1, keepdim=True),
        }

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only parameters that execute in the legal deployment graph."""

        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith(self.RECONSTRUCTION_PREFIXES)
        }


def adapter_parameter_count(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Count module parameters without inspecting the frozen X-VLA backbone."""

    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if (parameter.requires_grad or not trainable_only)
        )
    )


def state_dict_parameter_count(state: Mapping[str, torch.Tensor]) -> int:
    return int(sum(value.numel() for value in state.values()))
