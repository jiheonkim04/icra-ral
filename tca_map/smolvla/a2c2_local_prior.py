"""Mechanism-faithful local A2C2 prior for the asynchronous-delay pivot.

The implementation follows k1000dai/a2c2-libero at commit
54dd088302a0ef3f50c4add3ec927ab94d76a406.  It is deliberately labelled a
local port: the released repository is based on LeRobot 0.2.0, whereas the
locally runnable frozen SmolVLA checkpoint uses LeRobot 0.4.4.

This module is a prior module, not an Ours VLA and not VLA fine-tuning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


FIDELITY_LABEL = "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT"
OFFICIAL_COMMIT = "54dd088302a0ef3f50c4add3ec927ab94d76a406"


@dataclass(frozen=True)
class A2C2LocalConfig:
    """Frozen architecture settings matching the released LIBERO module."""

    action_dim: int = 7
    state_dim: int = 8
    chunk_size: int = 50
    image_views: int = 2
    image_feature_channels: int = 512
    dim_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    vlm_hidden_dim: int = 576
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    freeze_vision_backbone: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def phase_feature(offset: int | Tensor, chunk_size: int = 50) -> Tensor:
    """Return the released sin/cos chunk-position feature."""

    if isinstance(offset, Tensor):
        value = offset.to(dtype=torch.float32)
    else:
        value = torch.tensor(offset, dtype=torch.float32)
    denom = max(int(chunk_size) - 1, 1)
    phase = 2.0 * math.pi * value / float(denom)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def deterministic_task_scalar(tasks: str | Sequence[str], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Reproduce the official release's SHA1-derived scalar task token."""

    if isinstance(tasks, str):
        tasks = [tasks]
    values = []
    for task in tasks:
        digest = hashlib.sha1(str(task).encode("utf-8")).digest()  # noqa: S324 - fidelity, not security
        values.append(int.from_bytes(digest[:4], "little") / float(0xFFFFFFFF))
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(1)


def sinusoidal_position_embedding_2d(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return the released two-dimensional sinusoidal image encoding."""

    if dim % 2 != 0:
        raise ValueError(f"embedding dimension must be even, got {dim}")

    def encode(length: int, channels: int) -> Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / channels)
        )
        result = torch.zeros(length, channels, device=device, dtype=dtype)
        result[:, 0::2] = torch.sin(position * div_term)
        result[:, 1::2] = torch.cos(position * div_term)
        return result

    half = dim // 2
    h_encoding = encode(height, half)
    w_encoding = encode(width, half)
    result = torch.zeros(height, width, dim, device=device, dtype=dtype)
    result[:, :, :half] = h_encoding[:, None, :]
    result[:, :, half:] = w_encoding[None, :, :]
    return result.reshape(height * width, dim)


class A2C2ResidualTransformer(nn.Module):
    """Released A2C2 LIBERO correction-head graph with cached-feature support."""

    def __init__(
        self,
        config: A2C2LocalConfig,
        *,
        image_mean: Tensor,
        image_std: Tensor,
        state_mean: Tensor,
        state_std: Tensor,
        action_mean: Tensor,
        action_std: Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("image_mean", image_mean.to(torch.float32).reshape(config.image_views, 3, 1, 1))
        self.register_buffer("image_std", image_std.to(torch.float32).reshape(config.image_views, 3, 1, 1))
        self.register_buffer("state_mean", state_mean.to(torch.float32).reshape(config.state_dim))
        self.register_buffer("state_std", state_std.to(torch.float32).reshape(config.state_dim))
        self.register_buffer("action_mean", action_mean.to(torch.float32).reshape(config.action_dim))
        self.register_buffer("action_std", action_std.to(torch.float32).reshape(config.action_dim))

        self.image_encoder = self._make_image_encoder(config)
        self.image_proj = nn.Conv2d(config.image_feature_channels, config.dim_model, kernel_size=1)
        self.state_proj = nn.Linear(config.state_dim, config.dim_model)
        self.action_proj = nn.Linear(config.action_dim, config.dim_model)
        self.time_proj = nn.Linear(2, config.dim_model)
        self.vlm_hidden_proj = nn.Linear(config.vlm_hidden_dim, config.dim_model)
        self.task_proj = nn.Linear(1, config.dim_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim_model))

        layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_encoder_layers)
        self.out_norm = nn.LayerNorm(config.dim_model)
        residual_input_dim = config.dim_model + config.action_dim + 2 + config.state_dim
        mlp_hidden = config.dim_model * 2
        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden, config.action_dim),
        )
        self._spatial_pos_cache: dict[tuple[int, int, str, torch.dtype], Tensor] = {}

    @staticmethod
    def _make_image_encoder(config: A2C2LocalConfig) -> nn.Module:
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.ops.misc import FrozenBatchNorm2d

        weights = config.pretrained_backbone_weights
        backbone = getattr(torchvision.models, config.vision_backbone)(
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
            replace_stride_with_dilation=[False, False, False],
        )
        if int(backbone.fc.in_features) != int(config.image_feature_channels):
            raise ValueError(
                f"unexpected {config.vision_backbone} feature width: {backbone.fc.in_features}"
            )
        encoder = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        if config.freeze_vision_backbone:
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        return encoder

    def train(self, mode: bool = True) -> "A2C2ResidualTransformer":
        super().train(mode)
        if self.config.freeze_vision_backbone:
            self.image_encoder.eval()
        return self

    def normalize_images(self, images: Tensor) -> Tensor:
        return (images.to(torch.float32) - self.image_mean.unsqueeze(0)) / (
            self.image_std.unsqueeze(0) + 1e-8
        )

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state.to(torch.float32) - self.state_mean) / (self.state_std + 1e-8)

    def normalize_action(self, action: Tensor) -> Tensor:
        return (action.to(torch.float32) - self.action_mean) / (self.action_std + 1e-8)

    def unnormalize_action(self, action: Tensor) -> Tensor:
        return action.to(torch.float32) * (self.action_std + 1e-8) + self.action_mean

    @torch.no_grad()
    def encode_images(self, images: Tensor) -> Tensor:
        """Encode `(B,V,3,H,W)` latest observations with the frozen backbone."""

        if images.ndim != 5 or images.shape[1] != self.config.image_views:
            raise ValueError(f"expected (B,{self.config.image_views},3,H,W), got {tuple(images.shape)}")
        normalized = self.normalize_images(images)
        features = []
        for view in range(self.config.image_views):
            output = self.image_encoder(normalized[:, view])["feature_map"]
            features.append(output.to(torch.float32))
        return torch.stack(features, dim=1)

    def _position_encoding(self, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.config.dim_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.config.dim_model)
        )
        result = torch.zeros(length, self.config.dim_model, device=device, dtype=dtype)
        result[:, 0::2] = torch.sin(position * div_term)
        result[:, 1::2] = torch.cos(position * div_term)
        return result.unsqueeze(0)

    def _spatial_position(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (height, width, str(device), dtype)
        cached = self._spatial_pos_cache.get(key)
        if cached is None:
            cached = sinusoidal_position_embedding_2d(
                height,
                width,
                self.config.dim_model,
                device=device,
                dtype=dtype,
            )
            self._spatial_pos_cache[key] = cached
        return cached

    def forward_normalized(
        self,
        *,
        image_features: Tensor,
        state_normalized: Tensor,
        base_action_normalized: Tensor,
        base_chunk_normalized: Tensor,
        time_feature: Tensor,
        vlm_hidden: Tensor,
        tasks: str | Sequence[str],
    ) -> Tensor:
        """Predict a normalized additive residual from released-model inputs."""

        batch_size = base_action_normalized.shape[0]
        device = base_action_normalized.device
        dtype = base_action_normalized.dtype
        if image_features.ndim != 5 or image_features.shape[1] != self.config.image_views:
            raise ValueError(f"unexpected image feature shape {tuple(image_features.shape)}")

        tokens: list[Tensor] = [self.cls_token.to(device=device, dtype=dtype).expand(batch_size, -1, -1)]
        tokens.append(self.time_proj(time_feature.to(device=device, dtype=dtype)).unsqueeze(1))
        tokens.append(self.state_proj(state_normalized.to(device=device, dtype=dtype)).unsqueeze(1))
        tokens.append(self.vlm_hidden_proj(vlm_hidden.to(device=device, dtype=dtype)).unsqueeze(1))
        task_scalar = deterministic_task_scalar(tasks, device=device, dtype=dtype)
        if task_scalar.shape[0] == 1 and batch_size > 1:
            task_scalar = task_scalar.expand(batch_size, -1)
        tokens.append(self.task_proj(task_scalar).unsqueeze(1))

        chunk_tokens = self.action_proj(base_chunk_normalized.to(device=device, dtype=dtype))
        main_tokens = torch.cat(tokens, dim=1)
        total_length = main_tokens.shape[1] + chunk_tokens.shape[1]
        position = self._position_encoding(total_length, device, dtype)
        base_tokens = torch.cat(
            (
                main_tokens + position[:, : main_tokens.shape[1]],
                chunk_tokens + position[:, main_tokens.shape[1] :],
            ),
            dim=1,
        )

        image_tokens = []
        for view in range(self.config.image_views):
            feature_map = image_features[:, view].to(device=device, dtype=torch.float32)
            projected = self.image_proj(feature_map).to(dtype=dtype)
            bsz, _, height, width = projected.shape
            spatial = self._spatial_position(height, width, device, dtype).unsqueeze(0).expand(bsz, -1, -1)
            image_tokens.append(projected.flatten(2).transpose(1, 2) + spatial)
        encoded = self.encoder(torch.cat((base_tokens, *image_tokens), dim=1))
        cls_state = self.out_norm(encoded[:, 0])
        head_input = torch.cat(
            (
                cls_state,
                base_action_normalized,
                time_feature.to(device=device, dtype=dtype),
                state_normalized.to(device=device, dtype=dtype),
            ),
            dim=-1,
        )
        return self.residual_head(head_input)

    def training_loss(
        self,
        *,
        image_features: Tensor,
        state: Tensor,
        base_action: Tensor,
        target_action: Tensor,
        base_chunk: Tensor,
        time_feature: Tensor,
        vlm_hidden: Tensor,
        tasks: str | Sequence[str],
    ) -> tuple[Tensor, dict[str, float]]:
        state_norm = self.normalize_state(state)
        base_norm = self.normalize_action(base_action)
        target_norm = self.normalize_action(target_action)
        chunk_norm = self.normalize_action(base_chunk)
        predicted = self.forward_normalized(
            image_features=image_features,
            state_normalized=state_norm,
            base_action_normalized=base_norm,
            base_chunk_normalized=chunk_norm,
            time_feature=time_feature,
            vlm_hidden=vlm_hidden,
            tasks=tasks,
        )
        target_residual = target_norm - base_norm
        loss = F.mse_loss(predicted, target_residual, reduction="mean")
        return loss, {"mse_loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def predict_action(
        self,
        *,
        images: Tensor,
        state: Tensor,
        base_action: Tensor,
        base_chunk: Tensor,
        time_feature: Tensor,
        vlm_hidden: Tensor,
        tasks: str | Sequence[str],
    ) -> Tensor:
        features = self.encode_images(images)
        base_norm = self.normalize_action(base_action)
        residual = self.forward_normalized(
            image_features=features,
            state_normalized=self.normalize_state(state),
            base_action_normalized=base_norm,
            base_chunk_normalized=self.normalize_action(base_chunk),
            time_feature=time_feature,
            vlm_hidden=vlm_hidden,
            tasks=tasks,
        )
        return self.unnormalize_action(base_norm + residual)


class SmolVLAHiddenCapture:
    """Expose the same prefix token used by the released A2C2 SmolVLA fork."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.value: Tensor | None = None
        module = policy.model.vlm_with_expert
        self._handle = module.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, _module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        if not bool(kwargs.get("fill_kv_cache")):
            return
        prefix_outputs = output[0]
        hidden = None
        if isinstance(prefix_outputs, (list, tuple)):
            hidden = next((item for item in prefix_outputs if item is not None), None)
        elif prefix_outputs is not None:
            hidden = prefix_outputs
        if hidden is not None:
            self.value = hidden[:, 0].contiguous().detach().to(dtype=torch.float32)

    def pop(self) -> Tensor:
        if self.value is None:
            raise RuntimeError("SmolVLA prefix hidden state was not captured")
        result = self.value
        self.value = None
        return result

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "SmolVLAHiddenCapture":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "frozen": int(sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)),
    }


def tensor_sha256(tensors: Iterable[Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()
