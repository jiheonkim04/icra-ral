"""Frozen LIFT-VLA flow-sampling primitives.

The module wraps public SmolVLA model methods and never edits LeRobot or the
checkpoint.  The pure sampler is deliberately independent of a concrete model
so its equations and field-evaluation counts can be unit tested.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch


Variant = Literal["base", "cag", "lift", "last_step_ablation"]
Branch = Literal["conditioned", "empty"]
FieldFn = Callable[[Branch, torch.Tensor, torch.Tensor], torch.Tensor]

BASE = "base"
CAG = "cag"
LIFT = "lift"
LAST_STEP_ABLATION = "last_step_ablation"
VARIANTS = (BASE, CAG, LIFT, LAST_STEP_ABLATION)


@dataclass(frozen=True)
class PrefixCache:
    pad_masks: torch.Tensor
    past_key_values: Any
    embedding_shape: tuple[int, ...]
    pad_mask_shape: tuple[int, ...]


@dataclass(frozen=True)
class FlowSample:
    native: torch.Tensor
    variant: str
    omega: float
    num_steps: int
    field_evaluations: dict[str, int]
    step_metrics: tuple[dict[str, float], ...]


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.float().square())).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denominator.item()) == 0.0:
        return 0.0
    return float(torch.dot(left_flat, right_flat).div(denominator).item())


def _step_record(step: int, time_value: float, conditioned: torch.Tensor, empty: torch.Tensor) -> dict[str, float]:
    return {
        "step": float(step),
        "time": float(time_value),
        "conditioned_minus_empty_rms": _rms(conditioned - empty),
        "conditioned_empty_cosine": _cosine(conditioned, empty),
    }


def sample_flow_variant(
    field_fn: FieldFn,
    noise: torch.Tensor,
    *,
    variant: Variant,
    omega: float,
    num_steps: int = 10,
) -> FlowSample:
    """Apply exactly one preregistered LIFT sampling equation."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown LIFT variant: {variant}")
    if num_steps != 10:
        raise ValueError(f"LIFT is frozen to 10 flow steps, received {num_steps}")
    if noise.ndim != 3:
        raise ValueError(f"expected native noise [B,H,D], received {tuple(noise.shape)}")

    counts: Counter[str] = Counter()
    records: list[dict[str, float]] = []
    dt = -1.0 / float(num_steps)
    batch_size = int(noise.shape[0])

    def evaluate(branch: Branch, latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        value = field_fn(branch, latent, timestep)
        if value.shape != latent.shape:
            raise ValueError(f"field shape {tuple(value.shape)} does not match latent {tuple(latent.shape)}")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"nonfinite {branch} field")
        counts[branch] += 1
        return value

    if variant == CAG:
        conditioned_latent = noise.clone()
        empty_latent = noise.clone()
        for step in range(num_steps):
            time_value = 1.0 + step * dt
            timestep = torch.full(
                (batch_size,), time_value, dtype=torch.float32, device=noise.device
            )
            conditioned = evaluate("conditioned", conditioned_latent, timestep)
            empty = evaluate("empty", empty_latent, timestep)
            records.append(_step_record(step, time_value, conditioned, empty))
            conditioned_latent = conditioned_latent + dt * conditioned
            empty_latent = empty_latent + dt * empty
        native = empty_latent + float(omega) * (conditioned_latent - empty_latent)
    else:
        latent = noise.clone()
        for step in range(num_steps):
            time_value = 1.0 + step * dt
            timestep = torch.full(
                (batch_size,), time_value, dtype=torch.float32, device=noise.device
            )
            conditioned = evaluate("conditioned", latent, timestep)
            if variant == BASE:
                update = conditioned
            else:
                empty = evaluate("empty", latent, timestep)
                records.append(_step_record(step, time_value, conditioned, empty))
                if variant == LIFT and float(omega) == 1.0:
                    # Algebraic passthrough avoids cancellation error in v_u + (v_c - v_u).
                    update = conditioned
                elif variant == LIFT or step == num_steps - 1:
                    update = empty + float(omega) * (conditioned - empty)
                else:
                    update = conditioned
            latent = latent + dt * update
        native = latent

    return FlowSample(
        native=native,
        variant=str(variant),
        omega=float(omega),
        num_steps=int(num_steps),
        field_evaluations={
            "conditioned": int(counts["conditioned"]),
            "empty": int(counts["empty"]),
            "total": int(sum(counts.values())),
        },
        step_metrics=tuple(records),
    )


def build_prefix_cache(
    model: Any,
    images: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    language_tokens: torch.Tensor,
    language_masks: torch.Tensor,
    state: torch.Tensor,
) -> PrefixCache:
    """Build the same frozen prefix KV cache as SmolVLA.sample_actions."""

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    embeddings, pad_masks, attention_masks = model.embed_prefix(
        images, image_masks, language_tokens, language_masks, state=state
    )
    attention_2d = make_att_2d_masks(pad_masks, attention_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=attention_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embeddings, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )
    return PrefixCache(
        pad_masks=pad_masks,
        past_key_values=past_key_values,
        embedding_shape=tuple(int(dim) for dim in embeddings.shape),
        pad_mask_shape=tuple(int(dim) for dim in pad_masks.shape),
    )


def sample_smolvla_variant(
    policy: Any,
    conditioned_batch: dict[str, torch.Tensor],
    empty_batch: dict[str, torch.Tensor],
    noise: torch.Tensor,
    *,
    variant: Variant,
    omega: float,
) -> tuple[FlowSample, dict[str, Any]]:
    """Run a frozen variant through a loaded SmolVLA policy."""

    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    model = policy.model
    if int(model.config.num_steps) != 10:
        raise ValueError(f"checkpoint num_steps is {model.config.num_steps}, expected 10")
    images, image_masks = policy.prepare_images(conditioned_batch)
    state = policy.prepare_state(conditioned_batch)
    empty_images, empty_image_masks = policy.prepare_images(empty_batch)
    empty_state = policy.prepare_state(empty_batch)
    if not torch.equal(state, empty_state):
        raise ValueError("conditioned and empty branches have different state tensors")
    for left, right in zip(images, empty_images, strict=True):
        if not torch.equal(left, right):
            raise ValueError("conditioned and empty branches have different image tensors")
    for left, right in zip(image_masks, empty_image_masks, strict=True):
        if not torch.equal(left, right):
            raise ValueError("conditioned and empty branches have different camera masks")

    caches: dict[Branch, PrefixCache] = {}
    caches["conditioned"] = build_prefix_cache(
        model,
        images,
        image_masks,
        conditioned_batch[OBS_LANGUAGE_TOKENS],
        conditioned_batch[OBS_LANGUAGE_ATTENTION_MASK],
        state,
    )
    if variant != BASE:
        caches["empty"] = build_prefix_cache(
            model,
            images,
            image_masks,
            empty_batch[OBS_LANGUAGE_TOKENS],
            empty_batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
        )

    def field(branch: Branch, latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        cache = caches[branch]
        return model.denoise_step(
            x_t=latent,
            prefix_pad_masks=cache.pad_masks,
            past_key_values=cache.past_key_values,
            timestep=timestep,
        )

    sample = sample_flow_variant(
        field,
        noise,
        variant=variant,
        omega=float(omega),
        num_steps=int(model.config.num_steps),
    )
    prefix_audit = {
        branch: {
            "embedding_shape": list(cache.embedding_shape),
            "pad_mask_shape": list(cache.pad_mask_shape),
        }
        for branch, cache in caches.items()
    }
    return sample, prefix_audit


def unpad_native(policy: Any, native: torch.Tensor) -> torch.Tensor:
    action_dim = int(policy.config.action_feature.shape[0])
    if bool(getattr(policy.config, "adapt_to_pi_aloha", False)):
        raise ValueError("LIFT's frozen LIBERO path forbids Pi-Aloha action adaptation")
    return native[:, :, :action_dim]


def postprocess_native(policy: Any, native: torch.Tensor, postprocessor: Any) -> torch.Tensor:
    return postprocessor(unpad_native(policy, native))
