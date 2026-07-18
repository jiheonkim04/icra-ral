from __future__ import annotations

import torch

from tca_map.action_consistent_missing_view_distillation.adapter import (
    ActionConsistentMissingViewAdapter,
)
from tca_map.action_consistent_missing_view_distillation.preflight import (
    ActionHiddenHook,
    component_mse,
    normalized_full_loss,
    stable_seed,
    t_schedule,
)


def test_t_schedule_covers_frozen_effective_batch_centers() -> None:
    assert torch.equal(
        t_schedule(8),
        torch.tensor([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]),
    )
    assert torch.equal(t_schedule(2, effective_offset=6), torch.tensor([0.8125, 0.9375]))


def test_stable_seed_is_repeatable_and_keyed() -> None:
    assert stable_seed(1, "task", 3) == stable_seed(1, "task", 3)
    assert stable_seed(1, "task", 3) != stable_seed(1, "task", 4)


def test_hook_captures_teacher_and_applies_zero_initialized_student() -> None:
    adapter = ActionConsistentMissingViewAdapter(
        hidden_size=16,
        bottleneck_dim=4,
        wrist_token_count=3,
        wrist_token_dim=8,
    )
    hook = ActionHiddenHook()
    hidden = torch.randn(2, 5, 16)
    hook.activate_teacher()
    teacher = hook(torch.nn.Identity(), (), hidden)
    assert teacher is hidden
    assert hook.last_hidden_after is hidden

    hook.activate_student(adapter, torch.ones(2, 1), compute_reconstruction=True)
    student = hook(torch.nn.Identity(), (), hidden)
    assert torch.equal(student, hidden)
    assert hook.last_reconstruction is not None
    assert hook.forward_counts == {"teacher_capture": 1, "student_adapter": 1, "inactive": 0}


def test_component_and_normalized_loss_are_finite_and_differentiable() -> None:
    student_hidden = torch.randn(2, 4, 8, requires_grad=True)
    student_raw = torch.randn(2, 4, 20, requires_grad=True)
    reconstruction = torch.randn(2, 3, 8, requires_grad=True)
    outputs = {
        "student_hidden": student_hidden,
        "teacher_hidden": torch.zeros_like(student_hidden),
        "student_raw": student_raw,
        "teacher_raw": torch.zeros_like(student_raw),
        "reconstruction": reconstruction,
        "clean_wrist": torch.zeros_like(reconstruction),
    }
    components = component_mse(outputs)
    loss = normalized_full_loss(components, {key: 1.0 for key in components})
    assert torch.isfinite(loss)
    loss.backward()
    assert student_hidden.grad is not None
    assert student_raw.grad is not None
    assert reconstruction.grad is not None
