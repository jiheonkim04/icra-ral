from __future__ import annotations

import torch

from tca_map.smolvla.a2c2_local_prior import (
    A2C2LocalConfig,
    A2C2ResidualTransformer,
    FIDELITY_LABEL,
    deterministic_task_scalar,
    parameter_counts,
    phase_feature,
)


def _tiny_model() -> A2C2ResidualTransformer:
    cfg = A2C2LocalConfig(
        dim_model=32,
        n_heads=4,
        n_encoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        vlm_hidden_dim=16,
        pretrained_backbone_weights=None,
    )
    return A2C2ResidualTransformer(
        cfg,
        image_mean=torch.zeros(2, 3),
        image_std=torch.ones(2, 3),
        state_mean=torch.zeros(8),
        state_std=torch.ones(8),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )


def test_fidelity_label_and_phase_are_frozen() -> None:
    assert FIDELITY_LABEL == "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT"
    assert torch.allclose(phase_feature(0), torch.tensor([0.0, 1.0]))
    assert phase_feature(torch.tensor([0, 49])).shape == (2, 2)


def test_task_scalar_is_deterministic() -> None:
    first = deterministic_task_scalar(["task a", "task b"], device=torch.device("cpu"), dtype=torch.float32)
    second = deterministic_task_scalar(["task a", "task b"], device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(first, second)
    assert first.shape == (2, 1)
    assert first[0].item() != first[1].item()


def test_cached_feature_training_graph_has_finite_gradient() -> None:
    torch.manual_seed(7)
    model = _tiny_model()
    batch = 2
    loss, metrics = model.training_loss(
        image_features=torch.randn(batch, 2, 512, 2, 2),
        state=torch.randn(batch, 8),
        base_action=torch.randn(batch, 7),
        target_action=torch.randn(batch, 7),
        base_chunk=torch.randn(batch, 50, 7),
        time_feature=phase_feature(torch.tensor([3, 11])),
        vlm_hidden=torch.randn(batch, 16),
        tasks=["task a", "task b"],
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert torch.isfinite(loss)
    assert metrics["mse_loss"] == float(loss.detach())
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    counts = parameter_counts(model)
    assert counts["trainable"] > 0
    assert counts["frozen"] > 0
    assert all(not parameter.requires_grad for parameter in model.image_encoder.parameters())


def test_predict_action_is_additive_in_normalized_action_space() -> None:
    model = _tiny_model().eval()
    for parameter in model.parameters():
        if parameter.requires_grad:
            torch.nn.init.zeros_(parameter)
    base = torch.randn(1, 7)
    output = model.predict_action(
        images=torch.rand(1, 2, 3, 64, 64),
        state=torch.randn(1, 8),
        base_action=base,
        base_chunk=torch.randn(1, 50, 7),
        time_feature=phase_feature(5).unsqueeze(0),
        vlm_hidden=torch.randn(1, 16),
        tasks=["task"],
    )
    assert torch.allclose(output, base, atol=1e-6, rtol=0.0)
