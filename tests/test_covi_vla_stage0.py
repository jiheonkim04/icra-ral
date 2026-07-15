import copy

import numpy as np
import torch

from tca_map.smolvla.covi_vla import (
    COVIStage0Adapter,
    COVIStage0Config,
    PROPOSAL_HASH,
    classify_stage0,
    covi_stage0_loss,
    episode_cluster_bootstrap_margin,
    irregular_occlusion_mask,
    objective_gradient_audit,
    parameter_gradient_norms,
    partition_stage0_manifest,
    partition_summary,
)


def _manifest() -> dict:
    splits = {"train": [], "val": [], "test": []}
    for task in range(40):
        for episode_offset in range(2):
            episode = task * 10 + episode_offset
            for frame in range(15):
                splits["train"].append(
                    {
                        "sample_id": f"train_{task}_{episode}_{frame}",
                        "split": "train",
                        "task_index": task,
                        "episode_index": episode,
                        "frame_index": frame,
                    }
                )
        for frame in range(10):
            splits["val"].append(
                {
                    "sample_id": f"val_{task}_{frame}",
                    "split": "val",
                    "task_index": task,
                    "episode_index": task * 10 + 2,
                    "frame_index": frame,
                }
            )
        for episode_offset in range(2):
            for frame in range(15):
                splits["test"].append(
                    {
                        "sample_id": f"test_{task}_{episode_offset}_{frame}",
                        "split": "test",
                        "task_index": task,
                        "episode_index": task * 10 + 3 + episode_offset,
                        "frame_index": frame,
                    }
                )
    return {"splits": splits}


def test_covi_partitions_and_occlusion_are_frozen() -> None:
    partitions = partition_stage0_manifest(_manifest())
    summary = partition_summary(partitions)

    assert PROPOSAL_HASH == "338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621"
    assert summary["discovery_fit"] == {
        "records": 600,
        "episodes": 40,
        "tasks": 40,
        "duplicate_sample_ids": 0,
        "duplicate_frame_keys": 0,
    }
    assert summary["discovery_one_check"]["records"] == 600
    assert summary["validation"]["records"] == 400
    assert summary["confirmatory_reserved"]["records"] == 1200

    first = irregular_occlusion_mask(256, 256, sample_id="sample", stream=1)
    second = irregular_occlusion_mask(256, 256, sample_id="sample", stream=1)
    other_stream = irregular_occlusion_mask(256, 256, sample_id="sample", stream=2)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other_stream)
    assert 0.14 <= float(first.mean()) <= 0.22


def test_covi_adapter_is_identity_initialized_and_receives_gradients(tmp_path) -> None:
    torch.manual_seed(7)
    cfg = COVIStage0Config(feature_dim=16, task_dim=4, hidden_dim=12, context_hidden_dim=8)
    model = COVIStage0Adapter(cfg)
    batch = 6
    source = torch.randn(batch, cfg.source_without_context_dim)
    clean_source = torch.randn(batch, cfg.source_without_context_dim)
    camera2 = torch.randn(batch, cfg.feature_dim)
    clean_camera2 = torch.randn(batch, cfg.feature_dim)
    target = torch.randn(batch, cfg.feature_dim)
    context = torch.rand(batch, cfg.context_dim)

    with torch.no_grad():
        initialized = model(source, camera2)
    assert torch.equal(initialized["adapted"], camera2)
    assert float(initialized["gate"].max()) <= 1.01e-4

    total, terms = covi_stage0_loss(
        model,
        occluded_source=source,
        occluded_camera2=camera2,
        target=target,
        context_target=context,
        clean_source=clean_source,
        clean_camera2=clean_camera2,
    )
    total.backward()
    gradients = parameter_gradient_norms(model)
    assert torch.isfinite(total)
    assert set(terms) == {"view", "clean", "delta", "gate", "action"}
    assert gradients["context_head"] > 0.0
    assert gradients["predictor"] > 0.0
    assert gradients["residual_projection"] > 0.0
    assert gradients["gate_head"] > 0.0

    audit_model = COVIStage0Adapter(cfg)
    audit = objective_gradient_audit(
        audit_model,
        occluded_source=source,
        occluded_camera2=camera2,
        target=target,
        context_target=context,
        clean_source=clean_source,
        clean_camera2=clean_camera2,
    )
    assert set(audit["weighted_gradient_norms"]) == {"view", "clean", "delta", "gate", "action"}
    assert audit["finite_by_objective"] == {
        "view": True,
        "clean": True,
        "delta": True,
        "gate": True,
        "action": True,
    }
    assert audit["weighted_gradient_norms"]["view"] > 0.0
    assert audit["weighted_gradient_norms"]["gate"] > 0.0

    checkpoint = tmp_path / "adapter.pt"
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict()}, checkpoint)
    reloaded = COVIStage0Adapter(cfg)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
    with torch.no_grad():
        expected = model(source, camera2)["adapted"]
        observed = reloaded(source, camera2)["adapted"]
    assert torch.equal(expected, observed)


def test_covi_false_negative_safeguard_requires_uncertainty_evidence() -> None:
    cfg = COVIStage0Config()
    base = {
        "implementation_and_data_valid": True,
        "diagnostic_headroom_exists": True,
        "identity_and_safety_passed": True,
        "candidate_margin": -0.011718,
        "candidate_margin_vs_vim_proxy": -0.005,
        "candidate_margin_vs_random_cutout": -0.004,
        "bootstrap_interval": {"episode_count": 40, "low": -0.08, "high": 0.05},
        "normalization_sensitivity_resolved": True,
    }
    assert classify_stage0(base, cfg) == "COVI_STAGE_0_UNDERPOWERED_ONE_CHECK_ALLOWED"

    robust = copy.deepcopy(base)
    robust["candidate_margin"] = -0.05
    robust["candidate_margin_vs_vim_proxy"] = -0.04
    robust["candidate_margin_vs_random_cutout"] = -0.03
    robust["bootstrap_interval"] = {"episode_count": 40, "low": -0.09, "high": 0.01}
    assert classify_stage0(robust, cfg) == "ROBUST_EMPIRICAL_DESIGN_FAILURE"

    passed = copy.deepcopy(base)
    passed["candidate_margin"] = 0.04
    passed["candidate_margin_vs_vim_proxy"] = 0.03
    passed["candidate_margin_vs_random_cutout"] = 0.025
    passed["bootstrap_interval"] = {"episode_count": 40, "low": 0.01, "high": 0.07}
    assert classify_stage0(passed, cfg) == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"


def test_episode_cluster_bootstrap_tracks_independent_episode_count() -> None:
    rng = np.random.default_rng(9)
    target = rng.normal(size=(80, 6))
    candidate = target + rng.normal(scale=0.05, size=target.shape)
    baseline = target + rng.normal(scale=0.20, size=target.shape)
    episode_ids = np.repeat(np.arange(40), 2)
    interval = episode_cluster_bootstrap_margin(
        candidate=candidate,
        baseline=baseline,
        target=target,
        train_target_mean=np.zeros(6),
        episode_ids=episode_ids,
        iterations=200,
        seed=10,
    )
    assert interval["episode_count"] == 40
    assert interval["record_count"] == 80
    assert interval["low"] > 0.0
