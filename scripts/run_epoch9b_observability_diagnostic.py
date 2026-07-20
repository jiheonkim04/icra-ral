#!/usr/bin/env python3
"""Development-only mass observability diagnostic for legal Epoch 9 traces."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json

DATASET = ROOT / "reports/epoch9_relational_probe_dataset/development/paired_v1/result.json"
FROZEN_LDA = ROOT / "reports/epoch8_active_property_probe_belief_stage0_repair1.json"
OUTPUT_JSON = ROOT / "reports/epoch9b_observability_diagnostic.json"
OUTPUT_MD = ROOT / "reports/epoch9b_observability_diagnostic.md"
SEQUENCE_LENGTH = 64
SEEDS = (9009, 9010)
EPOCHS = 120
AUDITED_ROI_CENTER_32 = {"front": (23, 7), "back": (18, 15)}


@dataclass
class Example:
    episode_id: str
    demo_index: int
    label_front_heavier: int
    probe_order: str
    contrast: str
    sequence: np.ndarray
    first_rgb: np.ndarray
    endpoint_error: np.ndarray
    frozen_lda_probability: float
    response_summary: np.ndarray
    position_nuisance: np.ndarray
    displacement_nuisance: np.ndarray


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resample(array: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    indices = np.rint(np.linspace(0, len(array) - 1, length)).astype(np.int64)
    return np.asarray(array[indices], dtype=np.float32)


def trace_arrays(path: Path, slot: str) -> tuple[np.ndarray, np.ndarray, list[float], np.ndarray]:
    with np.load(path) as trace:
        forbidden = {"mass", "target_contact_eval", "target_pose", "reward", "expert_action"}
        if forbidden & set(trace.files):
            raise RuntimeError(f"privileged field in deployable trace {path}: {sorted(forbidden & set(trace.files))}")
        action = np.asarray(trace["action"], dtype=np.float32)
        eef = np.asarray(trace["eef_pos"], dtype=np.float32)
        goal = np.asarray(trace["controller_goal_pos"], dtype=np.float32)
        error = np.asarray(trace["controller_error"], dtype=np.float32)
        rgb = np.asarray(trace["rgb_diff_32"], dtype=np.float32) / 255.0
    lengths = {len(action), len(eef), len(goal), len(error), len(rgb)}
    if len(lengths) != 1 or not len(action):
        raise RuntimeError(f"invalid trace lengths in {path}: {lengths}")
    eef_delta = eef - eef[:1]
    velocity = np.diff(eef, axis=0, prepend=eef[:1])
    tracking = goal - eef
    x, y = AUDITED_ROI_CENTER_32[slot]
    roi = rgb[:, y - 2 : y + 3, x - 2 : x + 3].reshape(len(rgb), -1)
    pooled = rgb.reshape(len(rgb), 4, 8, 4, 8).mean(axis=(2, 4)).reshape(len(rgb), -1)
    progress = np.linspace(0.0, 1.0, len(action), dtype=np.float32)[:, None]
    duration = np.full((len(action), 1), len(action) / 256.0, dtype=np.float32)
    sequence = np.concatenate(
        (action, eef_delta, velocity, tracking, error[:, None], roi, pooled, progress, duration), axis=1
    )
    rgb_temporal = np.abs(np.diff(rgb, axis=0)).mean(axis=(1, 2)) if len(rgb) > 1 else np.zeros(1)
    velocity_norm = np.linalg.norm(velocity, axis=1)
    midpoint = max(1, len(error) // 2)
    response = [
        float(np.mean(error)),
        float(np.std(error)),
        float(np.quantile(error, 0.9)),
        float(np.max(error)),
        float(error[-1]),
        float(np.mean(error[:midpoint])),
        float(np.mean(error[midpoint:])),
        float(np.mean(velocity_norm)),
        float(np.max(velocity_norm)),
        float(np.mean(rgb_temporal)),
        float(np.max(rgb_temporal)),
        float(np.mean(np.abs(action))),
        float(len(action) / 256.0),
    ]
    frozen_shared = np.asarray(
        [
            float(np.mean(error)),
            float(np.quantile(error, 0.9)),
            float(np.max(error)),
            float(np.mean(rgb_temporal) * 255.0),
            float(np.max(rgb_temporal) * 255.0),
        ],
        dtype=np.float64,
    )
    return resample(sequence), error, response, frozen_shared


def frozen_lda_score(features: np.ndarray, model: dict[str, Any]) -> float:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    direction = np.asarray(model["direction"], dtype=np.float64)
    full = mean.copy()
    full[:5] = features
    return float(((full - mean) / scale) @ direction)


def load_examples() -> tuple[list[Example], dict[str, Any]]:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if summary["bounded_action_fraction"] != 1.0 or summary["contact_fraction"] != 1.0:
        raise RuntimeError("diagnostic requires legal contacting development traces")
    frozen_payload = json.loads(FROZEN_LDA.read_text(encoding="utf-8"))
    frozen_model = frozen_payload["classifier"]
    examples: list[Example] = []
    for row in payload["rows"]:
        if int(row["demo_index"]) >= 40:
            raise RuntimeError("sealed identity encountered")
        probes = {probe["slot"]: probe for probe in row["probes"]}
        if set(probes) != {"front", "back"}:
            raise RuntimeError(f"incomplete pair: {row['episode_id']}")
        sequences: dict[str, np.ndarray] = {}
        errors: dict[str, np.ndarray] = {}
        response: dict[str, np.ndarray] = {}
        lda_features: dict[str, np.ndarray] = {}
        for slot in ("front", "back"):
            sequence, error, summary_features, shared = trace_arrays(ROOT / probes[slot]["trace_path"], slot)
            sequences[slot] = sequence
            errors[slot] = error
            response[slot] = np.asarray(summary_features, dtype=np.float32)
            lda_features[slot] = shared
        front_score = frozen_lda_score(lda_features["front"], frozen_model)
        back_score = frozen_lda_score(lda_features["back"], frozen_model)
        lda_probability = float(1.0 / (1.0 + math.exp(-np.clip(front_score - back_score, -50.0, 50.0))))
        frame_path = DATASET.parent / "frames" / f"{row['episode_id']}_front_initial.png"
        frame = np.asarray(Image.open(frame_path).convert("L").resize((16, 16)), dtype=np.float32) / 255.0
        initial_xy = {
            slot: np.asarray(probes[slot]["initial_target_eval_only"], dtype=np.float64)[:2]
            for slot in ("front", "back")
        }
        displacement = row["candidate_final_displacement_m_eval_only"]
        probe_order = "/".join(row["probe_order"])
        order_code = 1.0 if probe_order == "front/back" else 0.0
        position_nuisance = np.concatenate((initial_xy["front"], initial_xy["back"], [order_code])).astype(np.float32)
        displacement_nuisance = np.asarray(
            [float(displacement["front"]), float(displacement["back"]), float(displacement["front"] - displacement["back"])],
            dtype=np.float32,
        )
        paired_response = np.concatenate(
            (response["front"], response["back"], response["front"] - response["back"])
        ).astype(np.float32)
        masses = {float(row["front_mass_factor"]), float(row["back_mass_factor"])}
        examples.append(
            Example(
                episode_id=str(row["episode_id"]),
                demo_index=int(row["demo_index"]),
                label_front_heavier=int(float(row["front_mass_factor"]) > float(row["back_mass_factor"])),
                probe_order=probe_order,
                contrast="8x_vs_1x" if masses == {1.0, 8.0} else "4x_vs_2x",
                sequence=np.stack((sequences["front"], sequences["back"])).astype(np.float32),
                first_rgb=frame.reshape(-1),
                endpoint_error=np.asarray(
                    [float(errors["front"][-1]), float(errors["back"][-1]), float(errors["front"][-1] - errors["back"][-1])],
                    dtype=np.float32,
                ),
                frozen_lda_probability=lda_probability,
                response_summary=paired_response,
                position_nuisance=position_nuisance,
                displacement_nuisance=displacement_nuisance,
            )
        )
    if len(examples) != 80 or len({value.episode_id for value in examples}) != 80:
        raise RuntimeError("expected 80 unique development episodes")
    return examples, payload


def grouped_folds(examples: list[Example]) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([value.demo_index for value in examples])
    unique = sorted(int(value) for value in np.unique(groups))
    if len(unique) != 10:
        raise RuntimeError(f"expected ten reset groups, received {unique}")
    folds = []
    # Deterministic position-balanced grouping: each fold holds two complete
    # reset identities, so every trajectory and mass/order assignment from an
    # identity stays together.
    for fold_index in range(5):
        test_groups = {unique[fold_index], unique[fold_index + 5]}
        test = np.asarray([index for index, group in enumerate(groups) if int(group) in test_groups])
        train = np.asarray([index for index, group in enumerate(groups) if int(group) not in test_groups])
        folds.append((train, test))
    return folds


def classic_cv(examples: list[Example], feature: Callable[[Example], np.ndarray]) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.stack([feature(value) for value in examples])
    y = np.asarray([value.label_front_heavier for value in examples])
    probabilities = np.zeros(len(examples), dtype=np.float64)
    for train, test in grouped_folds(examples):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=9009))
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
    return probabilities


def residualized_response_cv(examples: list[Example]) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    response = np.stack([value.response_summary for value in examples])
    nuisance = np.stack(
        [np.concatenate((value.position_nuisance, value.displacement_nuisance)) for value in examples]
    )
    y = np.asarray([value.label_front_heavier for value in examples])
    probabilities = np.zeros(len(examples), dtype=np.float64)
    for train, test in grouped_folds(examples):
        nuisance_map = make_pipeline(PolynomialFeatures(degree=2, include_bias=True), StandardScaler(), Ridge(alpha=1.0))
        nuisance_map.fit(nuisance[train], response[train])
        train_residual = response[train] - nuisance_map.predict(nuisance[train])
        test_residual = response[test] - nuisance_map.predict(nuisance[test])
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=2000, random_state=9009))
        classifier.fit(train_residual, y[train])
        probabilities[test] = classifier.predict_proba(test_residual)[:, 1]
    return probabilities


def shuffled_sequences(examples: list[Example]) -> np.ndarray:
    paired = np.stack([value.sequence for value in examples]).copy()
    for example_index, example in enumerate(examples):
        for slot_index, slot in enumerate(("front", "back")):
            seed = int.from_bytes(hashlib.sha256(f"{example.episode_id}:{slot}".encode()).digest()[:4], "little")
            paired[example_index, slot_index] = paired[
                example_index, slot_index, np.random.default_rng(seed).permutation(SEQUENCE_LENGTH)
            ]
    return paired


def temporal_cv(examples: list[Example], *, shuffled: bool) -> np.ndarray:
    import torch
    from torch import nn
    from torch.nn import functional as functional

    torch.set_num_threads(1)
    x = shuffled_sequences(examples) if shuffled else np.stack([value.sequence for value in examples])
    y = np.asarray([value.label_front_heavier for value in examples], dtype=np.float32)
    probabilities = np.zeros(len(examples), dtype=np.float64)

    class Comparator(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.projection = nn.Sequential(nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.GELU())
            self.encoder = nn.GRU(32, 16, batch_first=True, bidirectional=True)
            self.attention = nn.Linear(32, 1)
            self.relational = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))

        def encode(self, sequence: Any) -> Any:
            hidden, _ = self.encoder(self.projection(sequence))
            weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=1)
            return torch.sum(hidden * weights.unsqueeze(-1), dim=1)

        def forward(self, paired: Any) -> Any:
            return self.relational(self.encode(paired[:, 0]) - self.encode(paired[:, 1])).squeeze(-1)

    for train, test in grouped_folds(examples):
        ensemble = np.zeros(len(test), dtype=np.float64)
        mean = x[train].mean(axis=(0, 1, 2), keepdims=True)
        scale = x[train].std(axis=(0, 1, 2), keepdims=True)
        scale = np.where(scale < 1e-5, 1.0, scale)
        train_x = torch.from_numpy(((x[train] - mean) / scale).astype(np.float32))
        test_x = torch.from_numpy(((x[test] - mean) / scale).astype(np.float32))
        train_y = torch.from_numpy(y[train])
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = Comparator(x.shape[-1])
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
            generator = torch.Generator().manual_seed(seed)
            for _ in range(EPOCHS):
                permutation = torch.randperm(len(train), generator=generator)
                for start in range(0, len(permutation), 16):
                    indices = permutation[start : start + 16]
                    optimizer.zero_grad(set_to_none=True)
                    loss = functional.binary_cross_entropy_with_logits(model(train_x[indices]), train_y[indices])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                ensemble += torch.sigmoid(model(test_x)).numpy() / len(SEEDS)
        probabilities[test] = ensemble
    return probabilities


def cluster_bootstrap_ci(correct: np.ndarray, groups: np.ndarray, draws: int = 10000) -> list[float]:
    unique = np.unique(groups)
    rng = np.random.default_rng(9009)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.where(groups == group)[0] for group in sampled])
        values.append(float(np.mean(correct[indices])))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def metric(examples: list[Example], probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([value.label_front_heavier for value in examples])
    predictions = (probabilities >= 0.5).astype(np.int64)
    correct = predictions == labels
    groups = np.asarray([value.demo_index for value in examples])
    result: dict[str, Any] = {
        "episodes": len(examples),
        "successes": int(np.sum(correct)),
        "accuracy": float(np.mean(correct)),
        "reset_group_bootstrap_ci95": cluster_bootstrap_ci(correct, groups),
        "by_heavy_position": {},
        "by_probe_order": {},
        "by_mass_contrast": {},
    }
    for label, container, values in (
        ("by_heavy_position", result["by_heavy_position"], ("front", "back")),
        ("by_probe_order", result["by_probe_order"], ("front/back", "back/front")),
        ("by_mass_contrast", result["by_mass_contrast"], ("8x_vs_1x", "4x_vs_2x")),
    ):
        for value in values:
            if label == "by_heavy_position":
                mask = labels == (1 if value == "front" else 0)
            elif label == "by_probe_order":
                mask = np.asarray([example.probe_order == value for example in examples])
            else:
                mask = np.asarray([example.contrast == value for example in examples])
            container[value] = {"episodes": int(np.sum(mask)), "accuracy": float(np.mean(correct[mask]))}
    return result


def split_audit(examples: list[Example]) -> dict[str, Any]:
    folds = []
    for fold_index, (train, test) in enumerate(grouped_folds(examples)):
        train_groups = sorted({examples[index].demo_index for index in train})
        test_groups = sorted({examples[index].demo_index for index in test})
        test_labels = [examples[index].label_front_heavier for index in test]
        test_orders = [examples[index].probe_order for index in test]
        folds.append(
            {
                "fold": fold_index,
                "train_demo_indices": train_groups,
                "test_demo_indices": test_groups,
                "group_overlap": sorted(set(train_groups) & set(test_groups)),
                "test_front_heavy": int(sum(test_labels)),
                "test_back_heavy": int(len(test_labels) - sum(test_labels)),
                "test_front_first": int(sum(value == "front/back" for value in test_orders)),
                "test_back_first": int(sum(value == "back/front" for value in test_orders)),
            }
        )
    return {
        "grouping": "five-fold GroupKFold by demo/reset identity",
        "same_trajectory_or_identity_crosses_fold": any(value["group_overlap"] for value in folds),
        "every_test_fold_position_and_order_balanced": all(
            value["test_front_heavy"] == value["test_back_heavy"]
            and value["test_front_first"] == value["test_back_first"]
            for value in folds
        ),
        "folds": folds,
    }


def write_markdown(result: dict[str, Any]) -> None:
    models = result["models"]
    lines = [
        "# Epoch 9B Development-Only Observability Diagnostic",
        "",
        f"Date: {result['timestamp']}",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "This is a controller-investment diagnostic only. It uses Epoch 9 development demos 30..39, "
        "groups all trajectories from one reset identity into one fold, and never accesses validation "
        "40..44 or confirmation 45..49.",
        "",
        "| control | accuracy | reset-group bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    order = (
        "balanced_no_probe_chance",
        "first_frame_rgb",
        "endpoint_only_controller_error",
        "frozen_epoch8_aggregate_lda_shared_channels",
        "raw_sequence_temporal_encoder",
        "shuffled_temporal_order",
        "final_displacement_only_eval_control",
        "candidate_position_and_order_only_eval_control",
        "position_displacement_order_eval_control",
        "response_summary_residualized_against_position_displacement_order",
    )
    for name in order:
        value = models[name]
        interval = value["reset_group_bootstrap_ci95"]
        lines.append(f"| {name} | {value['accuracy']:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
            "The frozen Epoch 8 LDA is not refit. These traces lack its wrist-video channels, so those two "
            "features are fixed to the original training mean and the five available controller/agent-view "
            "features use the preserved normalization and direction. This is a transparent degraded frozen "
            "control, not an official re-evaluation of the Epoch 8 probe.",
            "",
            "The conditional control fits a nuisance-to-response map using candidate positions, final simulator "
            "displacements, and probe order on each training fold, then classifies only the held-out residual "
            "response. Simulator displacement and position are evaluation-only diagnostics and are absent from "
            "the raw temporal encoder's inputs.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite observability diagnostic")
    examples, dataset = load_examples()
    split = split_audit(examples)
    if split["same_trajectory_or_identity_crosses_fold"] or not split["every_test_fold_position_and_order_balanced"]:
        raise RuntimeError(f"invalid grouped split: {split}")
    models = {
        "balanced_no_probe_chance": metric(examples, np.full(len(examples), 0.5)),
        "first_frame_rgb": metric(examples, classic_cv(examples, lambda value: value.first_rgb)),
        "endpoint_only_controller_error": metric(
            examples, classic_cv(examples, lambda value: value.endpoint_error)
        ),
        "frozen_epoch8_aggregate_lda_shared_channels": metric(
            examples, np.asarray([value.frozen_lda_probability for value in examples])
        ),
        "raw_sequence_temporal_encoder": metric(examples, temporal_cv(examples, shuffled=False)),
        "shuffled_temporal_order": metric(examples, temporal_cv(examples, shuffled=True)),
        "final_displacement_only_eval_control": metric(
            examples, classic_cv(examples, lambda value: value.displacement_nuisance)
        ),
        "candidate_position_and_order_only_eval_control": metric(
            examples, classic_cv(examples, lambda value: value.position_nuisance)
        ),
        "position_displacement_order_eval_control": metric(
            examples,
            classic_cv(
                examples, lambda value: np.concatenate((value.position_nuisance, value.displacement_nuisance))
            ),
        ),
        "response_summary_residualized_against_position_displacement_order": metric(
            examples, residualized_response_cv(examples)
        ),
    }
    temporal = models["raw_sequence_temporal_encoder"]["accuracy"]
    shuffled = models["shuffled_temporal_order"]["accuracy"]
    residual = models["response_summary_residualized_against_position_displacement_order"]["accuracy"]
    displacement = models["final_displacement_only_eval_control"]["accuracy"]
    beyond = bool(temporal >= 0.65 and temporal >= shuffled + 0.05 and residual >= 0.60)
    if beyond:
        decision = "TEMPORAL_MASS_SIGNAL_SURVIVES_POSITION_AND_DISPLACEMENT_CONTROL"
        interpretation = (
            "A development-only property signal remains after grouped position/order control and explicit "
            "residualization against final displacement. This supports continued controller investment but is "
            "not validation or a paper claim."
        )
    elif displacement >= 0.65:
        decision = "MASS_SIGNAL_PRIMARILY_FINAL_DISPLACEMENT_UNDER_FIXED_PROBE"
        interpretation = (
            "The reliable development signal is primarily final displacement. After controlling candidate "
            "position, probe order, and final displacement, the remaining response does not meet the diagnostic "
            "support rule. Epoch 9B may still exploit label-blind displacement ranking, but must describe it as "
            "interaction-based mass discrimination rather than general latent physical reasoning."
        )
    else:
        decision = "NO_RELIABLE_MASS_SIGNAL_IN_EXISTING_FIXED_PROBE_TRAJECTORIES"
        interpretation = (
            "Existing legal fixed-probe trajectories do not provide a reliable grouped mass signal. A new "
            "dynamic probe would need to establish observability before model investment."
        )
    result = {
        "schema_version": "epoch9b.observability_diagnostic.v1",
        "timestamp": timestamp(),
        "evidence_class": "DEVELOPMENT_ONLY_DIAGNOSTIC_NOT_FOR_CLAIMS",
        "decision": decision,
        "interpretation": interpretation,
        "dataset": {
            "path": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(DATASET),
            "demo_indices": sorted({value.demo_index for value in examples}),
            "episodes": len(examples),
            "dataset_execution_gate_pass": bool(dataset["summary"]["execution_gate_pass"]),
            "diagnostic_legal_trace_basis": {
                "bounded_action_fraction": dataset["summary"]["bounded_action_fraction"],
                "contact_fraction": dataset["summary"]["contact_fraction"],
                "note": "task-preservation failure prevents final-model use but does not erase trajectory observability",
            },
        },
        "sealed_access": {"validation_40_44": False, "confirmation_45_49": False},
        "split_audit": split,
        "temporal_model": {
            "sequence_length": SEQUENCE_LENGTH,
            "input_dimension": int(examples[0].sequence.shape[-1]),
            "architecture": "shared Linear(.,32)+bidirectional GRU(16x2)+attention; relational delta MLP",
            "epochs": EPOCHS,
            "seeds": list(SEEDS),
            "online_inputs_only": ["RGB response", "EEF proprioception", "controller goal/error", "actions", "time"],
        },
        "frozen_lda_control": {
            "source": str(FROZEN_LDA.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256(FROZEN_LDA),
            "refit": False,
            "available_features": 5,
            "missing_wrist_features": 2,
            "missing_feature_rule": "original training mean (zero normalized contribution)",
        },
        "conditional_signal_rule": {
            "temporal_accuracy_min": 0.65,
            "temporal_minus_shuffled_min": 0.05,
            "residualized_accuracy_min": 0.60,
            "pass": beyond,
        },
        "models": models,
    }
    atomic_write_json(OUTPUT_JSON, result)
    write_markdown(result)
    print(json.dumps({"decision": decision, "temporal": temporal, "shuffled": shuffled, "displacement": displacement, "residualized": residual}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
