#!/usr/bin/env python3
"""Train, freeze, and evaluate the Epoch 9 temporal relational belief model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json

DEVELOPMENT_RESULT = (
    ROOT / "reports/epoch9_relational_probe_dataset/development/rotation2_full_v1/result.json"
)
DEVELOPMENT_REPORT = ROOT / "reports/epoch9_temporal_reference_development.json"
FREEZE_PATH = ROOT / "reports/epoch9_model_freeze.json"
MODEL_ROOT = ROOT / "reports/epoch9_temporal_reference_checkpoints"
SEQUENCE_LENGTH = 128
HIDDEN_DIM = 48
EPOCHS = 250
SEEDS = (9009, 9010, 9011)
PHASES = (
    "approach_above",
    "approach_contact",
    "probe_inward",
    "probe_hold",
    "withdraw_contact",
    "withdraw_low_above",
    "return_retreat_side",
    "return_central",
    "return_prehome",
    "return_neutral",
    "withdraw_clear",
    "lift_clear",
    "return_central_high",
)
ROI_CENTER_32 = {"front": (24, 16), "back": (18, 14)}


@dataclass
class Example:
    episode_id: str
    demo_index: int
    label_front_heavier: int
    contrast: str
    front: np.ndarray
    back: np.ndarray
    classic: dict[str, np.ndarray]


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _resample(sequence: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    indices = np.rint(np.linspace(0, len(sequence) - 1, length)).astype(np.int64)
    return np.asarray(sequence[indices], dtype=np.float32)


def trace_features(trace_path: Path, slot: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(trace_path) as trace:
        keys = set(trace.files)
        forbidden = {"mass", "target_contact_eval", "target_pose", "reward", "expert_action"}
        if keys & forbidden:
            raise RuntimeError(f"privileged key in deployable trace: {sorted(keys & forbidden)}")
        action = np.asarray(trace["action"], dtype=np.float32)
        eef = np.asarray(trace["eef_pos"], dtype=np.float32)
        goal = np.asarray(trace["controller_goal_pos"], dtype=np.float32)
        error = np.asarray(trace["controller_error"], dtype=np.float32)[:, None]
        rgb = np.asarray(trace["rgb_diff_32"], dtype=np.float32) / 255.0
        phase = np.asarray(trace["phase"]).astype(str)
    if not (len(action) == len(eef) == len(goal) == len(error) == len(rgb) == len(phase)) or not len(action):
        raise RuntimeError(f"inconsistent or empty trace: {trace_path}")
    eef_delta = eef - eef[:1]
    eef_velocity = np.diff(eef, axis=0, prepend=eef[:1])
    tracking_vector = goal - eef
    x_center, y_center = ROI_CENTER_32[slot]
    roi = rgb[:, y_center - 4 : y_center + 5, x_center - 4 : x_center + 5].reshape(len(rgb), -1)
    global_pool = rgb.reshape(len(rgb), 4, 8, 4, 8).mean(axis=(2, 4)).reshape(len(rgb), -1)
    phase_one_hot = np.zeros((len(phase), len(PHASES)), dtype=np.float32)
    phase_to_index = {name: index for index, name in enumerate(PHASES)}
    for row_index, name in enumerate(phase):
        if name not in phase_to_index:
            raise RuntimeError(f"unknown controller phase {name!r}")
        phase_one_hot[row_index, phase_to_index[name]] = 1.0
    progress = np.linspace(0.0, 1.0, len(action), dtype=np.float32)[:, None]
    duration = np.full((len(action), 1), len(action) / 256.0, dtype=np.float32)
    slot_code = np.tile(np.asarray([1.0, 0.0] if slot == "front" else [0.0, 1.0], dtype=np.float32), (len(action), 1))
    sequence = np.concatenate(
        (
            action,
            eef_delta,
            eef_velocity,
            tracking_vector,
            error,
            roi,
            global_pool,
            phase_one_hot,
            progress,
            duration,
            slot_code,
        ),
        axis=1,
    )
    temporal_mad = np.abs(np.diff(rgb, axis=0)).mean(axis=(1, 2)) if len(rgb) > 1 else np.zeros(1)
    classic = {
        "single_error": np.asarray([float(error.mean())], dtype=np.float32),
        "epoch8_agentview": np.asarray(
            [
                float(error.mean()),
                float(np.quantile(error, 0.9)),
                float(error.max()),
                float(temporal_mad.mean()),
                float(temporal_mad.max()),
            ],
            dtype=np.float32,
        ),
        "endpoint": np.concatenate((sequence[0], sequence[-1], sequence[-1] - sequence[0])).astype(np.float32),
    }
    return _resample(sequence), classic


def load_examples(result_path: Path) -> tuple[list[Example], dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = result.get("summary", {})
    if not summary.get("execution_gate_pass"):
        raise RuntimeError(f"dataset execution gate did not pass: {result_path}")
    examples: list[Example] = []
    for row in result["rows"]:
        probes = {probe["slot"]: probe for probe in row["probes"]}
        if "front" not in probes:
            raise RuntimeError("front reference trace is missing")
        sequences: dict[str, np.ndarray] = {}
        classic_by_slot: dict[str, dict[str, np.ndarray]] = {}
        for slot in probes:
            sequences[slot], classic_by_slot[slot] = trace_features(ROOT / probes[slot]["trace_path"], slot)
        classic: dict[str, np.ndarray] = {}
        if "back" in probes:
            for name in classic_by_slot["front"]:
                front = classic_by_slot["front"][name]
                back = classic_by_slot["back"][name]
                classic[name] = np.concatenate((front, back, front - back)).astype(np.float32)
            back_sequence = sequences["back"]
        else:
            classic = {name: value.copy() for name, value in classic_by_slot["front"].items()}
            back_sequence = np.zeros_like(sequences["front"])
        masses = {float(row["front_mass_factor"]), float(row["back_mass_factor"])}
        contrast = "8x_vs_1x" if masses == {1.0, 8.0} else "4x_vs_2x"
        examples.append(
            Example(
                episode_id=str(row["episode_id"]),
                demo_index=int(row["demo_index"]),
                label_front_heavier=int(float(row["front_mass_factor"]) > float(row["back_mass_factor"])),
                contrast=contrast,
                front=sequences["front"],
                back=back_sequence,
                classic=classic,
            )
        )
    if len({example.episode_id for example in examples}) != len(examples):
        raise RuntimeError("duplicate episode identity")
    return examples, result


def arrays(examples: list[Example], *, shuffled: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paired = np.stack([np.stack((example.front, example.back)) for example in examples]).astype(np.float32)
    if shuffled:
        for example_index, example in enumerate(examples):
            for slot_index, slot in enumerate(("front", "back")):
                seed = int.from_bytes(hashlib.sha256(f"{example.episode_id}:{slot}".encode()).digest()[:4], "little")
                permutation = np.random.default_rng(seed).permutation(SEQUENCE_LENGTH)
                paired[example_index, slot_index] = paired[example_index, slot_index, permutation]
    labels = np.asarray([example.label_front_heavier for example in examples], dtype=np.float32)
    groups = np.asarray([example.demo_index for example in examples], dtype=np.int64)
    return paired, labels, groups


def standardizer(train: np.ndarray, variant: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    source = train[:, :1] if variant == "reference" else train
    mean = source.mean(axis=(0, 1, 2), keepdims=True)
    scale = source.std(axis=(0, 1, 2), keepdims=True)
    scale = np.where(scale < 1e-5, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def build_model(input_dim: int, variant: str) -> Any:
    import torch
    from torch import nn

    class TemporalComparator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.GELU())
            self.gru = nn.GRU(
                HIDDEN_DIM,
                HIDDEN_DIM // 2,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = nn.Linear(HIDDEN_DIM, 1)
            self.variant = variant
            if variant == "relational":
                self.context_gate = nn.Sequential(
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                    nn.Tanh(),
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                    nn.Tanh(),
                )
            elif variant == "independent":
                self.score = nn.Linear(HIDDEN_DIM, 1, bias=False)
            elif variant == "reference":
                self.light_prototype = nn.Parameter(torch.randn(HIDDEN_DIM) * 0.02)
                self.heavy_prototype = nn.Parameter(torch.randn(HIDDEN_DIM) * 0.02)
            else:
                raise ValueError(variant)

        def encode(self, sequence: Any) -> Any:
            hidden, _ = self.gru(self.projection(sequence))
            weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=1)
            return torch.sum(hidden * weights.unsqueeze(-1), dim=1)

        def forward(self, paired: Any) -> Any:
            front = self.encode(paired[:, 0])
            back = self.encode(paired[:, 1])
            if self.variant == "independent":
                return (self.score(front) - self.score(back)).squeeze(-1)
            if self.variant == "reference":
                light_distance = torch.sum((front - self.light_prototype) ** 2, dim=-1)
                heavy_distance = torch.sum((front - self.heavy_prototype) ** 2, dim=-1)
                return (light_distance - heavy_distance) / math.sqrt(HIDDEN_DIM)
            delta = front - back
            gate = self.context_gate(front + back)
            return torch.sum(delta * gate, dim=-1) / math.sqrt(HIDDEN_DIM)

    return TemporalComparator()


def fit_neural(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    variant: str,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    mean, scale = standardizer(train_x, variant)
    train_tensor = torch.from_numpy((train_x - mean) / scale)
    target_tensor = torch.from_numpy(train_y)
    test_tensor = torch.from_numpy((test_x - mean) / scale)
    model = build_model(train_x.shape[-1], variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(train_tensor), generator=generator)
        for start in range(0, len(permutation), 16):
            indices = permutation[start : start + 16]
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(train_tensor[indices]), target_tensor[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(test_tensor)).numpy()
    return probabilities, model, mean, scale


def grouped_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    dummy = np.zeros((len(groups), 1), dtype=np.float32)
    return list(GroupKFold(n_splits=5).split(dummy, groups=groups))


def cross_validated_neural(examples: list[Example], *, variant: str, shuffled: bool = False) -> np.ndarray:
    x, y, groups = arrays(examples, shuffled=shuffled)
    probabilities = np.zeros(len(examples), dtype=np.float64)
    for train_indices, test_indices in grouped_folds(groups):
        ensemble = np.zeros(len(test_indices), dtype=np.float64)
        for seed in SEEDS:
            values, _, _, _ = fit_neural(
                x[train_indices], y[train_indices], x[test_indices], variant=variant, seed=seed
            )
            ensemble += values / len(SEEDS)
        probabilities[test_indices] = ensemble
    return probabilities


def cross_validated_classic(examples: list[Example], name: str) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.stack([example.classic[name] for example in examples])
    y = np.asarray([example.label_front_heavier for example in examples], dtype=np.int64)
    groups = np.asarray([example.demo_index for example in examples], dtype=np.int64)
    probabilities = np.zeros(len(examples), dtype=np.float64)
    for train_indices, test_indices in grouped_folds(groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=9009))
        model.fit(x[train_indices], y[train_indices])
        probabilities[test_indices] = model.predict_proba(x[test_indices])[:, 1]
    return probabilities


def metric(examples: list[Example], probabilities: np.ndarray) -> dict[str, Any]:
    from scipy.stats import binomtest

    labels = np.asarray([example.label_front_heavier for example in examples], dtype=np.int64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    correct = predictions == labels
    by_contrast = {}
    for contrast in ("8x_vs_1x", "4x_vs_2x"):
        mask = np.asarray([example.contrast == contrast for example in examples])
        by_contrast[contrast] = {
            "episodes": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        }
    return {
        "episodes": len(examples),
        "successes": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "exact_binomial_one_sided_p_vs_half": float(binomtest(int(correct.sum()), len(correct), 0.5, alternative="greater").pvalue),
        "by_mass_contrast": by_contrast,
        "predictions": [
            {
                "episode_id": example.episode_id,
                "demo_index": example.demo_index,
                "contrast": example.contrast,
                "label_front_heavier": int(label),
                "probability_front_heavier": float(probability),
                "correct": bool(is_correct),
            }
            for example, label, probability, is_correct in zip(examples, labels, probabilities, correct, strict=True)
        ],
    }


def initial_rgb_audit(examples: list[Example], result_path: Path) -> dict[str, Any]:
    """Prove that the no-probe image cannot reveal the randomized mass label."""

    by_demo: dict[int, set[str]] = {}
    for example in examples:
        frame = result_path.parent / "frames" / f"{example.episode_id}_front_initial.png"
        by_demo.setdefault(example.demo_index, set()).add(sha256(frame))
    return {
        "frame_count": len(examples),
        "reset_identity_count": len(by_demo),
        "unique_initial_frame_hashes_per_identity": {
            str(demo_index): len(values) for demo_index, values in sorted(by_demo.items())
        },
        "all_mass_assignments_pixel_identical_within_identity": all(len(values) == 1 for values in by_demo.values()),
        "control_prediction": "balanced 0.5 prior because each identity contains two front-heavier and two front-lighter labels",
    }


def development() -> dict[str, Any]:
    examples, dataset = load_examples(DEVELOPMENT_RESULT)
    models = {
        "temporal_reference_belief": metric(examples, cross_validated_neural(examples, variant="reference")),
        "temporally_shuffled_reference": metric(
            examples, cross_validated_neural(examples, variant="reference", shuffled=True)
        ),
        "endpoint_only_logistic": metric(examples, cross_validated_classic(examples, "endpoint")),
        "epoch8_agentview_aggregate_logistic": metric(
            examples, cross_validated_classic(examples, "epoch8_agentview")
        ),
        "single_mean_controller_error_logistic": metric(
            examples, cross_validated_classic(examples, "single_error")
        ),
        "balanced_no_probe_prior": metric(examples, np.full(len(examples), 0.5, dtype=np.float64)),
        "initial_rgb_only": metric(examples, np.full(len(examples), 0.5, dtype=np.float64)),
        "offline_oracle_mass_label_headroom": metric(
            examples, np.asarray([example.label_front_heavier for example in examples], dtype=np.float64)
        ),
    }
    primary = models["temporal_reference_belief"]["accuracy"]
    strongest_non_temporal = max(
        models[name]["accuracy"]
        for name in (
            "endpoint_only_logistic",
            "epoch8_agentview_aggregate_logistic",
            "single_mean_controller_error_logistic",
            "balanced_no_probe_prior",
        )
    )
    continuation = bool(
        primary >= 0.75
        and primary > models["initial_rgb_only"]["accuracy"]
        and primary >= strongest_non_temporal
        and primary > models["temporally_shuffled_reference"]["accuracy"]
    )
    report = {
        "schema_version": "epoch9.temporal_reference.development.v1",
        "timestamp": timestamp(),
        "evidence_class": "DEVELOPMENT_GROUPED_CROSS_VALIDATION",
        "dataset_path": str(DEVELOPMENT_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": sha256(DEVELOPMENT_RESULT),
        "dataset_execution_summary": dataset["summary"],
        "initial_rgb_control_audit": initial_rgb_audit(examples, DEVELOPMENT_RESULT),
        "grouping": "five folds grouped by reset demo_index; all mass assignments from an identity remain in one fold",
        "model_config": {
            "sequence_length": SEQUENCE_LENGTH,
            "input_features": (
                f"{examples[0].front.shape[-1]} legal action/proprio/controller-history/"
                "agentview-response/phase/slot channels"
            ),
            "projection_dim": HIDDEN_DIM,
            "temporal_encoder": "bidirectional GRU with learned temporal attention",
            "belief_head": (
                "learned heavy/light temporal-response prototypes; the front heavy probability defines the front belief "
                "and its complement defines the back belief under the frozen task generator"
            ),
            "epochs": EPOCHS,
            "optimizer": "AdamW(lr=0.003, weight_decay=0.001), batch=16, grad_clip=1",
            "ensemble_seeds": list(SEEDS),
        },
        "models": models,
        "epoch8_exact_checkpoint_note": (
            "The exact seven-feature Epoch 8 scripted-lift LDA requires wrist-video features and a different expert-action probe. "
            "It is not applied out of distribution; the five shared controller-error/agentview aggregate features are refit "
            "inside each grouped development fold and reported explicitly as an adapted control."
        ),
        "development_continuation_gate": {
            "primary_accuracy_min": 0.75,
            "must_beat_initial_rgb": True,
            "must_not_underperform_strongest_non_temporal": True,
            "must_beat_shuffled_time": True,
            "strongest_non_temporal_accuracy": strongest_non_temporal,
            "pass": continuation,
        },
        "decision": "FREEZE_FOR_VALIDATION" if continuation else "TEMPORAL_RELATIONAL_MECHANISM_NOT_SUPPORTED",
    }
    atomic_write_json(DEVELOPMENT_REPORT, report)
    return report


def fit_classic_full(examples: list[Example], name: str) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.stack([example.classic[name] for example in examples])
    y = np.asarray([example.label_front_heavier for example in examples], dtype=np.int64)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=9009).fit(scaler.transform(x), y)
    return {
        "name": name,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }


def save_neural_ensemble(examples: list[Example], variant: str, *, shuffled: bool = False) -> dict[str, Any]:
    import torch

    x, y, _ = arrays(examples, shuffled=shuffled)
    mean, scale = standardizer(x, variant)
    variant_root = MODEL_ROOT / (f"shuffled_{variant}" if shuffled else variant)
    variant_root.mkdir(parents=True, exist_ok=True)
    scaler_path = variant_root / "standardizer.npz"
    temporary = scaler_path.with_suffix(".tmp.npz")
    np.savez(temporary, mean=mean, scale=scale)
    temporary.replace(scaler_path)
    checkpoints = []
    for seed in SEEDS:
        _, model, fitted_mean, fitted_scale = fit_neural(x, y, x[:1], variant=variant, seed=seed)
        if not (np.array_equal(mean, fitted_mean) and np.array_equal(scale, fitted_scale)):
            raise RuntimeError("inconsistent final standardizer")
        checkpoint = variant_root / f"seed{seed}.pt"
        temporary_checkpoint = checkpoint.with_suffix(".tmp.pt")
        torch.save(model.state_dict(), temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint)
        checkpoints.append(
            {
                "seed": seed,
                "path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(checkpoint),
            }
        )
    return {
        "variant": variant,
        "temporally_shuffled": shuffled,
        "standardizer_path": str(scaler_path.relative_to(ROOT)).replace("\\", "/"),
        "standardizer_sha256": sha256(scaler_path),
        "checkpoints": checkpoints,
    }


def freeze() -> dict[str, Any]:
    if not DEVELOPMENT_REPORT.exists():
        raise RuntimeError("run development grouped cross-validation before freezing")
    development_report = json.loads(DEVELOPMENT_REPORT.read_text(encoding="utf-8"))
    if not development_report["development_continuation_gate"]["pass"]:
        raise RuntimeError("development continuation gate failed; validation remains unauthorized")
    if FREEZE_PATH.exists():
        raise FileExistsError(f"refusing to overwrite freeze record: {FREEZE_PATH}")
    examples, dataset = load_examples(DEVELOPMENT_RESULT)
    ensembles = {
        "temporal_reference_belief": save_neural_ensemble(examples, "reference"),
        "temporally_shuffled_reference": save_neural_ensemble(examples, "reference", shuffled=True),
    }
    script_path = Path(__file__).resolve()
    record = {
        "schema_version": "epoch9.temporal_reference.model_freeze.v1",
        "frozen_at": timestamp(),
        "status": "FROZEN_BEFORE_VALIDATION",
        "protocol_path": "reports/epoch9_active_grounding_protocol_rotation2.json",
        "development_result_path": str(DEVELOPMENT_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "development_result_sha256": sha256(DEVELOPMENT_RESULT),
        "development_report_path": str(DEVELOPMENT_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "development_report_sha256": sha256(DEVELOPMENT_REPORT),
        "model_script_path": str(script_path.relative_to(ROOT)).replace("\\", "/"),
        "model_script_sha256": sha256(script_path),
        "model_config": development_report["model_config"],
        "neural_ensembles": ensembles,
        "classic_controls": {
            name: fit_classic_full(examples, name)
            for name in ("endpoint", "epoch8_agentview", "single_error")
        },
        "validation_accessed": False,
        "confirmation_accessed": False,
        "dataset_execution_summary": dataset["summary"],
        "initial_rgb_control_audit": initial_rgb_audit(examples, DEVELOPMENT_RESULT),
    }
    atomic_write_json(FREEZE_PATH, record)
    return record


def predict_neural_ensemble(
    examples: list[Example],
    ensemble: dict[str, Any],
) -> np.ndarray:
    import torch

    x, _, _ = arrays(examples, shuffled=bool(ensemble["temporally_shuffled"]))
    with np.load(ROOT / ensemble["standardizer_path"]) as values:
        mean = np.asarray(values["mean"], dtype=np.float32)
        scale = np.asarray(values["scale"], dtype=np.float32)
    tensor = torch.from_numpy((x - mean) / scale)
    probabilities = np.zeros(len(examples), dtype=np.float64)
    for checkpoint in ensemble["checkpoints"]:
        path = ROOT / checkpoint["path"]
        if sha256(path) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        model = build_model(x.shape[-1], str(ensemble["variant"]))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            probabilities += torch.sigmoid(model(tensor)).numpy() / len(ensemble["checkpoints"])
    return probabilities


def predict_classic(examples: list[Example], parameters: dict[str, Any]) -> np.ndarray:
    x = np.stack([example.classic[parameters["name"]] for example in examples])
    mean = np.asarray(parameters["mean"], dtype=np.float64)
    scale = np.asarray(parameters["scale"], dtype=np.float64)
    coefficient = np.asarray(parameters["coefficient"], dtype=np.float64)
    logits = ((x - mean) / scale) @ coefficient + float(parameters["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def evaluate(partition: str, result_path: Path) -> dict[str, Any]:
    if partition not in {"validation", "confirmation"}:
        raise ValueError("frozen evaluation is restricted to validation or confirmation")
    output = ROOT / f"reports/epoch9_{partition}_adjudication.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite one-shot adjudication: {output}")
    freeze_record = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze_record.get("status") != "FROZEN_BEFORE_VALIDATION":
        raise RuntimeError("model freeze status is invalid")
    script_path = Path(__file__).resolve()
    if sha256(script_path) != freeze_record["model_script_sha256"]:
        raise RuntimeError("model/evaluation source changed after freeze")
    if partition == "confirmation":
        validation_path = ROOT / "reports/epoch9_validation_adjudication.json"
        if not validation_path.exists() or not json.loads(validation_path.read_text(encoding="utf-8")).get(
            "confirmation_authorized", False
        ):
            raise RuntimeError("confirmation is not authorized by validation")
    examples, dataset = load_examples(result_path)
    ensembles = freeze_record["neural_ensembles"]
    model_metrics = {
        "temporal_reference_belief": metric(
            examples, predict_neural_ensemble(examples, ensembles["temporal_reference_belief"])
        ),
        "temporally_shuffled_reference": metric(
            examples, predict_neural_ensemble(examples, ensembles["temporally_shuffled_reference"])
        ),
    }
    for report_name, parameter_name in (
        ("endpoint_only_logistic", "endpoint"),
        ("epoch8_agentview_aggregate_logistic", "epoch8_agentview"),
        ("single_mean_controller_error_logistic", "single_error"),
    ):
        model_metrics[report_name] = metric(
            examples,
            predict_classic(examples, freeze_record["classic_controls"][parameter_name]),
        )
    model_metrics["balanced_no_probe_prior"] = metric(
        examples, np.full(len(examples), 0.5, dtype=np.float64)
    )
    model_metrics["initial_rgb_only"] = metric(examples, np.full(len(examples), 0.5, dtype=np.float64))
    model_metrics["offline_oracle_mass_label_headroom"] = metric(
        examples, np.asarray([example.label_front_heavier for example in examples], dtype=np.float64)
    )
    protocol = json.loads((ROOT / freeze_record["protocol_path"]).read_text(encoding="utf-8"))
    gates = protocol["validation_gates"]
    primary = model_metrics["temporal_reference_belief"]
    non_temporal_names = (
        "endpoint_only_logistic",
        "epoch8_agentview_aggregate_logistic",
        "single_mean_controller_error_logistic",
        "balanced_no_probe_prior",
    )
    strongest_non_temporal = max(model_metrics[name]["accuracy"] for name in non_temporal_names)
    contrast_pass = all(
        value["accuracy"] is not None and value["accuracy"] >= gates["mass_contrast_accuracy_each_min"]
        for value in primary["by_mass_contrast"].values()
    )
    execution_pass = bool(dataset["summary"]["execution_gate_pass"])
    mechanism_pass = bool(
        primary["accuracy"] >= gates["physical_pair_accuracy_min"]
        and primary["exact_binomial_one_sided_p_vs_half"] <= gates["exact_binomial_one_sided_p_vs_half_max"]
        and contrast_pass
        and primary["accuracy"] > model_metrics["balanced_no_probe_prior"]["accuracy"]
        and primary["accuracy"] > model_metrics["initial_rgb_only"]["accuracy"]
        and primary["accuracy"] >= strongest_non_temporal
        and primary["accuracy"] > model_metrics["temporally_shuffled_reference"]["accuracy"]
    )
    passed = bool(execution_pass and mechanism_pass)
    report = {
        "schema_version": f"epoch9.temporal_reference.{partition}_adjudication.v1",
        "timestamp": timestamp(),
        "evidence_class": partition.upper(),
        "partition": partition,
        "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": sha256(result_path),
        "freeze_path": str(FREEZE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "freeze_sha256": sha256(FREEZE_PATH),
        "dataset_execution_summary": dataset["summary"],
        "initial_rgb_control_audit": initial_rgb_audit(examples, result_path),
        "models": model_metrics,
        "gates": {
            "execution_pass": execution_pass,
            "mechanism_pass": mechanism_pass,
            "contrast_pass": contrast_pass,
            "strongest_non_temporal_accuracy": strongest_non_temporal,
            "temporal_order_ablation_must_be_lower": True,
            "all_pass": passed,
        },
        "confirmation_authorized": bool(partition == "validation" and passed),
        "decision": (
            "VALIDATION_PASS_CONFIRMATION_AUTHORIZED"
            if partition == "validation" and passed
            else "CONFIRMATION_PASS"
            if partition == "confirmation" and passed
            else f"{partition.upper()}_GATE_FAIL"
        ),
    }
    atomic_write_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "freeze", "evaluate"), required=True)
    parser.add_argument("--partition", choices=("validation", "confirmation"))
    parser.add_argument("--result-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "development":
        result = development()
    elif args.mode == "freeze":
        result = freeze()
    else:
        if args.partition is None or args.result_path is None:
            raise ValueError("--partition and --result-path are required for evaluation")
        result = evaluate(args.partition, args.result_path.resolve())
    print(json.dumps({"decision": result.get("decision", result.get("status"))}, sort_keys=True))


if __name__ == "__main__":
    main()
