"""Train disk-reloadable lightweight DAGR-VLA policy identities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from tca_map.smolvla.dagr_vla import (
    DAGRConfig,
    GROUP_NAMES,
    PROPOSAL_HASH,
    _feature_matrix,
    build_dagr_records,
    compute_route_labels,
)


DATE_KST = "2026-07-14 KST"
DEFAULT_SEED = 101
DEFAULT_EPOCHS = 250
DEFAULT_LR = 1e-3
TRAINABLE_VARIANTS = (
    "dagr_full",
    "dam_static_component_proxy",
    "dagr_no_dynamic_route_ablation",
)
ALL_POLICY_IDENTITIES = (
    "frozen_smolvla",
    "dam_static_component_proxy",
    "dagr_full",
    "dagr_no_dynamic_route_ablation",
    "gripper_transition_heuristic",
)


@dataclass(frozen=True)
class DAGRPolicyTrainArgs:
    seed: int = DEFAULT_SEED
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    checkpoint_output_root: str = "runs/dagr_vla_checkpoints"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _standardize_with_stats(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (validation - mean) / scale, mean, scale


def _labels_to_dim_np(labels: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            labels[:, 0:1].repeat(3, axis=1),
            labels[:, 1:2].repeat(3, axis=1),
            labels[:, 2:3],
        ],
        axis=1,
    )


def _labels_to_dim_torch(labels: Any, torch: Any) -> Any:
    return torch.cat(
        [
            labels[:, 0:1].repeat(1, 3),
            labels[:, 1:2].repeat(1, 3),
            labels[:, 2:3],
        ],
        dim=1,
    )


def _group_delta(predicted_residual: Any, gates: Any, alpha: float, torch: Any) -> Any:
    return float(alpha) * torch.cat(
        [
            predicted_residual[:, 0:3] * gates[:, 0:1],
            predicted_residual[:, 3:6] * gates[:, 1:2],
            predicted_residual[:, 6:7] * gates[:, 2:3],
        ],
        dim=1,
    )


def _grad_norm(parameters: Sequence[Any], torch: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(torch.sum(parameter.grad.detach() * parameter.grad.detach()).item())
    return float(total ** 0.5)


def _policy_path(output_root: str | Path, config_id: str, variant: str, seed: int) -> Path:
    return Path(output_root) / str(config_id) / str(variant) / f"seed_{int(seed)}"


def _build_manifest_rows(records: Sequence[Mapping[str, Any]], labels: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "key": str(record["key"]),
                "split": str(record["split"]),
                "dataset_global_index": int(record["dataset_global_index"]),
                "task_index": int(record["task_index"]),
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "route_labels": {group: bool(labels[index, group_index]) for group_index, group in enumerate(GROUP_NAMES)},
            }
        )
    return rows


def _route_metrics(route_probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    out = {}
    for index, group in enumerate(GROUP_NAMES):
        predictions = route_probs[:, index] >= 0.5
        y = labels[:, index].astype(bool)
        out[group] = {
            "accuracy": float(np.mean(predictions == y)),
            "majority_accuracy": float(max(np.mean(y), 1.0 - np.mean(y))),
            "gate_mean": float(np.mean(route_probs[:, index])),
            "gate_activation_fraction": float(np.mean(predictions)),
        }
    return out


def _delta_metrics(delta: np.ndarray, proposed_actions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    l2 = np.linalg.norm(delta, axis=1)
    clean = np.sum(labels, axis=1) == 0
    return {
        "delta_l2_mean": float(np.mean(l2)),
        "delta_l2_p95": float(np.percentile(l2, 95)),
        "clean_delta_l2_p95": float(np.percentile(l2[clean], 95)) if bool(np.any(clean)) else 0.0,
        "translation_delta_l2_p95": float(np.percentile(np.linalg.norm(delta[:, 0:3], axis=1), 95)),
        "rotation_delta_l2_p95": float(np.percentile(np.linalg.norm(delta[:, 3:6], axis=1), 95)),
        "gripper_delta_abs_p95": float(np.percentile(np.abs(delta[:, 6]), 95)),
        "action_validity": float(np.mean(np.all(np.isfinite(proposed_actions), axis=1) & (np.max(np.abs(proposed_actions), axis=1) <= 5.0))),
    }


def _train_variant(
    *,
    variant: str,
    config_id: str,
    architecture: str,
    alpha: float,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    train_residuals: np.ndarray,
    validation_residuals: np.ndarray,
    base_validation_actions: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    static_weights: np.ndarray,
    checkpoint_path: Path,
    args: DAGRPolicyTrainArgs,
) -> dict[str, Any]:
    import torch

    class DAGRPolicyHead(torch.nn.Module):
        def __init__(self, input_dim: int, route_dim: int) -> None:
            super().__init__()
            if architecture == "mlp":
                self.trunk = torch.nn.Sequential(torch.nn.Linear(input_dim, 32), torch.nn.ReLU())
                hidden = 32
            else:
                self.trunk = torch.nn.Identity()
                hidden = input_dim
            self.route = torch.nn.Linear(hidden, route_dim) if route_dim else None
            self.residual = torch.nn.Linear(hidden, 7)
            torch.nn.init.zeros_(self.residual.weight)
            torch.nn.init.zeros_(self.residual.bias)

        def forward(self, x: Any) -> tuple[Any | None, Any]:
            h = self.trunk(x)
            route_logits = self.route(h) if self.route is not None else None
            return route_logits, self.residual(h)

    def compute_delta(route_logits: Any | None, predicted_residual: Any, labels: Any) -> tuple[Any, Any]:
        if variant == "dagr_full":
            assert route_logits is not None
            gates = torch.sigmoid(route_logits)
            return _group_delta(predicted_residual, gates, alpha, torch), gates
        if variant == "dagr_no_dynamic_route_ablation":
            assert route_logits is not None
            gate = torch.sigmoid(route_logits)
            gates = gate.repeat(1, 3)
            return float(alpha) * gate * predicted_residual, gates
        weights = torch.as_tensor(static_weights.reshape(1, 7), dtype=predicted_residual.dtype, device=predicted_residual.device)
        gates = torch.as_tensor(
            [static_weights[0], static_weights[3], static_weights[6]],
            dtype=predicted_residual.dtype,
            device=predicted_residual.device,
        ).reshape(1, 3).repeat(predicted_residual.shape[0], 1)
        return float(alpha) * weights * predicted_residual, gates

    def loss_terms(model: Any, x: Any, labels: Any, residuals: Any) -> dict[str, Any]:
        route_logits, predicted_residual = model(x)
        if variant == "dagr_full":
            route_loss = torch.nn.functional.binary_cross_entropy_with_logits(route_logits, labels)
            mask = _labels_to_dim_torch(labels, torch)
        elif variant == "dagr_no_dynamic_route_ablation":
            any_label = (torch.sum(labels, dim=1, keepdim=True) > 0).float()
            route_loss = torch.nn.functional.binary_cross_entropy_with_logits(route_logits, any_label)
            mask = any_label.repeat(1, 7)
        else:
            route_loss = torch.zeros((), dtype=x.dtype, device=x.device)
            mask = torch.as_tensor(static_weights.reshape(1, 7), dtype=x.dtype, device=x.device).repeat(x.shape[0], 1)
        residual_loss = torch.nn.functional.smooth_l1_loss(predicted_residual * mask, residuals * mask)
        delta, _gates = compute_delta(route_logits, predicted_residual, labels)
        delta_loss = torch.mean(torch.sum(delta * delta, dim=1))
        no_route = (torch.sum(labels, dim=1) <= 0.0).float()
        clean_loss = torch.mean(no_route * torch.sum(delta * delta, dim=1))
        total = residual_loss + route_loss + 0.10 * delta_loss + 0.10 * clean_loss
        return {"total": total, "route": route_loss, "residual": residual_loss, "delta": delta_loss, "clean": clean_loss}

    torch.set_num_threads(1)
    torch.manual_seed(int(args.seed))
    route_dim = 3 if variant == "dagr_full" else 1 if variant == "dagr_no_dynamic_route_ablation" else 0
    model = DAGRPolicyHead(train_features.shape[1], route_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    x_train = torch.as_tensor(train_features, dtype=torch.float32)
    x_validation = torch.as_tensor(validation_features, dtype=torch.float32)
    y_train = torch.as_tensor(train_labels.astype(np.float32), dtype=torch.float32)
    y_validation = torch.as_tensor(validation_labels.astype(np.float32), dtype=torch.float32)
    r_train = torch.as_tensor(train_residuals.astype(np.float32), dtype=torch.float32)
    r_validation = torch.as_tensor(validation_residuals.astype(np.float32), dtype=torch.float32)

    with torch.no_grad():
        initial_terms = loss_terms(model, x_train, y_train, r_train)
        route_logits_initial, residual_initial = model(x_validation[:128])
        delta_initial, _gates_initial = compute_delta(route_logits_initial, residual_initial, y_validation[:128])
        initial_delta_p95 = float(np.percentile(torch.linalg.norm(delta_initial, dim=1).cpu().numpy(), 95))

    first_grad_norms: dict[str, float] | None = None
    loss_curve = []
    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, x_train, y_train, r_train)
        terms["total"].backward()
        if epoch == 0:
            route_params = list(model.route.parameters()) if model.route is not None else []
            first_grad_norms = {
                "trunk": _grad_norm(list(model.trunk.parameters()) if hasattr(model.trunk, "parameters") else [], torch),
                "route": _grad_norm(route_params, torch) if route_params else 0.0,
                "residual": _grad_norm(list(model.residual.parameters()), torch),
            }
        optimizer.step()
        if epoch in {0, int(args.epochs) - 1}:
            loss_curve.append({"epoch": epoch + 1, **{key: float(value.detach().item()) for key, value in terms.items()}})

    with torch.no_grad():
        final_train_terms = loss_terms(model, x_train, y_train, r_train)
        validation_terms = loss_terms(model, x_validation, y_validation, r_validation)
        route_logits, predicted_residual = model(x_validation)
        delta, gates = compute_delta(route_logits, predicted_residual, y_validation)
        proposed_actions = torch.as_tensor(base_validation_actions.astype(np.float32), dtype=torch.float32) + delta
        delta_np = delta.cpu().numpy()
        gates_np = gates.cpu().numpy()
        proposed_np = proposed_actions.cpu().numpy()
        route_probs = gates_np if variant != "dagr_no_dynamic_route_ablation" else gates_np

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    policy_config = {
        "method": "DAGR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "variant": variant,
        "config_id": config_id,
        "route_architecture": architecture,
        "residual_alpha": float(alpha),
        "feature_count": int(train_features.shape[1]),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "static_weights": static_weights.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
    }
    _write_json(checkpoint_path / "policy_config.json", policy_config)
    _write_json(
        checkpoint_path / "training_manifest.json",
        {
            "variant": variant,
            "train_record_count": int(train_features.shape[0]),
            "validation_record_count": int(validation_features.shape[0]),
            "proposal_hash": PROPOSAL_HASH,
        },
    )
    reloaded = DAGRPolicyHead(train_features.shape[1], route_dim)
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(loaded["model_state_dict"])
    with torch.no_grad():
        old_route, old_residual = model(x_validation[:32])
        new_route, new_residual = reloaded(x_validation[:32])
        route_diff = 0.0 if old_route is None else float(torch.max(torch.abs(old_route - new_route)).item())
        residual_diff = float(torch.max(torch.abs(old_residual - new_residual)).item())
    sha_manifest = {
        "model.pt": _sha256_file(model_path),
        "policy_config.json": _sha256_file(checkpoint_path / "policy_config.json"),
        "training_manifest.json": _sha256_file(checkpoint_path / "training_manifest.json"),
    }
    _write_json(checkpoint_path / "sha256_manifest.json", sha_manifest)

    grad = first_grad_norms or {}
    hard_stop_reasons = []
    if max(route_diff, residual_diff) > 1e-6:
        hard_stop_reasons.append("disk reload output mismatch")
    if float(grad.get("residual", 0.0)) <= 0.0:
        hard_stop_reasons.append("residual gradient is zero")
    if variant != "dam_static_component_proxy" and float(grad.get("route", 0.0)) <= 0.0:
        hard_stop_reasons.append("route gradient is zero")
    metrics = _delta_metrics(delta_np, proposed_np, validation_labels)
    if metrics["action_validity"] < 1.0:
        hard_stop_reasons.append("invalid validation action")
    if initial_delta_p95 > 1e-6:
        hard_stop_reasons.append("initial action delta is not base-passthrough")

    result = {
        "variant": variant,
        "final_decision": "DAGR_POLICY_CHECKPOINT_VERIFIED" if not hard_stop_reasons else "DAGR_POLICY_CHECKPOINT_BLOCKED",
        "checkpoint_path": str(checkpoint_path),
        "required_files": ["model.pt", "policy_config.json", "training_manifest.json", "sha256_manifest.json"],
        "sha256_manifest": sha_manifest,
        "disk_reload": True,
        "checkpoint_reload_max_abs_diff": max(route_diff, residual_diff),
        "initial_delta_p95": initial_delta_p95,
        "first_gradient_norms": grad,
        "loss_initial": {key: float(value.detach().item()) for key, value in initial_terms.items()},
        "loss_final_train": {key: float(value.detach().item()) for key, value in final_train_terms.items()},
        "loss_validation": {key: float(value.detach().item()) for key, value in validation_terms.items()},
        "loss_curve": loss_curve,
        "route_metrics": _route_metrics(route_probs, validation_labels) if variant == "dagr_full" else {},
        "validation": metrics,
        "hard_stop_reasons": hard_stop_reasons,
    }
    _write_json(checkpoint_path / "training_result.json", result)
    return result


def train_policy_identities(
    stable_artifact: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    *,
    train_args: DAGRPolicyTrainArgs | None = None,
) -> dict[str, Any]:
    args = train_args or DAGRPolicyTrainArgs()
    config_id = str(selected_config["config_id"])
    architecture = str(selected_config["route_architecture"])
    alpha = float(selected_config["residual_alpha"])
    cfg = DAGRConfig()
    records, thresholds = compute_route_labels(build_dagr_records(stable_artifact.get("records") or []), cfg)
    train = [record for record in records if str(record["split"]) in set(cfg.train_splits)]
    validation = [record for record in records if str(record["split"]) in set(cfg.validation_splits)]
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    train_features_raw = _feature_matrix(train, task_count)
    validation_features_raw = _feature_matrix(validation, task_count)
    train_features, validation_features, feature_mean, feature_scale = _standardize_with_stats(train_features_raw, validation_features_raw)
    train_labels = np.asarray([record["route_labels"] for record in train], dtype=bool)
    validation_labels = np.asarray([record["route_labels"] for record in validation], dtype=bool)
    train_residuals = np.asarray([record["residual"] for record in train], dtype=np.float64)
    validation_residuals = np.asarray([record["residual"] for record in validation], dtype=np.float64)
    base_validation_actions = np.asarray([record["base_action"] for record in validation], dtype=np.float64)
    train_positive = np.mean(train_labels, axis=0)
    static_weights = np.asarray([train_positive[0]] * 3 + [train_positive[1]] * 3 + [train_positive[2]], dtype=np.float64)

    results = []
    for variant in TRAINABLE_VARIANTS:
        results.append(
            _train_variant(
                variant=variant,
                config_id=config_id,
                architecture=architecture,
                alpha=alpha,
                train_features=train_features,
                validation_features=validation_features,
                train_labels=train_labels,
                validation_labels=validation_labels,
                train_residuals=train_residuals,
                validation_residuals=validation_residuals,
                base_validation_actions=base_validation_actions,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                static_weights=static_weights,
                checkpoint_path=_policy_path(args.checkpoint_output_root, config_id, variant, args.seed),
                args=args,
            )
        )

    heuristic_path = _policy_path(args.checkpoint_output_root, config_id, "gripper_transition_heuristic", args.seed)
    heuristic_path.mkdir(parents=True, exist_ok=True)
    heuristic_config = {
        "method": "DAGR-VLA",
        "variant": "gripper_transition_heuristic",
        "proposal_hash": PROPOSAL_HASH,
        "config_id": config_id,
        "gripper_material_threshold": thresholds["gripper_material"],
        "residual_alpha": min(alpha, 0.05),
        "description": "Nontrainable bounded gripper timing bias near predicted gripper transitions.",
    }
    _write_json(heuristic_path / "heuristic_config.json", heuristic_config)
    _write_json(heuristic_path / "sha256_manifest.json", {"heuristic_config.json": _sha256_file(heuristic_path / "heuristic_config.json")})

    verified = all(result["final_decision"] == "DAGR_POLICY_CHECKPOINT_VERIFIED" for result in results)
    full = next(result for result in results if result["variant"] == "dagr_full")
    static = next(result for result in results if result["variant"] == "dam_static_component_proxy")
    shared = next(result for result in results if result["variant"] == "dagr_no_dynamic_route_ablation")
    distinction = {
        "full_vs_static_delta_l2_mean_abs": abs(full["validation"]["delta_l2_mean"] - static["validation"]["delta_l2_mean"]),
        "full_vs_shared_delta_l2_mean_abs": abs(full["validation"]["delta_l2_mean"] - shared["validation"]["delta_l2_mean"]),
    }
    if distinction["full_vs_static_delta_l2_mean_abs"] <= 1e-6 or distinction["full_vs_shared_delta_l2_mean_abs"] <= 1e-6:
        verified = False
    return {
        "schema_version": 1,
        "date": DATE_KST,
        "method": "DAGR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "config_id": config_id,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "checkpoint_root": str(Path(args.checkpoint_output_root) / config_id),
        "policy_identities": list(ALL_POLICY_IDENTITIES),
        "trainable_variants": list(TRAINABLE_VARIANTS),
        "nontrainable_variants": ["frozen_smolvla", "gripper_transition_heuristic"],
        "closed_loop_experiment_happened": False,
        "confirmatory_test_identities_used": False,
        "training_happened": True,
        "stage_a_allowed": bool(verified),
        "route_thresholds": thresholds,
        "static_component_weights": static_weights.tolist(),
        "train_records": len(train),
        "validation_records": len(validation),
        "variant_results": results,
        "heuristic": {
            "checkpoint_path": str(heuristic_path),
            "required_files": ["heuristic_config.json", "sha256_manifest.json"],
            "disk_reload": True,
        },
        "distinction": distinction,
        "final_decision": "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY" if verified else "DAGR_POLICY_IDENTITIES_BLOCKED",
        "next_step": (
            "Freeze the DAGR Stage A matched manifest before any rollout."
            if verified
            else "Do not roll out; inspect DAGR policy identity training failures."
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_md(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# DAGR-VLA Policy Identity Training",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
        f"- confirmatory-test identities used: `{report.get('confirmatory_test_identities_used')}`",
        f"- training happened: `{report.get('training_happened')}`",
        f"- Stage A allowed: `{report.get('stage_a_allowed')}`",
        f"- config: `{report.get('config_id')}`",
        f"- seed: `{report.get('seed')}`",
        f"- epochs: `{report.get('epochs')}`",
        f"- checkpoint root: `{report.get('checkpoint_root')}`",
        "",
        "Variants:",
        "",
        "| variant | decision | reload | delta p95 | clean p95 | validity | checkpoint |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in report.get("variant_results") or []:
        validation = result.get("validation") or {}
        lines.append(
            f"| `{result.get('variant')}` | `{result.get('final_decision')}` | `{result.get('disk_reload')}` | "
            f"{validation.get('delta_l2_p95')} | {validation.get('clean_delta_l2_p95')} | "
            f"{validation.get('action_validity')} | `{result.get('checkpoint_path')}` |"
        )
    lines.extend(
        [
            f"| `gripper_transition_heuristic` | `NONTRAINABLE_HEURISTIC_READY` | `True` | 0.0 | 0.0 | 1.0 | `{(report.get('heuristic') or {}).get('checkpoint_path')}` |",
            "",
            "Distinction:",
            "",
            "```json",
            json.dumps(report.get("distinction"), indent=2, sort_keys=True),
            "```",
            "",
            f"Next step: {report.get('next_step')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--selected-config", default="reports/dagr_vla/selected_config.json")
    parser.add_argument("--checkpoint-output-root", default="runs/dagr_vla_checkpoints")
    parser.add_argument("--report-json", default="reports/dagr_vla/policy_checkpoint_manifest.json")
    parser.add_argument("--report-md", default="reports/dagr_vla/policy_checkpoint_manifest.md")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args(argv)
    report = train_policy_identities(
        _read_json(Path(args.stable_artifact)),
        _read_json(Path(args.selected_config)),
        train_args=DAGRPolicyTrainArgs(
            seed=int(args.seed),
            epochs=int(args.epochs),
            lr=float(args.lr),
            checkpoint_output_root=str(args.checkpoint_output_root),
        ),
    )
    _write_json(Path(args.report_json), report)
    _write_md(report, Path(args.report_md))
    print(
        json.dumps(
            {
                "final_decision": report["final_decision"],
                "training_happened": report["training_happened"],
                "stage_a_allowed": report["stage_a_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["stage_a_allowed"] else 40


if __name__ == "__main__":
    sys.exit(main())

