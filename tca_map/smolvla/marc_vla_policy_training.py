"""Train disk-reloadable lightweight MARC-VLA policy identities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from tca_map.smolvla.marc_vla import (
    MARCConfig,
    PROPOSAL_HASH,
    _feature_matrix,
    build_marc_records,
    compute_disagreement_labels,
)


DATE_KST = "2026-07-15 KST"
DEFAULT_SEED = 101
DEFAULT_EPOCHS = 250
DEFAULT_LR = 1e-3
TRAINABLE_VARIANTS = (
    "openvla_oft_l1_proxy",
    "marc_full",
    "marc_no_disagreement_gate_ablation",
    "static_l1_mixture_baseline",
)
ALL_POLICY_IDENTITIES = (
    "frozen_smolvla",
    "openvla_oft_l1_proxy",
    "marc_full",
    "marc_no_disagreement_gate_ablation",
    "static_l1_mixture_baseline",
)


@dataclass(frozen=True)
class MARCPolicyTrainArgs:
    seed: int = DEFAULT_SEED
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    checkpoint_output_root: str = "runs/marc_vla_checkpoints"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _grad_norm(parameters: Sequence[Any], torch: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(torch.sum(parameter.grad.detach() * parameter.grad.detach()).item())
    return float(total**0.5)


def _policy_path(output_root: str | Path, config_id: str, variant: str, seed: int) -> Path:
    return Path(output_root) / str(config_id) / str(variant) / f"seed_{int(seed)}"


def _action_validity(actions: np.ndarray) -> float:
    if actions.size == 0:
        return 0.0
    return float(np.mean(np.all(np.isfinite(actions), axis=1) & (np.max(np.abs(actions), axis=1) <= 5.0)))


def _delta_metrics(delta: np.ndarray, proposed_actions: np.ndarray, target_actions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    l2 = np.linalg.norm(delta, axis=1)
    clean = ~labels.astype(bool)
    return {
        "delta_l2_mean": float(np.mean(l2)),
        "delta_l2_p95": float(np.percentile(l2, 95)),
        "clean_delta_l2_p95": float(np.percentile(l2[clean], 95)) if bool(np.any(clean)) else 0.0,
        "translation_delta_l2_p95": float(np.percentile(np.linalg.norm(delta[:, 0:3], axis=1), 95)),
        "rotation_delta_l2_p95": float(np.percentile(np.linalg.norm(delta[:, 3:6], axis=1), 95)),
        "gripper_delta_abs_p95": float(np.percentile(np.abs(delta[:, 6]), 95)),
        "action_l2_to_target": float(np.mean(np.linalg.norm(proposed_actions - target_actions, axis=1))),
        "action_validity": _action_validity(proposed_actions),
    }


def _train_variant(
    *,
    variant: str,
    config_id: str,
    architecture: str,
    alpha: float,
    static_beta: float,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    train_residuals: np.ndarray,
    validation_residuals: np.ndarray,
    base_train_actions: np.ndarray,
    base_validation_actions: np.ndarray,
    target_train_actions: np.ndarray,
    target_validation_actions: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    checkpoint_path: Path,
    args: MARCPolicyTrainArgs,
    config: MARCConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    class MARCPolicyHead(torch.nn.Module):
        def __init__(self, input_dim: int, use_gate: bool) -> None:
            super().__init__()
            if architecture == "mlp":
                self.trunk = torch.nn.Sequential(torch.nn.Linear(input_dim, config.mlp_hidden_dim), torch.nn.ReLU())
                hidden_dim = config.mlp_hidden_dim
            else:
                self.trunk = torch.nn.Identity()
                hidden_dim = input_dim
            self.gate = torch.nn.Linear(hidden_dim, 1) if use_gate else None
            self.residual = torch.nn.Linear(hidden_dim, 7)
            if self.gate is not None:
                torch.nn.init.constant_(self.gate.bias, -6.0)
            torch.nn.init.zeros_(self.residual.weight)
            torch.nn.init.zeros_(self.residual.bias)

        def forward(self, x: Any) -> tuple[Any | None, Any]:
            h = self.trunk(x)
            gate_logits = self.gate(h) if self.gate is not None else None
            return gate_logits, self.residual(h)

    def clipped(delta: Any) -> Any:
        norm = torch.linalg.norm(delta, dim=1, keepdim=True)
        scale = torch.clamp(float(alpha) / (norm + config.eps), max=1.0)
        return delta * scale

    def variant_delta(gate_logits: Any | None, predicted_residual: Any) -> tuple[Any, Any]:
        if variant == "openvla_oft_l1_proxy":
            gate = torch.ones((predicted_residual.shape[0], 1), dtype=predicted_residual.dtype, device=predicted_residual.device)
            return predicted_residual, gate
        clipped_residual = clipped(predicted_residual)
        if variant == "marc_full":
            assert gate_logits is not None
            gate = torch.sigmoid(gate_logits)
            return clipped_residual * gate, gate
        if variant == "marc_no_disagreement_gate_ablation":
            gate = torch.ones((predicted_residual.shape[0], 1), dtype=predicted_residual.dtype, device=predicted_residual.device)
            return clipped_residual, gate
        gate = torch.full((predicted_residual.shape[0], 1), float(static_beta), dtype=predicted_residual.dtype, device=predicted_residual.device)
        return clipped_residual * gate, gate

    def loss_terms(model: Any, x: Any, labels: Any, base_actions: Any, target_actions: Any) -> dict[str, Any]:
        gate_logits, predicted_residual = model(x)
        anchor = base_actions + predicted_residual
        anchor_loss = torch.nn.functional.smooth_l1_loss(anchor, target_actions)
        if variant == "marc_full":
            gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(gate_logits.reshape(-1), labels.reshape(-1))
        else:
            gate_loss = torch.zeros((), dtype=x.dtype, device=x.device)
        delta, _gate = variant_delta(gate_logits, predicted_residual)
        delta_loss = torch.mean(torch.sum(delta * delta, dim=1))
        clean_loss = torch.mean((1.0 - labels.reshape(-1)) * torch.sum(delta * delta, dim=1))
        if variant == "openvla_oft_l1_proxy":
            total = anchor_loss + 0.02 * delta_loss
        else:
            total = anchor_loss + gate_loss + 0.10 * delta_loss + 0.10 * clean_loss
        return {"total": total, "anchor": anchor_loss, "gate": gate_loss, "delta": delta_loss, "clean": clean_loss}

    torch.set_num_threads(1)
    torch.manual_seed(int(args.seed))
    use_gate = variant == "marc_full"
    model = MARCPolicyHead(train_features.shape[1], use_gate)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    x_train = torch.as_tensor(train_features.astype(np.float32), dtype=torch.float32)
    x_validation = torch.as_tensor(validation_features.astype(np.float32), dtype=torch.float32)
    y_train = torch.as_tensor(train_labels.astype(np.float32), dtype=torch.float32)
    y_validation = torch.as_tensor(validation_labels.astype(np.float32), dtype=torch.float32)
    base_train = torch.as_tensor(base_train_actions.astype(np.float32), dtype=torch.float32)
    base_validation = torch.as_tensor(base_validation_actions.astype(np.float32), dtype=torch.float32)
    target_train = torch.as_tensor(target_train_actions.astype(np.float32), dtype=torch.float32)
    target_validation = torch.as_tensor(target_validation_actions.astype(np.float32), dtype=torch.float32)

    with torch.no_grad():
        initial_terms = loss_terms(model, x_train, y_train, base_train, target_train)
        init_logits, init_residual = model(x_validation[:128])
        init_delta, _init_gate = variant_delta(init_logits, init_residual)
        initial_delta_p95 = float(np.percentile(torch.linalg.norm(init_delta, dim=1).cpu().numpy(), 95))

    first_grad_norms: dict[str, float] | None = None
    loss_curve = []
    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, x_train, y_train, base_train, target_train)
        terms["total"].backward()
        if epoch == 0:
            gate_params = list(model.gate.parameters()) if model.gate is not None else []
            first_grad_norms = {
                "trunk": _grad_norm(list(model.trunk.parameters()) if hasattr(model.trunk, "parameters") else [], torch),
                "gate": _grad_norm(gate_params, torch) if gate_params else 0.0,
                "anchor_residual": _grad_norm(list(model.residual.parameters()), torch),
            }
        optimizer.step()
        if epoch in {0, int(args.epochs) - 1}:
            loss_curve.append({"epoch": epoch + 1, **{key: float(value.detach().item()) for key, value in terms.items()}})

    with torch.no_grad():
        final_train_terms = loss_terms(model, x_train, y_train, base_train, target_train)
        validation_terms = loss_terms(model, x_validation, y_validation, base_validation, target_validation)
        gate_logits, predicted_residual = model(x_validation)
        delta, gate = variant_delta(gate_logits, predicted_residual)
        proposed_actions = base_validation + delta

    delta_np = delta.cpu().numpy()
    gate_np = gate.cpu().numpy().reshape(-1)
    proposed_np = proposed_actions.cpu().numpy()
    predictions = gate_np >= 0.5
    validation_labels_bool = validation_labels.astype(bool)
    if variant == "marc_full":
        gate_accuracy = float(np.mean(predictions == validation_labels_bool))
        gate_majority = float(max(np.mean(validation_labels_bool), 1.0 - np.mean(validation_labels_bool)))
        gate_margin = gate_accuracy - gate_majority
    else:
        gate_accuracy = None
        gate_majority = None
        gate_margin = None

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    policy_config = {
        "method": "MARC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "variant": variant,
        "config_id": config_id,
        "gate_architecture": architecture,
        "correction_alpha": float(alpha),
        "static_beta": float(static_beta),
        "feature_count": int(train_features.shape[1]),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
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
    reloaded = MARCPolicyHead(train_features.shape[1], use_gate)
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(loaded["model_state_dict"])
    with torch.no_grad():
        old_gate, old_residual = model(x_validation[:32])
        new_gate, new_residual = reloaded(x_validation[:32])
        gate_diff = 0.0 if old_gate is None else float(torch.max(torch.abs(old_gate - new_gate)).item())
        residual_diff = float(torch.max(torch.abs(old_residual - new_residual)).item())
    sha_manifest = {
        "model.pt": _sha256_file(model_path),
        "policy_config.json": _sha256_file(checkpoint_path / "policy_config.json"),
        "training_manifest.json": _sha256_file(checkpoint_path / "training_manifest.json"),
    }
    _write_json(checkpoint_path / "sha256_manifest.json", sha_manifest)

    grad = first_grad_norms or {}
    hard_stop_reasons = []
    reload_diff = max(gate_diff, residual_diff)
    if reload_diff > 1e-6:
        hard_stop_reasons.append("disk reload output mismatch")
    if float(grad.get("anchor_residual", 0.0)) <= 0.0:
        hard_stop_reasons.append("anchor residual gradient is zero")
    if variant == "marc_full" and float(grad.get("gate", 0.0)) <= 0.0:
        hard_stop_reasons.append("gate gradient is zero")
    metrics = _delta_metrics(delta_np, proposed_np, target_validation_actions, validation_labels_bool)
    if metrics["action_validity"] < 1.0:
        hard_stop_reasons.append("invalid validation action")
    if initial_delta_p95 > config.init_delta_p95_max:
        hard_stop_reasons.append("initial action delta is not base-passthrough")
    if variant == "marc_full":
        if gate_margin is not None and gate_margin < config.min_gate_probe_accuracy_margin:
            hard_stop_reasons.append(f"validation gate accuracy margin below minimum: {gate_margin:.6f}")
        activation = float(np.mean(predictions))
        if activation < config.min_positive_fraction or activation > config.max_positive_fraction:
            hard_stop_reasons.append(f"validation gate activation collapsed: {activation:.6f}")
    else:
        activation = float(np.mean(predictions))

    result = {
        "variant": variant,
        "final_decision": "MARC_POLICY_CHECKPOINT_VERIFIED" if not hard_stop_reasons else "MARC_POLICY_CHECKPOINT_BLOCKED",
        "checkpoint_path": str(checkpoint_path),
        "required_files": ["model.pt", "policy_config.json", "training_manifest.json", "sha256_manifest.json"],
        "sha256_manifest": sha_manifest,
        "disk_reload": True,
        "checkpoint_reload_max_abs_diff": reload_diff,
        "initial_delta_p95": initial_delta_p95,
        "first_gradient_norms": grad,
        "loss_initial": {key: float(value.detach().item()) for key, value in initial_terms.items()},
        "loss_final_train": {key: float(value.detach().item()) for key, value in final_train_terms.items()},
        "loss_validation": {key: float(value.detach().item()) for key, value in validation_terms.items()},
        "loss_curve": loss_curve,
        "gate_metrics": {
            "accuracy": gate_accuracy,
            "majority_accuracy": gate_majority,
            "accuracy_margin": gate_margin,
            "predicted_positive_fraction": float(np.mean(predictions)),
            "mean_probability": float(np.mean(gate_np)),
        },
        "validation": metrics,
        "hard_stop_reasons": hard_stop_reasons,
    }
    _write_json(checkpoint_path / "training_result.json", result)
    return result, proposed_np


def train_policy_identities(
    stable_artifact: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    *,
    train_args: MARCPolicyTrainArgs | None = None,
) -> dict[str, Any]:
    args = train_args or MARCPolicyTrainArgs()
    config_id = str(selected_config["config_id"])
    architecture = str(selected_config["gate_architecture"])
    alpha = float(selected_config["correction_alpha"])
    cfg = MARCConfig()
    records, thresholds = compute_disagreement_labels(build_marc_records(stable_artifact.get("records") or []), cfg)
    train = [record for record in records if str(record["split"]) in set(cfg.train_splits)]
    validation = [record for record in records if str(record["split"]) in set(cfg.validation_splits)]
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    train_features_raw = _feature_matrix(train, task_count)
    validation_features_raw = _feature_matrix(validation, task_count)
    train_features, validation_features, feature_mean, feature_scale = _standardize_with_stats(train_features_raw, validation_features_raw)
    train_labels = np.asarray([record["disagreement_label"] for record in train], dtype=bool)
    validation_labels = np.asarray([record["disagreement_label"] for record in validation], dtype=bool)
    train_residuals = np.asarray([record["residual"] for record in train], dtype=np.float64)
    validation_residuals = np.asarray([record["residual"] for record in validation], dtype=np.float64)
    base_train_actions = np.asarray([record["base_action"] for record in train], dtype=np.float64)
    base_validation_actions = np.asarray([record["base_action"] for record in validation], dtype=np.float64)
    target_train_actions = np.asarray([record["target_action"] for record in train], dtype=np.float64)
    target_validation_actions = np.asarray([record["target_action"] for record in validation], dtype=np.float64)
    static_beta = float(np.mean(train_labels))

    results = []
    validation_actions: dict[str, np.ndarray] = {}
    for variant in TRAINABLE_VARIANTS:
        result, actions = _train_variant(
            variant=variant,
            config_id=config_id,
            architecture=architecture,
            alpha=alpha,
            static_beta=static_beta,
            train_features=train_features,
            validation_features=validation_features,
            train_labels=train_labels,
            validation_labels=validation_labels,
            train_residuals=train_residuals,
            validation_residuals=validation_residuals,
            base_train_actions=base_train_actions,
            base_validation_actions=base_validation_actions,
            target_train_actions=target_train_actions,
            target_validation_actions=target_validation_actions,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            checkpoint_path=_policy_path(args.checkpoint_output_root, config_id, variant, args.seed),
            args=args,
            config=cfg,
        )
        results.append(result)
        validation_actions[variant] = actions

    full_actions = validation_actions["marc_full"]
    distinction = {
        f"marc_full_vs_{variant}_mean_l2": float(np.mean(np.linalg.norm(full_actions - actions, axis=1)))
        for variant, actions in validation_actions.items()
        if variant != "marc_full"
    }
    verified = all(result["final_decision"] == "MARC_POLICY_CHECKPOINT_VERIFIED" for result in results)
    if any(value <= 1e-6 for value in distinction.values()):
        verified = False

    return {
        "schema_version": 1,
        "date": DATE_KST,
        "method": "MARC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "config_id": config_id,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "checkpoint_root": str(Path(args.checkpoint_output_root) / config_id),
        "selected_validation_checkpoint": str(selected_config.get("checkpoint_path")),
        "policy_identities": list(ALL_POLICY_IDENTITIES),
        "trainable_variants": list(TRAINABLE_VARIANTS),
        "nontrainable_variants": ["frozen_smolvla"],
        "closed_loop_experiment_happened": False,
        "confirmatory_test_identities_used": False,
        "training_happened": True,
        "stage_a_allowed": bool(verified),
        "disagreement_thresholds": thresholds,
        "static_beta": static_beta,
        "train_records": len(train),
        "validation_records": len(validation),
        "variant_results": results,
        "distinction": distinction,
        "static_mixture_remains_live_reviewer_killer": distinction["marc_full_vs_static_l1_mixture_baseline_mean_l2"] < 0.003,
        "final_decision": "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY" if verified else "MARC_POLICY_IDENTITIES_BLOCKED",
        "next_step": (
            "Freeze the MARC Stage A matched manifest before any rollout."
            if verified
            else "Do not roll out; inspect MARC policy identity training failures."
        ),
    }


def _write_md(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# MARC-VLA Policy Identity Training",
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
        f"- static mixture remains live reviewer-killer: `{report.get('static_mixture_remains_live_reviewer_killer')}`",
        "",
        "Variants:",
        "",
        "| variant | decision | reload | delta p95 | clean p95 | validity | target L2 | checkpoint |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report.get("variant_results") or []:
        validation = result.get("validation") or {}
        lines.append(
            f"| `{result.get('variant')}` | `{result.get('final_decision')}` | `{result.get('disk_reload')}` | "
            f"{validation.get('delta_l2_p95')} | {validation.get('clean_delta_l2_p95')} | "
            f"{validation.get('action_validity')} | {validation.get('action_l2_to_target')} | `{result.get('checkpoint_path')}` |"
        )
    lines.extend(
        [
            "",
            "Full-policy distinctions:",
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
    parser.add_argument("--selected-config", default="reports/marc_vla/selected_config.json")
    parser.add_argument("--checkpoint-output-root", default="runs/marc_vla_checkpoints")
    parser.add_argument("--report-json", default="reports/marc_vla/policy_checkpoint_manifest.json")
    parser.add_argument("--report-md", default="reports/marc_vla/policy_checkpoint_manifest.md")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args(argv)
    report = train_policy_identities(
        _read_json(Path(args.stable_artifact)),
        _read_json(Path(args.selected_config)),
        train_args=MARCPolicyTrainArgs(
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
