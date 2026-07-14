"""RAC-VLA development-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TASK_KEYS = ("libero_spatial/task_4", "libero_10/task_4")
PROPOSAL_HASH = "71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F"
PERTURBATION_NAMES = (
    "identity",
    "x_attenuate",
    "y_attenuate",
    "xy_swap",
    "gripper_bias",
)
VARIANTS = (
    "base_smolvla_shifted",
    "reflective_history_proxy",
    "rac_full",
    "rac_no_consequence_ablation",
    "online_diagonal_inverse_gain",
)
FORBIDDEN_INFERENCE_KEYS = {
    "object_state",
    "object_pose",
    "reward",
    "terminal_success",
    "success",
    "future_state",
    "future_action",
    "identity",
}


@dataclass(frozen=True)
class RACConfig:
    train_identities: tuple[int, ...] = tuple(range(20260901, 20260911))
    validation_identities: tuple[int, ...] = tuple(range(20260911, 20260917))
    forbidden_development_identities: tuple[int, ...] = tuple(range(20260917, 20261201))
    history_horizon: int = 2
    min_consequence_pairs: int = 5000
    min_task_pairs: int = 1000
    min_label_fraction: float = 0.10
    max_label_fraction: float = 0.90
    min_accuracy_margin: float = 0.05
    ridge_lambda: float = 1e-2
    gate_confidence: float = 0.35
    min_gate_fraction: float = 0.02
    max_gate_fraction: float = 0.98
    residual_alpha: float = 0.10
    residual_delta_max: float = 0.20
    translation_delta_max: float = 0.10
    rotation_delta_max: float = 0.10
    gripper_delta_max: float = 0.05
    min_scale: float = 1e-6


def _as_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} expected {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def task_one_hot(task_key: str) -> np.ndarray:
    if task_key not in TASK_KEYS:
        raise ValueError(f"unknown RAC task key: {task_key}")
    out = np.zeros(len(TASK_KEYS), dtype=np.float64)
    out[TASK_KEYS.index(task_key)] = 1.0
    return out


def validate_inference_fields(fields: Mapping[str, Any]) -> None:
    leaked = sorted(str(key) for key in fields if str(key) in FORBIDDEN_INFERENCE_KEYS)
    if leaked:
        raise ValueError(f"privileged RAC inference fields: {leaked}")


def perturb_action(action: Sequence[float], label: int) -> np.ndarray:
    """Return the command that would yield ``action`` under perturbation ``label``."""
    out = _as_vector("action", action, 7).copy()
    name = PERTURBATION_NAMES[int(label)]
    if name == "identity":
        return out
    if name == "x_attenuate":
        out[0] = out[0] / 0.65
    elif name == "y_attenuate":
        out[1] = out[1] / 0.65
    elif name == "xy_swap":
        out[0], out[1] = out[1], out[0]
    elif name == "gripper_bias":
        out[6] = np.clip(out[6] - 0.35, -1.0, 1.0)
    else:  # pragma: no cover - guarded by tuple labels.
        raise ValueError(f"unknown perturbation: {name}")
    return out.astype(np.float64)


def calibration_residual(base_action: Sequence[float], label: int, config: RACConfig | None = None) -> np.ndarray:
    cfg = config or RACConfig()
    action = _as_vector("base_action", base_action, 7)
    residual = perturb_action(action, label) - action
    residual[:3] = _clip_l2(residual[:3], cfg.translation_delta_max)
    residual[3:6] = _clip_l2(residual[3:6], cfg.rotation_delta_max)
    residual[6] = float(np.clip(residual[6], -cfg.gripper_delta_max, cfg.gripper_delta_max))
    return _clip_l2(residual, cfg.residual_delta_max)


def inverse_residual(base_action: Sequence[float], label: int, config: RACConfig | None = None) -> np.ndarray:
    return calibration_residual(base_action, label, config)


def apply_environment_shift(command_action: Sequence[float], label: int = 1) -> np.ndarray:
    out = _as_vector("command_action", command_action, 7).copy()
    name = PERTURBATION_NAMES[int(label)]
    if name == "identity":
        return out
    if name == "x_attenuate":
        out[0] *= 0.65
    elif name == "y_attenuate":
        out[1] *= 0.65
    elif name == "xy_swap":
        out[0], out[1] = out[1], out[0]
    elif name == "gripper_bias":
        out[6] = np.clip(out[6] + 0.35, -1.0, 1.0)
    else:  # pragma: no cover
        raise ValueError(f"unknown environment shift: {name}")
    return out.astype(np.float64)


def _clip_l2(values: np.ndarray, max_norm: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= float(max_norm) or norm <= 1e-12:
        return values.copy()
    return values * (float(max_norm) / norm)


def _transition_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(record["task_key"]), int(record["identity"]), int(record["step"]))


def build_consequence_pairs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    relevant = [record for record in records if str(record.get("task_key")) in TASK_KEYS]
    relevant = sorted(relevant, key=lambda row: (str(row["task_key"]), int(row["identity"]), int(row["step"])))
    by_key = {_transition_key(record): record for record in relevant}
    pairs: list[dict[str, Any]] = []
    for record in relevant:
        task_key, identity, step = _transition_key(record)
        next_record = by_key.get((task_key, identity, step + 1))
        if next_record is None:
            continue
        state = _as_vector("state", record["state"], 8)
        next_state = _as_vector("next_state", next_record["state"], 8)
        pairs.append(
            {
                "task_key": task_key,
                "identity": identity,
                "step": step,
                "state": state,
                "action": _as_vector("action", record["action"], 7),
                "previous_action": _as_vector("previous_action", record["previous_action"], 7),
                "chunk_index_fraction": float(record["chunk_index_fraction"]),
                "delta_state": next_state - state,
            }
        )
    return pairs


def _single_feature(pair: Mapping[str, Any], perturbed_action: np.ndarray, kind: str) -> np.ndarray:
    task = str(pair["task_key"])
    state = _as_vector("state", pair["state"], 8)
    previous_action = _as_vector("previous_action", pair["previous_action"], 7)
    delta = _as_vector("delta_state", pair["delta_state"], 8)
    common = np.concatenate(
        [
            perturbed_action,
            previous_action,
            np.asarray([float(pair["chunk_index_fraction"])], dtype=np.float64),
            task_one_hot(task),
        ]
    )
    if kind == "action":
        return common.astype(np.float64)
    if kind == "noconsequence":
        return np.concatenate([state, common]).astype(np.float64)
    if kind != "full":
        raise ValueError(f"unknown RAC feature kind: {kind}")
    outer = np.outer(perturbed_action, delta).reshape(-1)
    effect_terms = np.asarray(
        [
            perturbed_action[0] * delta[0],
            perturbed_action[1] * delta[1],
            perturbed_action[2] * delta[2],
            np.linalg.norm(perturbed_action[:3]),
            np.linalg.norm(delta[:3]),
            np.dot(perturbed_action[:3], delta[:3]),
            perturbed_action[6] * delta[-1],
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            state,
            common,
            delta,
            outer,
            effect_terms,
            np.abs(perturbed_action),
            np.abs(delta),
        ]
    ).astype(np.float64)


def history_feature(history: Sequence[Mapping[str, Any]], kind: str, config: RACConfig | None = None) -> np.ndarray | None:
    cfg = config or RACConfig()
    if len(history) < cfg.history_horizon:
        return None
    values = []
    for pair in list(history)[-cfg.history_horizon :]:
        values.append(_single_feature(pair, _as_vector("action", pair["action"], 7), kind))
    return np.mean(np.asarray(values, dtype=np.float64), axis=0)


def build_labeled_examples(
    pairs: Sequence[Mapping[str, Any]],
    config: RACConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RACConfig()
    by_key = {
        (str(pair["task_key"]), int(pair["identity"]), int(pair["step"])): pair
        for pair in pairs
    }
    features = {"action": [], "noconsequence": [], "full": []}
    labels: list[int] = []
    identities: list[int] = []
    tasks: list[str] = []
    keys: list[tuple[str, int, int, int]] = []
    perturbed_actions: list[np.ndarray] = []
    base_actions: list[np.ndarray] = []
    for pair in pairs:
        for label, _name in enumerate(PERTURBATION_NAMES):
            history_features: dict[str, list[np.ndarray]] = {"action": [], "noconsequence": [], "full": []}
            ok = True
            for offset in range(cfg.history_horizon):
                hist = by_key.get((str(pair["task_key"]), int(pair["identity"]), int(pair["step"]) - offset))
                if hist is None:
                    ok = False
                    break
                perturbed = perturb_action(hist["action"], label)
                for kind in history_features:
                    history_features[kind].append(_single_feature(hist, perturbed, kind))
            if not ok:
                continue
            for kind, values in history_features.items():
                features[kind].append(np.mean(np.asarray(values, dtype=np.float64), axis=0))
            labels.append(label)
            identities.append(int(pair["identity"]))
            tasks.append(str(pair["task_key"]))
            keys.append((str(pair["task_key"]), int(pair["identity"]), int(pair["step"]), label))
            perturbed_actions.append(perturb_action(pair["action"], label))
            base_actions.append(_as_vector("action", pair["action"], 7))
    return {
        "features": {key: np.asarray(value, dtype=np.float64) for key, value in features.items()},
        "labels": np.asarray(labels, dtype=np.int64),
        "identities": np.asarray(identities, dtype=np.int64),
        "tasks": tasks,
        "keys": keys,
        "perturbed_actions": np.asarray(perturbed_actions, dtype=np.float64),
        "base_actions": np.asarray(base_actions, dtype=np.float64),
    }


def duplicate_example_count(keys: Sequence[tuple[str, int, int, int]]) -> int:
    seen: set[tuple[str, int, int, int]] = set()
    duplicates = 0
    for key in keys:
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _standardization(features: np.ndarray, train_mask: np.ndarray, config: RACConfig) -> tuple[np.ndarray, np.ndarray]:
    train = features[train_mask]
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale < config.min_scale, 1.0, scale)
    return mean.astype(np.float64), scale.astype(np.float64)


def _standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (features - mean.reshape(1, -1)) / scale.reshape(1, -1)


def fit_ridge_classifier(features: np.ndarray, labels: np.ndarray, train_mask: np.ndarray, config: RACConfig) -> dict[str, np.ndarray]:
    mean, scale = _standardization(features, train_mask, config)
    train_x = _standardize(features[train_mask], mean, scale)
    x = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=np.float64)], axis=1)
    y = np.eye(len(PERTURBATION_NAMES), dtype=np.float64)[labels[train_mask]]
    reg = np.eye(x.shape[1], dtype=np.float64) * float(config.ridge_lambda)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights.astype(np.float64)}


def predict_classifier(model: Mapping[str, np.ndarray], features: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    x = _standardize(features, mean, scale)
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    logits = x @ weights
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return {
        "logits": logits,
        "probabilities": probs,
        "prediction": np.argmax(probs, axis=1).astype(np.int64),
        "confidence": np.max(probs, axis=1).astype(np.float64),
    }


def _accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float(np.mean(prediction == labels))


def _classify_hard_stop(reasons: Sequence[str]) -> str:
    joined = " ".join(reasons).lower()
    if "forbidden" in joined or "duplicate" in joined or "below minimum" in joined or "collapsed" in joined or "missing" in joined:
        return "DATA_FAILURE"
    if "headroom" in joined:
        return "NO_HEADROOM"
    if "nonfinite" in joined or "validity" in joined:
        return "IMPLEMENTATION_FAILURE"
    return "DESIGN_FAILURE"


def audit_rac_records(records: Sequence[Mapping[str, Any]], config: RACConfig | None = None) -> dict[str, Any]:
    cfg = config or RACConfig()
    pairs = build_consequence_pairs(records)
    examples = build_labeled_examples(pairs, cfg)
    hard_stop_reasons: list[str] = []

    identities = np.asarray(examples["identities"], dtype=np.int64)
    labels = np.asarray(examples["labels"], dtype=np.int64)
    train_ids = {int(value) for value in cfg.train_identities}
    validation_ids = {int(value) for value in cfg.validation_identities}
    train_mask = np.asarray([int(value) in train_ids for value in identities], dtype=bool)
    validation_mask = np.asarray([int(value) in validation_ids for value in identities], dtype=bool)
    forbidden_ids = {int(value) for value in cfg.forbidden_development_identities}
    forbidden_present = sorted({int(value) for value in identities if int(value) in forbidden_ids})
    if forbidden_present:
        hard_stop_reasons.append(f"forbidden development identities present: {forbidden_present[:5]}")

    pair_task_counts = {task: sum(1 for pair in pairs if str(pair["task_key"]) == task) for task in TASK_KEYS}
    if len(pairs) < cfg.min_consequence_pairs:
        hard_stop_reasons.append(f"consequence pairs below minimum: {len(pairs)} < {cfg.min_consequence_pairs}")
    for task, count in pair_task_counts.items():
        if count < cfg.min_task_pairs:
            hard_stop_reasons.append(f"{task} consequence pairs below minimum: {count} < {cfg.min_task_pairs}")

    duplicates = duplicate_example_count(examples["keys"])
    if duplicates:
        hard_stop_reasons.append(f"duplicate perturbation keys: {duplicates}")
    if not np.any(train_mask) or not np.any(validation_mask):
        hard_stop_reasons.append("missing train or validation RAC examples")

    label_counts = {PERTURBATION_NAMES[i]: int(np.sum(labels == i)) for i in range(len(PERTURBATION_NAMES))}
    label_fractions = {name: count / max(int(labels.size), 1) for name, count in label_counts.items()}
    for name, fraction in label_fractions.items():
        if fraction < cfg.min_label_fraction or fraction > cfg.max_label_fraction:
            hard_stop_reasons.append(f"perturbation label collapsed for {name}: {fraction:.6f}")

    report = _base_report(cfg, pairs, examples, duplicates, pair_task_counts, label_counts, hard_stop_reasons)
    if hard_stop_reasons:
        report["final_decision"] = "AUDIT_STOP_" + _classify_hard_stop(hard_stop_reasons)
        return report

    model_reports: dict[str, Any] = {}
    for kind, matrix in examples["features"].items():
        model = fit_ridge_classifier(matrix, labels, train_mask, cfg)
        pred = predict_classifier(model, matrix[validation_mask])
        val_labels = labels[validation_mask]
        model_reports[kind] = {
            "feature_dim": int(matrix.shape[1]),
            "validation_accuracy": _accuracy(pred["prediction"], val_labels),
            "confidence_mean": float(np.mean(pred["confidence"])) if pred["confidence"].size else 0.0,
            "model": model,
            "prediction": pred,
        }

    full_acc = float(model_reports["full"]["validation_accuracy"])
    action_acc = float(model_reports["action"]["validation_accuracy"])
    noconseq_acc = float(model_reports["noconsequence"]["validation_accuracy"])
    val_labels = labels[validation_mask]
    shifted_mask = val_labels != 0
    identity_mask = val_labels == 0
    for kind, payload in model_reports.items():
        prediction = np.asarray(payload["prediction"]["prediction"], dtype=np.int64)
        payload["shifted_validation_accuracy"] = _accuracy(prediction[shifted_mask], val_labels[shifted_mask])
        payload["identity_validation_accuracy"] = _accuracy(prediction[identity_mask], val_labels[identity_mask])
    best_baseline = max(action_acc, noconseq_acc)
    margin = full_acc - best_baseline
    if margin < cfg.min_accuracy_margin:
        hard_stop_reasons.append(f"full consequence classifier margin below minimum: {margin:.6f}")

    full_pred = model_reports["full"]["prediction"]
    val_actions = np.asarray(examples["perturbed_actions"], dtype=np.float64)[validation_mask]
    val_base_actions = np.asarray(examples["base_actions"], dtype=np.float64)[validation_mask]
    predictions = np.asarray(full_pred["prediction"], dtype=np.int64)
    confidence = np.asarray(full_pred["confidence"], dtype=np.float64)
    gate = (confidence >= cfg.gate_confidence) & (predictions != 0)
    gate_fraction = float(np.mean(gate)) if gate.size else 0.0
    if gate_fraction < cfg.min_gate_fraction or gate_fraction > cfg.max_gate_fraction:
        hard_stop_reasons.append(f"gate positive fraction outside bounds: {gate_fraction:.6f}")

    clean_mask = val_labels == 0
    clean_deltas = []
    shifted_deltas = []
    for active, pred_label, base_action, true_label in zip(gate, predictions, val_base_actions, val_labels):
        residual = calibration_residual(base_action, int(pred_label), cfg) * cfg.residual_alpha if active else np.zeros(7)
        delta_norm = float(np.linalg.norm(residual))
        if int(true_label) == 0:
            clean_deltas.append(delta_norm)
        else:
            shifted_deltas.append(delta_norm)
    clean_p95 = float(np.percentile(clean_deltas, 95)) if clean_deltas else 0.0
    shifted_p95 = float(np.percentile(shifted_deltas, 95)) if shifted_deltas else 0.0
    if clean_p95 > cfg.residual_delta_max + 1e-12:
        hard_stop_reasons.append(f"clean p95 action delta exceeds cap: {clean_p95:.6f}")
    action_validity = float(np.mean(np.all(np.isfinite(val_actions), axis=1))) if val_actions.size else 0.0
    if action_validity < 1.0:
        hard_stop_reasons.append(f"validation action validity below 1.0: {action_validity:.6f}")

    serializable_models = {
        kind: {
            "feature_dim": payload["feature_dim"],
            "validation_accuracy": payload["validation_accuracy"],
            "shifted_validation_accuracy": payload["shifted_validation_accuracy"],
            "identity_validation_accuracy": payload["identity_validation_accuracy"],
            "confidence_mean": payload["confidence_mean"],
        }
        for kind, payload in model_reports.items()
    }
    full_shifted_acc = float(model_reports["full"]["shifted_validation_accuracy"])
    best_shifted_baseline_acc = max(
        float(model_reports["action"]["shifted_validation_accuracy"]),
        float(model_reports["noconsequence"]["shifted_validation_accuracy"]),
    )
    report.update(
        {
            "model_reports": serializable_models,
            "full_validation_accuracy": full_acc,
            "action_only_validation_accuracy": action_acc,
            "no_consequence_validation_accuracy": noconseq_acc,
            "full_vs_best_baseline_accuracy_margin": margin,
            "full_shifted_validation_accuracy": full_shifted_acc,
            "best_baseline_shifted_validation_accuracy": best_shifted_baseline_acc,
            "full_vs_best_baseline_shifted_accuracy_margin": full_shifted_acc - best_shifted_baseline_acc,
            "full_identity_validation_accuracy": float(model_reports["full"]["identity_validation_accuracy"]),
            "gate_positive_fraction": gate_fraction,
            "clean_identity_validation_count": int(np.sum(clean_mask)),
            "clean_gate_positive_fraction": float(np.mean(gate[clean_mask])) if np.any(clean_mask) else 0.0,
            "clean_action_delta_p95": clean_p95,
            "shifted_action_delta_p95": shifted_p95,
            "validation_action_validity": action_validity,
        }
    )
    report["hard_stop_reasons"] = hard_stop_reasons
    report["final_decision"] = "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" if not hard_stop_reasons else "AUDIT_STOP_" + _classify_hard_stop(hard_stop_reasons)
    report["next_step"] = "Run bounded six-config validation search." if not hard_stop_reasons else "Do not roll out RAC; archive the hard stop and continue."
    return report


def validation_score(report: Mapping[str, Any]) -> dict[str, float]:
    margin = max(0.0, float(report.get("full_vs_best_baseline_accuracy_margin") or 0.0))
    shifted_margin = max(0.0, float(report.get("full_vs_best_baseline_shifted_accuracy_margin") or 0.0))
    clean_delta = float(report.get("clean_action_delta_p95") or 0.0)
    delta_cap = float(((report.get("config") or {}).get("residual_delta_max")) or RACConfig().residual_delta_max)
    clean_retention = max(0.0, 1.0 - min(clean_delta / max(delta_cap, 1e-12), 1.0))
    gate = float(report.get("gate_positive_fraction") or 0.0)
    mechanism_activation = max(0.0, min(gate / 0.10, 1.0) * min((0.50 - gate) / 0.40, 1.0)) if gate < 0.50 else 0.0
    action_validity = float(report.get("validation_action_validity") or 0.0)
    total = (
        0.35 * margin
        + 0.25 * shifted_margin
        + 0.20 * clean_retention
        + 0.15 * mechanism_activation
        + 0.05 * action_validity
    )
    return {
        "full_vs_best_baseline_margin": margin,
        "shifted_proxy_gain": shifted_margin,
        "clean_retention": clean_retention,
        "mechanism_activation": mechanism_activation,
        "action_validity": action_validity,
        "total": float(total),
    }


def run_validation_search(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tried: list[dict[str, Any]] = []
    for horizon in (2, 4):
        for alpha in (0.05, 0.10, 0.20):
            cfg = RACConfig(history_horizon=horizon, residual_alpha=alpha)
            report = audit_rac_records(records, cfg)
            score = validation_score(report) if report["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" else {"total": -1.0}
            tried.append(
                {
                    "config_id": f"rac_h{horizon}_a{alpha:.2f}",
                    "history_horizon": horizon,
                    "residual_alpha": alpha,
                    "final_decision": report["final_decision"],
                    "full_validation_accuracy": report.get("full_validation_accuracy"),
                    "action_only_validation_accuracy": report.get("action_only_validation_accuracy"),
                    "no_consequence_validation_accuracy": report.get("no_consequence_validation_accuracy"),
                    "full_vs_best_baseline_accuracy_margin": report.get("full_vs_best_baseline_accuracy_margin"),
                    "full_vs_best_baseline_shifted_accuracy_margin": report.get("full_vs_best_baseline_shifted_accuracy_margin"),
                    "gate_positive_fraction": report.get("gate_positive_fraction"),
                    "clean_action_delta_p95": report.get("clean_action_delta_p95"),
                    "shifted_action_delta_p95": report.get("shifted_action_delta_p95"),
                    "validation_action_validity": report.get("validation_action_validity"),
                    "score_terms": score,
                    "hard_stop_reasons": report.get("hard_stop_reasons", []),
                }
            )
    selected = max(tried, key=lambda item: float((item.get("score_terms") or {}).get("total", -1.0)))
    decision = "VALIDATION_SEARCH_SELECT_CONFIG" if float((selected.get("score_terms") or {}).get("total", -1.0)) >= 0.0 else "VALIDATION_SEARCH_NO_PASSING_CONFIG"
    return {
        "method": "RAC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "search_budget": "6 configs: H in {2, 4}, alpha in {0.05, 0.10, 0.20}",
        "tried_config_count": len(tried),
        "tried_configs": tried,
        "selected_config": selected,
        "final_decision": decision,
        "next_step": "Freeze selected config and implement Stage A runner." if decision == "VALIDATION_SEARCH_SELECT_CONFIG" else "Archive validation failure and continue.",
    }


def fit_rac_runtime(records: Sequence[Mapping[str, Any]], selected_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    horizon = int((selected_config or {}).get("history_horizon", 4))
    alpha = float((selected_config or {}).get("residual_alpha", 0.05))
    cfg = RACConfig(history_horizon=horizon, residual_alpha=alpha)
    pairs = build_consequence_pairs(records)
    examples = build_labeled_examples(pairs, cfg)
    identities = np.asarray(examples["identities"], dtype=np.int64)
    train_mask = np.asarray([int(value) in set(cfg.train_identities) for value in identities], dtype=bool)
    labels = np.asarray(examples["labels"], dtype=np.int64)
    models = {
        kind: fit_ridge_classifier(matrix, labels, train_mask, cfg)
        for kind, matrix in examples["features"].items()
    }
    nominal = _nominal_effect_ratios(pairs, cfg)
    return {
        "config": cfg,
        "models": models,
        "nominal_effect_ratios": nominal,
        "proposal_hash": PROPOSAL_HASH,
        "selected_config": dict(selected_config or {"history_horizon": horizon, "residual_alpha": alpha}),
    }


def _nominal_effect_ratios(pairs: Sequence[Mapping[str, Any]], config: RACConfig) -> dict[str, float]:
    train_ids = {int(value) for value in config.train_identities}
    selected = [pair for pair in pairs if int(pair["identity"]) in train_ids]
    ratios: dict[str, float] = {}
    for axis, name in ((0, "x"), (1, "y")):
        values = []
        for pair in selected:
            action = abs(float(_as_vector("action", pair["action"], 7)[axis]))
            delta = abs(float(_as_vector("delta_state", pair["delta_state"], 8)[axis]))
            if action > 1e-4:
                values.append(delta / action)
        ratios[name] = float(np.median(values)) if values else 0.0
    return ratios


def _predict_label(runtime: Mapping[str, Any], history: Sequence[Mapping[str, Any]], kind: str) -> tuple[int, float]:
    cfg = runtime["config"]
    feature = history_feature(history, kind, cfg)
    if feature is None:
        return 0, 0.0
    pred = predict_classifier(runtime["models"][kind], feature.reshape(1, -1))
    return int(pred["prediction"][0]), float(pred["confidence"][0])


def _reflective_proxy_label(runtime: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> tuple[int, float]:
    cfg = runtime["config"]
    if len(history) < cfg.history_horizon:
        return 0, 0.0
    recent = list(history)[-cfg.history_horizon :]
    ax = np.asarray([float(_as_vector("action", row["action"], 7)[0]) for row in recent], dtype=np.float64)
    ay = np.asarray([float(_as_vector("action", row["action"], 7)[1]) for row in recent], dtype=np.float64)
    dx = np.asarray([float(_as_vector("delta_state", row["delta_state"], 8)[0]) for row in recent], dtype=np.float64)
    dy = np.asarray([float(_as_vector("delta_state", row["delta_state"], 8)[1]) for row in recent], dtype=np.float64)
    score_x = float(np.mean(np.abs(dx)) / (np.mean(np.abs(ax)) + 1e-6))
    score_y = float(np.mean(np.abs(dy)) / (np.mean(np.abs(ay)) + 1e-6))
    nominal = runtime.get("nominal_effect_ratios") or {}
    x_nom = float(nominal.get("x") or 0.0)
    y_nom = float(nominal.get("y") or 0.0)
    cross = abs(float(np.dot(ax, dy))) + abs(float(np.dot(ay, dx)))
    direct = abs(float(np.dot(ax, dx))) + abs(float(np.dot(ay, dy))) + 1e-6
    if cross > 1.25 * direct:
        return 3, min(1.0, cross / direct / 2.0)
    if x_nom > 0.0 and score_x < 0.75 * x_nom:
        return 1, min(1.0, (0.75 * x_nom - score_x) / max(0.75 * x_nom, 1e-6) + 0.35)
    if y_nom > 0.0 and score_y < 0.75 * y_nom:
        return 2, min(1.0, (0.75 * y_nom - score_y) / max(0.75 * y_nom, 1e-6) + 0.35)
    return 0, 0.0


def apply_rac_action(
    runtime: Mapping[str, Any],
    *,
    variant: str,
    state: Sequence[float],
    action: Sequence[float],
    previous_action: Sequence[float],
    chunk_index_fraction: float,
    task_key: str,
    history: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    del state, previous_action, chunk_index_fraction, task_key
    cfg: RACConfig = runtime["config"]
    base = _as_vector("action", action, 7)
    label = 0
    confidence = 0.0
    if variant == "base_smolvla_shifted":
        command = base
    elif variant == "rac_full":
        label, confidence = _predict_label(runtime, history, "full")
        command = base + calibration_residual(base, label, cfg) * cfg.residual_alpha if confidence >= cfg.gate_confidence and label != 0 else base
    elif variant == "rac_no_consequence_ablation":
        label, confidence = _predict_label(runtime, history, "noconsequence")
        command = base + calibration_residual(base, label, cfg) * cfg.residual_alpha if confidence >= cfg.gate_confidence and label != 0 else base
    elif variant == "reflective_history_proxy":
        label, confidence = _reflective_proxy_label(runtime, history)
        command = base + calibration_residual(base, label, cfg) * cfg.residual_alpha if confidence >= cfg.gate_confidence and label != 0 else base
    elif variant == "online_diagonal_inverse_gain":
        label, confidence = _reflective_proxy_label(runtime, history)
        command = base + calibration_residual(base, label, cfg) * cfg.residual_alpha if label in {1, 2} and confidence >= 0.30 else base
    else:
        raise ValueError(f"unknown RAC variant: {variant}")
    command = np.asarray(command, dtype=np.float64).reshape(-1)
    delta = command - base
    return command, {
        "gate": float(label != 0 and confidence >= cfg.gate_confidence),
        "predicted_label": int(label),
        "predicted_name": PERTURBATION_NAMES[int(label)],
        "confidence": float(confidence),
        "action_delta_l2": float(np.linalg.norm(delta)),
    }


def _base_report(
    config: RACConfig,
    pairs: Sequence[Mapping[str, Any]],
    examples: Mapping[str, Any],
    duplicates: int,
    pair_task_counts: Mapping[str, int],
    label_counts: Mapping[str, int],
    hard_stop_reasons: Sequence[str],
) -> dict[str, Any]:
    identities = np.asarray(examples["identities"], dtype=np.int64)
    labels = np.asarray(examples["labels"], dtype=np.int64)
    return {
        "method": "RAC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "consequence_pairs": len(pairs),
        "labeled_examples": int(labels.size),
        "train_examples": int(np.sum(np.isin(identities, config.train_identities))),
        "validation_examples": int(np.sum(np.isin(identities, config.validation_identities))),
        "duplicate_perturbation_keys": int(duplicates),
        "task_consequence_counts": dict(pair_task_counts),
        "label_counts": dict(label_counts),
        "hard_stop_reasons": list(hard_stop_reasons),
        "config": {
            "train_identities": list(config.train_identities),
            "validation_identities": list(config.validation_identities),
            "forbidden_development_identity_min": min(config.forbidden_development_identities),
            "history_horizon": config.history_horizon,
            "min_consequence_pairs": config.min_consequence_pairs,
            "min_task_pairs": config.min_task_pairs,
            "min_accuracy_margin": config.min_accuracy_margin,
            "ridge_lambda": config.ridge_lambda,
            "gate_confidence": config.gate_confidence,
            "residual_alpha": config.residual_alpha,
            "residual_delta_max": config.residual_delta_max,
        },
        "final_decision": "AUDIT_STOP_DATA_FAILURE" if hard_stop_reasons else "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH",
        "next_step": "Do not roll out RAC; archive the hard stop and continue." if hard_stop_reasons else "Run bounded six-config validation search.",
    }


__all__ = [
    "FORBIDDEN_INFERENCE_KEYS",
    "PERTURBATION_NAMES",
    "PROPOSAL_HASH",
    "RACConfig",
    "TASK_KEYS",
    "VARIANTS",
    "audit_rac_records",
    "apply_environment_shift",
    "apply_rac_action",
    "build_consequence_pairs",
    "build_labeled_examples",
    "calibration_residual",
    "duplicate_example_count",
    "fit_ridge_classifier",
    "fit_rac_runtime",
    "history_feature",
    "inverse_residual",
    "perturb_action",
    "predict_classifier",
    "run_validation_search",
    "task_one_hot",
    "validate_inference_fields",
    "validation_score",
]
