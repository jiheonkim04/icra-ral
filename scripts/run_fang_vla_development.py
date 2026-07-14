"""Run FANG-VLA development-only audits and lightweight validation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.fang_vla import (  # noqa: E402
    compute_gate_targets,
    records_to_arrays,
    split_development_records,
    standardize_train_validation,
    audit_fang_records,
)


DATE_KST = "2026-07-14"
PROPOSAL_HASH = "6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B"
SEARCH_CONFIGS = [
    {"name": "fang_c01", "alpha": 0.10, "lambda_delta": 0.10, "lambda_gate_fit": 1.00, "lambda_gate_sparse": 0.01, "beta": 0.50},
    {"name": "fang_c02", "alpha": 0.20, "lambda_delta": 0.10, "lambda_gate_fit": 1.00, "lambda_gate_sparse": 0.01, "beta": 0.50},
    {"name": "fang_c03", "alpha": 0.35, "lambda_delta": 0.10, "lambda_gate_fit": 1.00, "lambda_gate_sparse": 0.01, "beta": 0.50},
    {"name": "fang_c04", "alpha": 0.10, "lambda_delta": 0.30, "lambda_gate_fit": 1.00, "lambda_gate_sparse": 0.01, "beta": 0.50},
    {"name": "fang_c05", "alpha": 0.20, "lambda_delta": 0.30, "lambda_gate_fit": 1.00, "lambda_gate_sparse": 0.01, "beta": 0.50},
    {"name": "fang_c06", "alpha": 0.35, "lambda_delta": 0.30, "lambda_gate_fit": 1.00, "lambda_gate_sparse": 0.01, "beta": 0.50},
]
DELTA_MAX = 0.35
SEED = 314159
EPOCHS = 80
LEARNING_RATE = 1e-3


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# FANG-VLA Development Audit",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- development records: `{report['development_records']}`",
        f"- train records: `{report['train_records']}`",
        f"- validation records: `{report['validation_records']}`",
        f"- duplicate development keys: `{report['duplicate_development_keys']}`",
        f"- validation gateable fraction: `{report['validation_gateable_fraction']}`",
        f"- validation median action-field separation: `{(report['validation_action_field_separation'] or {}).get('median')}`",
        "",
        "Hard stop reasons:",
    ]
    reasons = list(report.get("hard_stop_reasons") or [])
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Class counts:",
            "",
            "```json",
            json.dumps(report.get("combined_identity_counts"), indent=2, sort_keys=True),
            "```",
            "",
            f"Next step: {report['next_step']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


class FANGHead(torch.nn.Module):
    def __init__(self, input_dim: int = 25, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.m_plus = torch.nn.Linear(hidden_dim, 7)
        self.m_minus = torch.nn.Linear(hidden_dim, 7)
        self.gate = torch.nn.Linear(hidden_dim, 1)
        torch.nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.m_plus(h), self.m_minus(h), self.gate(h).reshape(-1)


def _clip_l2(value: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(value, dim=1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(float(max_norm) / norm, max=1.0)
    return value * scale


def _delta(
    *,
    m_plus: torch.Tensor,
    m_minus: torch.Tensor,
    gate_logits: torch.Tensor,
    base_actions: torch.Tensor,
    alpha: float,
    beta: float,
    tau: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    guidance = (m_plus - base_actions) + float(beta) * (m_plus - m_minus)
    clipped = _clip_l2(guidance, DELTA_MAX)
    gate = torch.sigmoid(gate_logits - float(tau))
    delta = float(alpha) * gate.reshape(-1, 1) * clipped
    return delta, gate, guidance


def _loss_terms(
    *,
    model: FANGHead,
    x: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
    gate_targets: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    m_plus, m_minus, gate_logits = model(x)
    success_mask = labels == 1
    failure_mask = labels == 0
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    success_loss = torch.nn.functional.smooth_l1_loss(m_plus[success_mask], actions[success_mask]) if bool(success_mask.any()) else zero
    failure_loss = torch.nn.functional.smooth_l1_loss(m_minus[failure_mask], actions[failure_mask]) if bool(failure_mask.any()) else zero
    delta, gate, _guidance = _delta(
        m_plus=m_plus,
        m_minus=m_minus,
        gate_logits=gate_logits,
        base_actions=actions,
        alpha=float(config["alpha"]),
        beta=float(config["beta"]),
        tau=0.0,
    )
    delta_loss = torch.mean(torch.sum(delta * delta, dim=1))
    gate_fit = torch.nn.functional.binary_cross_entropy_with_logits(gate_logits, gate_targets)
    gate_sparse = torch.mean(gate)
    total = (
        success_loss
        + failure_loss
        + float(config["lambda_delta"]) * delta_loss
        + float(config["lambda_gate_fit"]) * gate_fit
        + float(config["lambda_gate_sparse"]) * gate_sparse
    )
    return {
        "total": total,
        "success": success_loss.detach(),
        "failure": failure_loss.detach(),
        "delta": delta_loss.detach(),
        "gate_fit": gate_fit.detach(),
        "gate_sparse": gate_sparse.detach(),
    }


def _grad_norms(model: FANGHead) -> dict[str, float]:
    groups = {
        "trunk": list(model.trunk.parameters()),
        "m_plus": list(model.m_plus.parameters()),
        "m_minus": list(model.m_minus.parameters()),
        "gate": list(model.gate.parameters()),
    }
    out: dict[str, float] = {}
    for name, params in groups.items():
        total = 0.0
        for param in params:
            if param.grad is None:
                continue
            value = float(torch.sum(param.grad.detach() * param.grad.detach()).item())
            total += value
        out[name] = float(total ** 0.5)
    return out


def _score_validation(metrics: Mapping[str, float]) -> dict[str, float]:
    mechanism = min(float(metrics["median_head_separation"]) / 0.10, 1.0)
    clean = 1.0 - float(np.clip(float(metrics["mean_delta_l2"]) / 0.20, 0.0, 1.0))
    validity = float(metrics["action_validity"])
    activation = float(metrics["gate_activation_fraction"])
    if activation < 0.05:
        bounded = activation / 0.05
    elif activation <= 0.60:
        bounded = 1.0
    else:
        bounded = max(0.0, 1.0 - (activation - 0.60) / 0.40)
    efficiency = 1.0
    total = 0.35 * mechanism + 0.25 * clean + 0.20 * validity + 0.10 * bounded + 0.10 * efficiency
    return {
        "mechanism_separation": float(mechanism),
        "clean_retention": float(clean),
        "action_validity": float(validity),
        "bounded_activation": float(bounded),
        "compute_efficiency": float(efficiency),
        "total": float(total),
    }


def _tensor(array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype)


def _calibrate_tau(gate_logits: torch.Tensor, *, target_activation: float = 0.50, activation_threshold: float = 0.05) -> float:
    logits = gate_logits.detach().cpu().numpy().reshape(-1)
    if logits.size == 0:
        return 0.0
    threshold_logit = float(np.log(float(activation_threshold) / (1.0 - float(activation_threshold))))
    activation_cut = float(np.quantile(logits, 1.0 - float(target_activation)))
    return activation_cut - threshold_logit


def _run_train_validate(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    audit = audit_fang_records(records)
    if audit["final_decision"] != "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH":
        return {
            "schema_version": 1,
            "method": "FANG-VLA",
            "final_decision": "VALIDATION_SEARCH_BLOCKED_BY_AUDIT",
            "audit": audit,
            "configs": [],
            "selected_config": None,
        }

    splits = split_development_records(records)
    train_arrays = records_to_arrays(splits["train"])
    val_arrays = records_to_arrays(splits["validation"])
    standardized = standardize_train_validation(train_arrays["features"], val_arrays["features"])
    gate_train = compute_gate_targets(
        train_features_std=standardized["train_features"],
        train_actions=train_arrays["actions"],
        train_tasks=train_arrays["tasks"],
        train_labels=train_arrays["labels"],
        query_features_std=standardized["train_features"],
        query_tasks=train_arrays["tasks"],
    )
    gate_val = compute_gate_targets(
        train_features_std=standardized["train_features"],
        train_actions=train_arrays["actions"],
        train_tasks=train_arrays["tasks"],
        train_labels=train_arrays["labels"],
        query_features_std=standardized["validation_features"],
        query_tasks=val_arrays["tasks"],
    )

    torch.set_num_threads(1)
    x_train = _tensor(standardized["train_features"])
    a_train = _tensor(train_arrays["actions"])
    y_train = torch.as_tensor(train_arrays["labels"], dtype=torch.long)
    g_train = _tensor(gate_train["targets"])
    x_val = _tensor(standardized["validation_features"])
    a_val = _tensor(val_arrays["actions"])
    y_val = torch.as_tensor(val_arrays["labels"], dtype=torch.long)
    g_val = _tensor(gate_val["targets"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    configs: list[dict[str, Any]] = []
    for index, config in enumerate(SEARCH_CONFIGS):
        torch.manual_seed(SEED + index)
        model = FANGHead()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        with torch.no_grad():
            initial_terms = _loss_terms(model=model, x=x_train, actions=a_train, labels=y_train, gate_targets=g_train, config=config)
        first_grad_norms: dict[str, float] | None = None
        for epoch in range(EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            terms = _loss_terms(model=model, x=x_train, actions=a_train, labels=y_train, gate_targets=g_train, config=config)
            terms["total"].backward()
            if epoch == 0:
                first_grad_norms = _grad_norms(model)
            optimizer.step()
        with torch.no_grad():
            final_train_terms = _loss_terms(model=model, x=x_train, actions=a_train, labels=y_train, gate_targets=g_train, config=config)
            val_terms = _loss_terms(model=model, x=x_val, actions=a_val, labels=y_val, gate_targets=g_val, config=config)
            m_plus, m_minus, gate_logits = model(x_val)
            tau = _calibrate_tau(gate_logits, target_activation=0.50, activation_threshold=0.05)
            delta, gate, _guidance = _delta(
                m_plus=m_plus,
                m_minus=m_minus,
                gate_logits=gate_logits,
                base_actions=a_val,
                alpha=float(config["alpha"]),
                beta=float(config["beta"]),
                tau=tau,
            )
            proposed_actions = a_val + delta
            delta_l2 = torch.linalg.norm(delta, dim=1).detach().cpu().numpy()
            head_sep = torch.linalg.norm(m_plus - m_minus, dim=1).detach().cpu().numpy()
            gate_np = gate.detach().cpu().numpy()
            proposed_np = proposed_actions.detach().cpu().numpy()
        checkpoint_path = checkpoint_dir / f"{config['name']}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": dict(config),
                "feature_mean": standardized["feature_mean"],
                "feature_scale": standardized["feature_scale"],
                "gate_tau": tau,
                "proposal_hash": PROPOSAL_HASH,
            },
            checkpoint_path,
        )
        reloaded = FANGHead()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reloaded.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            old = model(x_val[:32])[0]
            new = reloaded(x_val[:32])[0]
            reload_diff = float(torch.max(torch.abs(old - new)).item()) if old.numel() else 0.0
        val_metrics = {
            "validation_loss": float(val_terms["total"].item()),
            "median_head_separation": float(np.median(head_sep)) if head_sep.size else 0.0,
            "mean_head_separation": float(np.mean(head_sep)) if head_sep.size else 0.0,
            "mean_delta_l2": float(np.mean(delta_l2)) if delta_l2.size else 0.0,
            "p95_delta_l2": float(np.percentile(delta_l2, 95)) if delta_l2.size else 0.0,
            "mean_gate": float(np.mean(gate_np)) if gate_np.size else 0.0,
            "gate_activation_fraction": float(np.mean(gate_np > 0.05)) if gate_np.size else 0.0,
            "action_validity": float(np.mean(np.all(np.isfinite(proposed_np), axis=1) & (np.max(np.abs(proposed_np), axis=1) <= 5.0))) if proposed_np.size else 0.0,
        }
        score = _score_validation(val_metrics)
        config_report = {
            "name": config["name"],
            "config": dict(config),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_reload_max_abs_diff": reload_diff,
            "loss_initial": {key: float(value.item()) for key, value in initial_terms.items()},
            "loss_final_train": {key: float(value.item()) for key, value in final_train_terms.items()},
            "loss_validation": {key: float(value.item()) for key, value in val_terms.items()},
            "first_gradient_norms": first_grad_norms or {},
            "validation_metrics": val_metrics,
            "validation_score_terms": score,
            "gate_tau": tau,
        }
        configs.append(config_report)

    selected = max(configs, key=lambda item: float(item["validation_score_terms"]["total"]))
    selected_config_path = output_dir / "selected_config.json"
    _write_json(selected_config_path, selected)
    hard_stop_reasons: list[str] = []
    if float(selected["checkpoint_reload_max_abs_diff"]) > 1e-6:
        hard_stop_reasons.append("selected checkpoint reload mismatch")
    if any(float(value) <= 0.0 or not np.isfinite(float(value)) for value in selected["first_gradient_norms"].values()):
        hard_stop_reasons.append("selected config has missing or nonfinite first-step gradients")
    if float(selected["validation_metrics"]["action_validity"]) < 1.0:
        hard_stop_reasons.append("selected config has invalid validation actions")
    activation = float(selected["validation_metrics"]["gate_activation_fraction"])
    if activation <= 0.0:
        hard_stop_reasons.append("selected config gate never activates")
    if activation >= 0.95:
        hard_stop_reasons.append("selected config gate activates almost everywhere")
    final_decision = "VALIDATION_SEARCH_COMPLETED_CONFIG_SELECTED" if not hard_stop_reasons else "VALIDATION_SEARCH_STOP_DESIGN_FAILURE"
    return {
        "schema_version": 1,
        "method": "FANG-VLA",
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": False,
        "training_happened": True,
        "final_decision": final_decision,
        "hard_stop_reasons": hard_stop_reasons,
        "audit_final_decision": audit["final_decision"],
        "train_records": len(splits["train"]),
        "validation_records": len(splits["validation"]),
        "gate_target_summary": {
            "train_mean": float(np.mean(gate_train["targets"])),
            "validation_mean": float(np.mean(gate_val["targets"])),
            "eta": float(gate_train["eta"]),
            "gamma": float(gate_train["gamma"]),
        },
        "search_budget": {
            "total_configurations": len(SEARCH_CONFIGS),
            "epochs_per_configuration": EPOCHS,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "delta_max": DELTA_MAX,
        },
        "configs": configs,
        "selected_config": selected,
        "selected_config_path": str(selected_config_path),
        "next_step": "Proceed to implementation smoke or Stage A only if reviewer accepts validation search." if not hard_stop_reasons else "Do not roll out FANG; classify the design failure.",
    }


def _write_validation_md(path: Path, report: Mapping[str, Any]) -> None:
    selected = report.get("selected_config") or {}
    selected_metrics = selected.get("validation_metrics") or {}
    selected_score = selected.get("validation_score_terms") or {}
    lines = [
        "# FANG-VLA Validation Search",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- train records: `{report['train_records']}`",
        f"- validation records: `{report['validation_records']}`",
        f"- total configurations: `{report['search_budget']['total_configurations']}`",
        f"- selected config: `{selected.get('name')}`",
        f"- selected score: `{selected_score.get('total')}`",
        f"- selected mean delta L2: `{selected_metrics.get('mean_delta_l2')}`",
        f"- selected gate activation fraction: `{selected_metrics.get('gate_activation_fraction')}`",
        f"- selected gate tau: `{selected.get('gate_tau')}`",
        f"- selected action validity: `{selected_metrics.get('action_validity')}`",
        "",
        "Hard stop reasons:",
    ]
    reasons = list(report.get("hard_stop_reasons") or [])
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines.extend(["", "Configurations:", ""])
    for item in report.get("configs") or []:
        score = (item.get("validation_score_terms") or {}).get("total")
        metrics = item.get("validation_metrics") or {}
        lines.append(
            f"- `{item['name']}` score `{score}` delta `{metrics.get('mean_delta_l2')}` gate `{metrics.get('gate_activation_fraction')}` tau `{item.get('gate_tau')}` validity `{metrics.get('action_validity')}`"
        )
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "train-validate"], default="audit")
    parser.add_argument("--records", default="reports/cavm_vla/acquisition_records.jsonl")
    parser.add_argument("--json-output", default="reports/fang_vla/development_audit.json")
    parser.add_argument("--md-output", default="reports/fang_vla/development_audit.md")
    parser.add_argument("--validation-json-output", default="reports/fang_vla/validation_search.json")
    parser.add_argument("--validation-md-output", default="reports/fang_vla/validation_search.md")
    parser.add_argument("--output-dir", default="reports/fang_vla")
    args = parser.parse_args()

    records = _read_jsonl(Path(args.records))
    if args.mode == "train-validate":
        report = _run_train_validate(records, Path(args.output_dir))
        _write_json(Path(args.validation_json_output), report)
        _write_validation_md(Path(args.validation_md_output), report)
        print(
            json.dumps(
                {
                    "final_decision": report["final_decision"],
                    "json_output": args.validation_json_output,
                    "md_output": args.validation_md_output,
                    "selected_config": (report.get("selected_config") or {}).get("name"),
                },
                sort_keys=True,
            )
        )
        return 0

    report = audit_fang_records(records)
    report = {
        **report,
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "mode": args.mode,
        "source_records": str(args.records),
    }
    _write_json(Path(args.json_output), report)
    _write_md(Path(args.md_output), report)
    print(json.dumps({"final_decision": report["final_decision"], "json_output": args.json_output, "md_output": args.md_output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
