"""Freeze and preflight EAC-VLA Stage A manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.eac_vla import PROPOSAL_HASH, chunk_sha256, eac_commitment_prefix  # noqa: E402
from tca_map.smolvla.official_canonical_eval import _make_noise, _postprocess_chunk  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import (  # noqa: E402
    PolicySpec,
    _cuda_memory,
    _dummy_observation,
    _json_default,
    _load_policy_and_processors,
    _round,
    _set_runtime_env,
)


DATE_KST = "2026-07-15"
RESET_SEEDS = [20261211, 20261212]
STAGE_A_TASKS = [
    {
        "suite": "libero_spatial",
        "task_id": 0,
        "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    },
    {
        "suite": "libero_spatial",
        "task_id": 8,
        "instruction": "pick up the black bowl next to the plate and place it on the plate",
    },
    {
        "suite": "libero_object",
        "task_id": 6,
        "instruction": "pick up the butter and place it in the basket",
    },
    {
        "suite": "libero_goal",
        "task_id": 4,
        "instruction": "put the bowl on top of the cabinet",
    },
    {
        "suite": "libero_10",
        "task_id": 2,
        "instruction": "turn on the stove and put the moka pot on it",
    },
]
POLICIES = [
    {
        "policy": "frozen_smolvla_fixed_queue",
        "role": "base",
        "scheduler": "fixed_commitment",
        "commitment": 50,
        "proxy_or_reproduction_label": "unmodified frozen SmolVLA action values with fixed full queue commitment",
    },
    {
        "policy": "aac_entropy_proxy",
        "role": "closest_prior_proxy",
        "scheduler": "dispersion_only_quantile_proxy",
        "commitment": 8,
        "commitment_map": {"short": 2, "medium": 8, "long": 50},
        "quantile_margin": 0.33,
        "proxy_or_reproduction_label": "faithful transparent local proxy, not an official AAC reproduction",
    },
    {
        "policy": "eac_full",
        "role": "ours",
        "scheduler": "selected_validation_config",
        "commitment": 4,
        "config_id": "eac_q33_aggressive_1_4_50",
        "commitment_map": {"short": 1, "medium": 4, "long": 50},
        "quantile_margin": 0.33,
        "proxy_or_reproduction_label": "ours",
    },
    {
        "policy": "eac_no_calibration_no_hysteresis_ablation",
        "role": "key_ablation",
        "scheduler": "raw_risk_fixed_threshold_ablation",
        "commitment": 4,
        "commitment_map": {"short": 1, "medium": 4, "long": 50},
        "raw_thresholds": {"low": 1.0 / 3.0, "high": 2.0 / 3.0},
        "proxy_or_reproduction_label": "key ablation",
    },
    {
        "policy": "fixed_short_replan_baseline",
        "role": "simple_killer",
        "scheduler": "fixed_commitment",
        "commitment": 1,
        "proxy_or_reproduction_label": "strong simple fixed short-replan baseline",
    },
]


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest().upper()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# EAC-VLA Stage A Manifest And Preflight",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- validation search happened: `{report['validation_search_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- planned episode count: `{report['planned_episode_count']}`",
        f"- paired cases per policy: `{report['paired_cases_per_policy']}`",
        f"- reset seeds: `{report['reset_seeds']}`",
        f"- policies: `{report['policy_order']}`",
        f"- canonical payload sha256: `{report['canonical_payload_sha256']}`",
        "",
        "Policy identities:",
        "",
        "```json",
        json.dumps(report["policy_identities"], indent=2, sort_keys=True),
        "```",
        "",
        "Preflight records:",
        "",
        "```json",
        json.dumps(report.get("preflight_records", []), indent=2, sort_keys=True),
        "```",
        "",
        "Errors:",
    ]
    errors = list(report.get("errors") or [])
    if errors:
        lines.extend(f"- `{error}`" for error in errors)
    else:
        lines.append("- none")
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _episode_manifest() -> list[dict[str, Any]]:
    episodes = []
    index = 0
    for policy in POLICIES:
        policy_name = str(policy["policy"])
        for task in STAGE_A_TASKS:
            for seed in RESET_SEEDS:
                pair_id = f"{task['suite']}|task_{task['task_id']}|seed_{seed}"
                episodes.append(
                    {
                        "planned_episode_index": index,
                        "episode_id": f"{policy_name}|{pair_id}",
                        "pair_id": pair_id,
                        "policy": policy_name,
                        "suite": task["suite"],
                        "task_id": int(task["task_id"]),
                        "instruction": task["instruction"],
                        "reset_seed": int(seed),
                    }
                )
                index += 1
    return episodes


def _validate_manifest(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    episodes = list(payload.get("episodes") or [])
    policy_order = list(payload.get("policy_order") or [])
    pair_sets = {}
    episode_ids = [row["episode_id"] for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        errors.append("duplicate episode ids")
    for policy in policy_order:
        pair_sets[policy] = {row["pair_id"] for row in episodes if row["policy"] == policy}
    if len({tuple(sorted(values)) for values in pair_sets.values()}) != 1:
        errors.append("policy pair identities are not matched")
    if len(episodes) != 50:
        errors.append(f"planned episode count is {len(episodes)}, expected 50")
    per_policy = Counter(row["policy"] for row in episodes)
    if any(count != 10 for count in per_policy.values()):
        errors.append(f"per-policy count mismatch: {dict(per_policy)}")
    if sorted(set(row["reset_seed"] for row in episodes)) != RESET_SEEDS:
        errors.append("reset seed set mismatch")
    return errors


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    selected_config = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    episodes = _episode_manifest()
    payload = {
        "schema_version": 1,
        "date": f"{DATE_KST} KST",
        "method": "EAC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "mode": "stage_a_manifest",
        "branch": "codex/autonomous-until-paper-governance-v2",
        "selected_config": selected_config,
        "config_id": selected_config["config_id"],
        "policy_order": [item["policy"] for item in POLICIES],
        "policy_identities": POLICIES,
        "reset_seeds": RESET_SEEDS,
        "tasks": STAGE_A_TASKS,
        "episodes": episodes,
        "planned_episode_count": len(episodes),
        "paired_cases_per_policy": len(STAGE_A_TASKS) * len(RESET_SEEDS),
        "official_success_condition": "LIBERO official task success from environment final_info/is_success",
        "policy_order_affects_env_initialization": False,
        "fixed_task_balanced_allocation": True,
        "no_post_hoc_task_or_reset_selection": True,
        "confirmatory_test_identities_used_for_training_or_validation": False,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "errors": [],
    }
    payload["errors"] = _validate_manifest(payload)
    canonical_payload = {key: value for key, value in payload.items() if key not in {"canonical_payload_sha256", "errors"}}
    payload["canonical_payload_sha256"] = _sha256_payload(canonical_payload)
    payload["final_decision"] = "EAC_STAGE_A_PLAN_FROZEN_PREFLIGHT_PENDING" if not payload["errors"] else "IMPLEMENTATION_FAILURE"
    payload["next_step"] = (
        "Run the EAC Stage A policy preflight before any rollout."
        if not payload["errors"]
        else "Fix only the concrete manifest defect before preflight."
    )
    return payload


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_runtime_env(args)
    manifest = json.loads(Path(args.stage_a_manifest).read_text(encoding="utf-8"))
    loaded = _load_policy_and_processors(args, PolicySpec("frozen_base"))
    policy = loaded["policy"]
    env_preprocessor = loaded["env_preprocessor"]
    preprocessor = loaded["preprocessor"]
    postprocessor = loaded["postprocessor"]
    dummy = env_preprocessor(_dummy_observation(torch))
    batch = preprocessor(dummy)
    policy.reset()
    noise = _make_noise(policy, int(args.runtime_seed), torch)
    with torch.inference_mode():
        raw_chunk = policy.predict_action_chunk(batch, noise=noise)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    postprocessed_chunk, _ = _postprocess_chunk(raw_chunk, postprocessor, 7)
    errors = list(manifest.get("errors") or [])
    records = []
    for identity in manifest["policy_identities"]:
        commitment = int(identity["commitment"])
        prefix = eac_commitment_prefix(postprocessed_chunk, commitment)
        max_diff = float(np.max(np.abs(prefix - postprocessed_chunk[:commitment]))) if prefix.size else 0.0
        records.append(
            {
                "policy": identity["policy"],
                "role": identity["role"],
                "scheduler": identity["scheduler"],
                "commitment": commitment,
                "prefix_shape": [int(dim) for dim in prefix.shape],
                "prefix_sha256": chunk_sha256(prefix),
                "prefix_max_abs_diff": max_diff,
                "action_values_modified": bool(max_diff > 0.0),
                "proxy_or_reproduction_label": identity["proxy_or_reproduction_label"],
            }
        )
        if max_diff > 0.0:
            errors.append(f"{identity['policy']} changed action values")
    output_shape_ok = [int(dim) for dim in postprocessed_chunk.shape] == [50, 7]
    if not output_shape_ok:
        errors.append(f"postprocessed chunk shape mismatch: {list(postprocessed_chunk.shape)}")
    if not bool(np.isfinite(postprocessed_chunk).all()):
        errors.append("postprocessed chunk contains nonfinite values")
    if (loaded.get("audit") or {}).get("old_custom_libero_7d_route_used"):
        errors.append("old custom LIBERO_7D route used")
    report = {
        **{key: value for key, value in manifest.items() if key != "episodes"},
        "mode": "stage_a_preflight",
        "final_decision": "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING" if not errors else "IMPLEMENTATION_FAILURE",
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_sha256": _sha256_payload(manifest),
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "preflight_records": records,
        "policy_count": len(records),
        "checkpoint_policy_count": 0,
        "cuda_ok": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_memory": _cuda_memory(torch),
        "policy_output_shape": [int(dim) for dim in postprocessed_chunk.shape],
        "policy_output_shape_ok": output_shape_ok,
        "policy_output_finite": bool(np.isfinite(postprocessed_chunk).all()),
        "all_policy_prefixes_value_preserving": bool(all(not row["action_values_modified"] for row in records)),
        "no_accidental_checkpoint_reuse": True,
        "old_custom_libero_7d_route_used": bool((loaded.get("audit") or {}).get("old_custom_libero_7d_route_used")),
        "errors": errors,
        "next_step": (
            "Implement the minimal EAC Stage A runner and launch only after runner validation."
            if not errors
            else "Fix only the concrete preflight defect before runner implementation."
        ),
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["freeze-stage-a-manifest", "preflight"], default="freeze-stage-a-manifest")
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--runtime-seed", type=int, default=20260715)
    parser.add_argument("--selected-config", default="reports/eac_vla/selected_config.json")
    parser.add_argument("--stage-a-manifest", default="reports/eac_vla/stage_a_manifest.json")
    parser.add_argument("--stage-a-manifest-md", default="reports/eac_vla/stage_a_manifest.md")
    parser.add_argument("--stage-a-preflight-output", default="reports/eac_vla/stage_a_preflight.json")
    parser.add_argument("--stage-a-preflight-md", default="reports/eac_vla/stage_a_preflight.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "freeze-stage-a-manifest":
        report = build_manifest(args)
        _write_json(Path(args.stage_a_manifest), report)
        _write_md(Path(args.stage_a_manifest_md), report)
    else:
        if not Path(args.stage_a_manifest).exists():
            manifest = build_manifest(args)
            _write_json(Path(args.stage_a_manifest), manifest)
            _write_md(Path(args.stage_a_manifest_md), manifest)
        report = build_preflight(args)
        _write_json(Path(args.stage_a_preflight_output), report)
        _write_md(Path(args.stage_a_preflight_md), report)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "final_decision": report["final_decision"],
                "planned_episode_count": report["planned_episode_count"],
                "policy_count": len(report.get("policy_identities") or []),
                "error_count": len(report.get("errors") or []),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
