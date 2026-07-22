"""Build the outcome-blind Epoch 10B whole-seed checkpoint expansion.

This runner reuses the already validated Epoch 10 rank-4 LoRA training path but
writes only to a new Epoch 10B asset root.  Existing Epoch 10 adapters are read
and hash-verified; they are never rewritten.  The development/holdout split and
the two retained stages are constants fixed before checkpoint actions or
comparative simulator outcomes are opened.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_epoch10_checkpoint_panel as legacy


SCHEMA_VERSION = 1
CAMPAIGN = "epoch10b_icae_fresh_controller"
ORIGINAL_SEEDS = (101, 202, 303, 404)
NEW_SEEDS = (505, 606, 707, 808, 909, 1010, 1111, 1212)
DEVELOPMENT_SEEDS = frozenset((101, 202, 505, 606, 707, 808, 909, 1010))
HOLDOUT_SEEDS = frozenset((303, 404, 1111, 1212))
RETAINED_STAGES = (30, 100)
EXPECTED_ORIGINAL_ADAPTERS = 12
EXPECTED_NEW_ADAPTERS = len(NEW_SEEDS) * len(RETAINED_STAGES)
ORIGINAL_RESULT_SHA256 = "90225e06a22cd4d9f6f4589ed76d06624b9f391aa407db8582b1d59b2f09f530"
ORIGINAL_PROMPT_SHA256 = "24c2198d83ea262ff4133ffe7d44d63af65bd7ab93f237fdebcd4a47aeaefa66"
CONTINUATION_PROMPT_SHA256 = "29abb6edf9a4c662a42ac0eea7f8543ac5c1e88f42294056d1b1f4b07b0db420"
ERRATUM_PARENT_COMMIT = "9216ac1aaf93caaeb2a00597832e6d9d0afaddbe"


class ExpansionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _partition(seed: int) -> str:
    if seed in DEVELOPMENT_SEEDS:
        return "checkpoint_development_panel"
    if seed in HOLDOUT_SEEDS:
        return "checkpoint_holdout_panel"
    raise ExpansionError("UNREGISTERED_SEED", f"Seed {seed} is outside the frozen whole-seed split")


def _validate_output_root(path: Path, original_root: Path) -> Path:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 4:
        raise ExpansionError("UNSAFE_OUTPUT_ROOT", f"Refusing broad output path: {resolved}")
    if resolved == original_root.resolve() or original_root.resolve() in resolved.parents:
        raise ExpansionError("PROTECTED_ROOT_COLLISION", f"Expansion root overlaps protected Epoch 10 root: {resolved}")
    return resolved


def _original_inventory(result_path: Path, original_root: Path) -> list[dict[str, Any]]:
    if legacy._sha256(result_path) != ORIGINAL_RESULT_SHA256:
        raise ExpansionError("ORIGINAL_MANIFEST_DRIFT", f"Unexpected hash for {result_path}")
    result = _read_json(result_path)
    checkpoints = [
        checkpoint
        for seed_row in result.get("seeds", [])
        for checkpoint in seed_row.get("checkpoints", [])
    ]
    if len(checkpoints) != EXPECTED_ORIGINAL_ADAPTERS:
        raise ExpansionError("ORIGINAL_PANEL_INCOMPLETE", f"Expected 12 original adapters, found {len(checkpoints)}")
    inventory: list[dict[str, Any]] = []
    for row in checkpoints:
        seed = int(row["seed"] if "seed" in row else str(row["lineage_cluster"]).rsplit("_", 1)[-1])
        if seed not in ORIGINAL_SEEDS:
            raise ExpansionError("ORIGINAL_PANEL_DRIFT", f"Unexpected original seed {seed}")
        adapter = Path(row["path"]) / "adapter_model.safetensors"
        try:
            adapter.relative_to(original_root.resolve())
        except ValueError as exc:
            raise ExpansionError("ORIGINAL_PANEL_DRIFT", f"Adapter is outside protected root: {adapter}") from exc
        actual = legacy._sha256(adapter)
        if actual != row["adapter_sha256"]:
            raise ExpansionError("ORIGINAL_ADAPTER_DRIFT", f"Hash mismatch for {adapter}")
        inventory.append(
            {
                "policy_identity": row["policy_identity"],
                "lineage_cluster": row["lineage_cluster"],
                "partition": _partition(seed),
                "seed": seed,
                "optimizer_step": int(Path(row["path"]).name.rsplit("_", 1)[-1]),
                "path": str(Path(row["path"]).resolve()),
                "adapter_sha256": actual,
                "source": "immutable_epoch10_panel",
            }
        )
    return inventory


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    import psutil

    original_result = Path(args.original_result).resolve()
    original_root = Path(args.original_root).resolve()
    output_root = _validate_output_root(Path(args.output_root), original_root)
    required = {
        "checkpoint_path": Path(args.checkpoint_path),
        "dataset_root": Path(args.dataset_root),
        "hf_home": Path(args.hf_home),
        "vlm_root": Path(args.vlm_root),
        "split_manifest": Path(args.split_manifest),
        "source_repro_lock": Path(args.source_repro_lock),
        "metric_protocol": Path(args.metric_protocol),
        "original_result": original_result,
        "original_root": original_root,
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise ExpansionError("MISSING_REQUIRED_ASSET", "; ".join(missing))
    host_ram_percent = float(psutil.virtual_memory().percent)
    if host_ram_percent >= 80.0:
        raise ExpansionError("HOST_RAM_SOFT_STOP", f"Host RAM is already {host_ram_percent:.2f}%")
    disk = shutil.disk_usage(output_root.anchor)
    if disk.free < 10 * 1024**3:
        raise ExpansionError("INSUFFICIENT_DISK", f"Only {disk.free} bytes free on {output_root.anchor}")
    original_inventory = _original_inventory(original_result, original_root)
    return {
        "source_commit": _git_head(),
        "original_inventory": original_inventory,
        "original_adapter_count": len(original_inventory),
        "new_seeds": list(NEW_SEEDS),
        "retained_stages": list(RETAINED_STAGES),
        "new_adapter_count_expected": EXPECTED_NEW_ADAPTERS,
        "whole_seed_partitions": {str(seed): _partition(seed) for seed in (*ORIGINAL_SEEDS, *NEW_SEEDS)},
        "host_ram_percent": host_ram_percent,
        "disk_free_bytes": int(disk.free),
        "output_root": str(output_root),
        "checkpoint_actions_queried": 0,
        "comparative_simulator_outcomes_opened": False,
        "development_success_labels_opened": False,
        "heldout_success_labels_opened": False,
        "confirmation_results_opened": False,
    }


def _freeze_payload(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "FROZEN_BEFORE_LINEAGE_EXPANSION_OR_CHECKPOINT_ACTION_QUERY",
        "authority": {
            "erratum_parent_commit": ERRATUM_PARENT_COMMIT,
            "original_prompt_sha256": ORIGINAL_PROMPT_SHA256,
            "continuation_prompt_sha256": CONTINUATION_PROMPT_SHA256,
        },
        "uncertainty_unit": "whole_training_seed_lineage",
        "hard_minimum_lineages": 8,
        "target_lineages": 12,
        "frozen_total_lineages": len(ORIGINAL_SEEDS) + len(NEW_SEEDS),
        "development_seeds": sorted(DEVELOPMENT_SEEDS),
        "heldout_seeds": sorted(HOLDOUT_SEEDS),
        "new_training_seeds": list(NEW_SEEDS),
        "new_retained_optimizer_steps": list(RETAINED_STAGES),
        "retained_stage_rationale": "Predeclared intermediate and converged steps 30 and 100 match the Epoch 10 competitive-subset rule; no outcome selects a step.",
        "training_recipe": {
            "method": "standard_rank4_lora",
            "learning_rate": args.learning_rate,
            "batch_size": 1,
            "optimizer": "AdamW",
            "maximum_optimizer_steps": max(RETAINED_STAGES),
            "training_path": "scripts/run_epoch10_checkpoint_panel.py reused without changing the protected Epoch 10 asset root",
        },
        "prohibited_identity_construction": [
            "synthetic_action_noise",
            "renamed_checkpoint_copy",
            "weight_interpolation",
            "outcome_selected_training_step",
        ],
        "original_adapter_inventory": preflight["original_inventory"],
        "new_output_root": preflight["output_root"],
        "leakage_boundaries": {
            "checkpoint_actions_queried": 0,
            "comparative_simulator_outcomes_opened": False,
            "development_success_labels_opened": False,
            "heldout_success_labels_opened": False,
            "confirmation_results_opened": False,
        },
    }


def _completed_seed_from_report(report: dict[str, Any], seed: int) -> dict[str, Any] | None:
    for row in report.get("seeds", []):
        if int(row.get("seed", -1)) != seed:
            continue
        checkpoints = row.get("checkpoints", [])
        if len(checkpoints) != len(RETAINED_STAGES):
            return None
        for checkpoint in checkpoints:
            root = Path(checkpoint["path"])
            if not legacy._is_complete(root):
                return None
            if legacy._sha256(root / "adapter_model.safetensors") != checkpoint["adapter_sha256"]:
                return None
            if checkpoint.get("status") != "CHECKPOINT_COMPLETE_VERIFIED":
                return None
        return row
    return None


def _run(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(args.result_json)
    if result_path.exists():
        prior = _read_json(result_path)
        if prior.get("campaign") != CAMPAIGN:
            raise ExpansionError("RESUME_MANIFEST_MISMATCH", f"Unexpected campaign in {result_path}")
    else:
        prior = {}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "STARTED",
        "source_commit": _git_head(),
        "preflight": preflight,
        "frozen_development_seeds": sorted(DEVELOPMENT_SEEDS),
        "frozen_holdout_seeds": sorted(HOLDOUT_SEEDS),
        "retained_stages": list(RETAINED_STAGES),
        "checkpoint_actions_queried": 0,
        "comparative_simulator_outcomes_opened": False,
        "seeds": [],
    }
    manifest = legacy._read_json(Path(args.split_manifest))
    legacy._partition = _partition
    started = time.monotonic()
    for seed in NEW_SEEDS:
        completed = _completed_seed_from_report(prior, seed)
        if completed is not None:
            completed = dict(completed)
            completed["resume_status"] = "EXISTING_COMPLETE_SEED_REUSED_WITH_HASH_VERIFICATION"
            report["seeds"].append(completed)
        else:
            print(f"[epoch10b-expansion] training seed {seed}", flush=True)
            report["seeds"].append(
                legacy._train_seed(
                    args=args,
                    seed=seed,
                    stages=RETAINED_STAGES,
                    manifest=manifest,
                    started=started,
                )
            )
        _write_json(result_path, report)
    checkpoints = [row for seed in report["seeds"] for row in seed["checkpoints"]]
    report["new_checkpoint_count"] = len(checkpoints)
    report["all_disk_reloads_passed"] = all(
        row.get("status") == "CHECKPOINT_COMPLETE_VERIFIED" for row in checkpoints
    )
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["status"] = (
        "EPOCH10B_CHECKPOINT_LINEAGE_EXPANSION_COMPLETE"
        if len(checkpoints) == EXPECTED_NEW_ADAPTERS and report["all_disk_reloads_passed"]
        else "EPOCH10B_CHECKPOINT_LINEAGE_EXPANSION_INCOMPLETE"
    )
    _write_json(result_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "preflight", "run"), default="preflight")
    parser.add_argument("--checkpoint-path", default="C:/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="C:/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--vlm-root", default="C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--split-manifest", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--source-repro-lock", default="configs/official_smolvla_repro_lock.yaml")
    parser.add_argument("--metric-protocol", default="reports/official_smolvla_metric_protocol.md")
    parser.add_argument("--original-root", default="C:/assets/checkpoints/epoch10_icae_panel/rank4")
    parser.add_argument("--original-result", default="reports/epoch10_checkpoint_generation_result.json")
    parser.add_argument("--output-root", default="C:/assets/checkpoints/epoch10b_icae_lineage_expansion/rank4")
    parser.add_argument("--result-json", default="reports/epoch10b_checkpoint_generation_result.json")
    parser.add_argument("--freeze-json", default="reports/epoch10b_checkpoint_expansion_freeze.json")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        preflight = _preflight(args)
        if args.mode == "freeze":
            payload = _freeze_payload(args, preflight)
            _write_json(Path(args.freeze_json), payload)
        elif args.mode == "preflight":
            payload = {"status": "EPOCH10B_CHECKPOINT_EXPANSION_PREFLIGHT_PASS", **preflight}
        else:
            payload = _run(args, preflight)
        print(json.dumps({"status": payload["status"]}, sort_keys=True))
        return 0
    except (ExpansionError, legacy.PanelError) as exc:
        result = {"status": getattr(exc, "code", "EXPANSION_ERROR"), "error": str(exc), "traceback": traceback.format_exc()}
        print(json.dumps(result, sort_keys=True))
        return 2
    except Exception as exc:  # pragma: no cover - integration failure path
        result = {"status": "UNEXPECTED_IMPLEMENTATION_FAILURE", "error": str(exc), "traceback": traceback.format_exc()}
        print(json.dumps(result, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
