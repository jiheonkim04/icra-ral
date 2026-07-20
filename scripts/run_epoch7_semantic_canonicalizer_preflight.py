#!/usr/bin/env python3
"""Evaluate the frozen MiniLM semantic canonicalizer without robot outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.epoch7_selective_language_grounding import atomic_write_json, load_json, iter_pair_specs

DEFAULT_PROTOCOL = REPO_ROOT / "reports/epoch7_selective_language_grounding/problem_verification_protocol.json"
DEFAULT_METADATA = Path("/mnt/c/assets/repos/LIBERO-Para/metrics/libero_para_metadata.csv")
DEFAULT_CACHE = Path("/home/jiheon/assets/checkpoints/epoch7_controls/hf")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_row_to_bddl(row: dict[str, str]) -> str:
    return f"{row['high']}_{row['mid']}_{row['low']}_eval{int(row['eval'])}_ver{int(row['batch_idx'])}.bddl"


def encode_texts(texts: list[str], tokenizer: Any, model: Any, torch_module: Any, batch_size: int) -> Any:
    import torch.nn.functional as functional

    outputs = []
    model.eval()
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch_module.no_grad():
            model_output = model(**encoded)
        token_embeddings = model_output[0]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch_module.sum(token_embeddings * mask, dim=1) / torch_module.clamp(mask.sum(dim=1), min=1e-9)
        outputs.append(functional.normalize(pooled, p=2, dim=1).cpu())
    return torch_module.cat(outputs, dim=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_json(args.protocol)
    frozen = protocol["strong_semantic_control"]
    model_id = str(frozen["model_id"])
    revision = str(frozen["model_revision"])
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows = list(csv.DictReader(args.metadata.open(encoding="utf-8")))
    if len(rows) != 4092:
        raise ValueError(f"expected 4092 metadata rows, found {len(rows)}")
    catalog = {int(task["eval_id"]): str(task["canonical_instruction"]) for task in protocol["tasks"]}
    if set(catalog) != set(range(10)):
        raise ValueError("canonical catalog must contain eval IDs 0..9")

    started = time.monotonic()
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(args.cache_dir),
        local_files_only=bool(args.offline),
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(args.cache_dir),
        local_files_only=bool(args.offline),
    ).to(args.device)
    catalog_ids = sorted(catalog)
    catalog_embeddings = encode_texts([catalog[index] for index in catalog_ids], tokenizer, model, torch, args.batch_size)
    query_embeddings = encode_texts([row["new_instruction"] for row in rows], tokenizer, model, torch, args.batch_size)
    scores = query_embeddings @ catalog_embeddings.T
    predicted_columns = torch.argmax(scores, dim=1).tolist()
    predictions = [catalog_ids[index] for index in predicted_columns]

    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    wrong_examples: list[dict[str, Any]] = []
    by_filename: dict[str, dict[str, Any]] = {}
    correct = 0
    for index, (row, predicted_eval) in enumerate(zip(rows, predictions)):
        true_eval = int(row["eval"])
        correct += int(predicted_eval == true_eval)
        confusion[true_eval][predicted_eval] += 1
        ranked = torch.argsort(scores[index], descending=True).tolist()
        top_score = float(scores[index, ranked[0]])
        runner_up_score = float(scores[index, ranked[1]])
        filename = metadata_row_to_bddl(row)
        record = {
            "true_eval_id": true_eval,
            "predicted_eval_id": int(predicted_eval),
            "predicted_instruction": catalog[int(predicted_eval)],
            "score": top_score,
            "runner_up_score": runner_up_score,
            "margin": top_score - runner_up_score,
            "mapping_correct": bool(predicted_eval == true_eval),
        }
        by_filename[filename] = record
        if predicted_eval != true_eval and len(wrong_examples) < 100:
            wrong_examples.append({"bddl": filename, "instruction": row["new_instruction"], **record})

    panel = []
    for spec in iter_pair_specs(protocol):
        mapping = by_filename[spec["paraphrase_bddl"]]
        panel.append(
            {
                "pair_id": spec["pair_id"],
                "eval_id": spec["eval_id"],
                "family": spec["family"],
                "paraphrase_bddl": spec["paraphrase_bddl"],
                **mapping,
            }
        )
    panel_correct = sum(int(item["mapping_correct"]) for item in panel)

    snapshot = args.cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    snapshot_files = []
    if snapshot.exists():
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
            snapshot_files.append(
                {
                    "path": str(path.relative_to(snapshot)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    payload = {
        "schema_version": "epoch7.semantic_canonicalizer_preflight.v1",
        "execution_type": "TEXT_ONLY_CONTROL_NO_SIMULATOR_NO_VLA_NO_OUTCOMES",
        "model_id": model_id,
        "model_revision": revision,
        "model_commit_hash_loaded": getattr(model.config, "_commit_hash", None),
        "license": frozen["license"],
        "implementation": frozen["implementation"],
        "device": args.device,
        "metadata_path": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "catalog": [{"eval_id": index, "instruction": catalog[index]} for index in catalog_ids],
        "full_benchmark": {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "confusion": {str(key): dict(value) for key, value in sorted(confusion.items())},
            "wrong_examples_first_100": wrong_examples,
        },
        "frozen_discovery_panel": {
            "correct": panel_correct,
            "total": len(panel),
            "accuracy": panel_correct / len(panel),
            "predictions": panel,
        },
        "snapshot": str(snapshot),
        "snapshot_files": snapshot_files,
        "elapsed_seconds": time.monotonic() - started,
        "simulator_episode_count": 0,
        "vla_model_loaded": False,
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "ours_design_happened": False,
        "ours_rollout_happened": False,
        "closed_loop_outcome_read": False,
    }
    atomic_write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
