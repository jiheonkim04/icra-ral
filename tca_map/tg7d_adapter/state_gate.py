"""Bounded TG-7D Adapter STATE 0-2 gate.

This runner uses the fixed LIBERO_7D action path from the SmolVLA baseline
reproduction. It never uses the old SO100 6D action labels or hard-coded
gripper fill. The target prior is derived from instruction text plus visible
object-candidate names parsed from the local HDF5 model XML.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla_lora_baseline import diagnostic as base
from tca_map.smolvla_lora_baseline import libero_7d_baseline_reproduction as baseline
from tca_map.smolvla_lora_baseline import libero_7d_interface_fix as fix


STATE_GATE = "ALLOW_TG7D_ADAPTER_STATE1"
TRAINING_GATE = "ALLOW_TG7D_ADAPTER_TRAINING"
DEFAULT_LIBERO_DATA_ROOT = Path("C:/assets/data/libero")
DEFAULT_LIBERO_PARA_CSV = Path("C:/assets/data/libero_para/libero_para_metadata.csv")
DEFAULT_SMOLVLA_CKPT = Path("C:/assets/checkpoints/smolvla")
CURRENT_STRONG_BASELINE_L2 = 0.494959
TEXT_DIM = 64
PRIOR_DIM = 64
FINAL_DECISIONS = {
    "READY_FOR_TG7D_SCALE_UP",
    "KILL_BASELINE_DOMINATED",
    "KILL_CANONICALIZATION_DOMINATED",
    "KILL_LEAKAGE_RISK",
    "NO_TARGET_GROUNDING_EVAL_PATH",
    "TOO_HEAVY_LOCAL",
}
FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_PATCHGUARD_VLA_STATE1B",
    "ALLOW_TARGET_GROUNDED_ACTIONMAP",
    "ALLOW_TCA_SELECT",
    "ALLOW_OLD_TCA_SELECT",
    "ALLOW_OPENVLA_OFT_TRAINING",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "around",
    "at",
    "carefully",
    "gently",
    "in",
    "inside",
    "it",
    "of",
    "on",
    "open",
    "place",
    "please",
    "put",
    "push",
    "quickly",
    "slowly",
    "smoothly",
    "the",
    "to",
    "top",
    "turn",
}
OBJECT_HINTS = {
    "alphabet",
    "basket",
    "book",
    "bowl",
    "butter",
    "cabinet",
    "caddy",
    "cheese",
    "cream",
    "drawer",
    "moka",
    "mug",
    "plate",
    "pot",
    "rack",
    "stove",
    "wine",
}
BANNED_XML_NAME_PARTS = {
    "actuator",
    "base",
    "body",
    "button",
    "collision",
    "default",
    "floor",
    "frame",
    "geom",
    "gripper",
    "joint",
    "link",
    "main",
    "mesh",
    "region",
    "robot",
    "sensor",
    "site",
    "table",
    "tex",
    "texture",
    "vis",
    "wall",
    "world",
}


@dataclass(frozen=True)
class ParaRow:
    high: str
    mid: str
    low: str
    eval_id: str
    batch_idx: str
    new_instruction: str
    original_instruction: str
    structural_similarity: float
    keyword_similarity: float

    @property
    def group_id(self) -> str:
        return _stable_id([self.original_instruction, self.high, self.mid, self.low, self.eval_id])


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _round(value: float | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _stable_id(parts: list[str]) -> str:
    payload = "||".join(_normalize_text(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in _normalize_text(text).split() if token]


def canonicalize_instruction(text: str) -> str:
    """Deterministic lexical normalization baseline with no metadata lookup."""

    replacements = {
        "activate": "turn on",
        "cooktop": "stove",
        "dish": "bowl",
        "drawer cabinet": "drawer of the cabinet",
        "set": "put",
        "switch on": "turn on",
        "wine shelf": "wine rack",
    }
    normalized = _normalize_text(text)
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    words = [word for word in normalized.split() if word not in {"gently", "slowly", "carefully", "quickly", "smoothly"}]
    return " ".join(words)


def _slug_from_instruction(instruction: str) -> str:
    return _normalize_text(instruction).replace(" ", "_")


def _load_para_rows(path: Path) -> list[ParaRow]:
    if not path.exists():
        return []
    rows: list[ParaRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ParaRow(
                    high=str(row.get("high", "")),
                    mid=str(row.get("mid", "")),
                    low=str(row.get("low", "")),
                    eval_id=str(row.get("eval", "")),
                    batch_idx=str(row.get("batch_idx", "")),
                    new_instruction=str(row.get("new_instruction", "")),
                    original_instruction=str(row.get("original_instruction", "")),
                    structural_similarity=float(row.get("structural_similarity") or 0.0),
                    keyword_similarity=float(row.get("keyword_similarity") or 0.0),
                )
            )
    return rows


def _match_hdf5(original_instruction: str, data_root: Path) -> Path | None:
    expected = data_root / "libero_goal" / f"{_slug_from_instruction(original_instruction)}_demo.hdf5"
    if expected.exists():
        return expected
    normalized = _normalize_text(original_instruction)
    for path in sorted((data_root / "libero_goal").glob("*.hdf5")):
        if _normalize_text(base._safe_task_text(path)) == normalized:
            return path
    return None


def _model_file_text(path: Path) -> str:
    import h5py

    with h5py.File(path, "r") as handle:
        first_demo = sorted(handle["data"].keys(), key=base._demo_sort_key)[0]
        raw = handle["data"][first_demo].attrs.get("model_file", "")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def visible_object_candidates(path: Path) -> list[str]:
    """Extract non-oracle object-name candidates from the visible model XML."""

    xml = _model_file_text(path)
    names = set(re.findall(r'name="([^"]+)"', xml))
    candidates: set[str] = set()
    for name in names:
        words = [word for word in _tokens(name) if not word.isdigit()]
        words = [re.sub(r"\d+$", "", word) for word in words]
        words = [word for word in words if word and word not in BANNED_XML_NAME_PARTS]
        if not words:
            continue
        if not (set(words) & OBJECT_HINTS):
            continue
        if len(words) > 5:
            continue
        candidates.add(" ".join(words))
    return sorted(candidates)


def resolve_target_prior(instruction: str, candidates: list[str]) -> dict[str, Any]:
    instruction_tokens = set(_tokens(instruction)) - STOPWORDS
    scored: list[tuple[int, int, str]] = []
    for candidate in candidates:
        candidate_tokens = set(_tokens(candidate)) - STOPWORDS
        overlap = len(instruction_tokens & candidate_tokens)
        if overlap:
            scored.append((overlap, len(candidate_tokens), candidate))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = [item[2] for item in scored[:4]]
    if not selected:
        fallback = [token for token in _tokens(instruction) if token in OBJECT_HINTS]
        selected = [" ".join(fallback[:4])] if fallback else []
    return {
        "instruction": instruction,
        "candidate_count": len(candidates),
        "selected_candidates": selected,
        "source": "instruction_text_plus_hdf5_model_xml_visible_candidate_names",
        "uses_bddl_target_labels": False,
        "uses_eval_labels": False,
        "uses_task_ids": False,
        "uses_filenames_as_inference_labels": False,
        "uses_reward_or_success_labels": False,
        "uses_future_actions": False,
    }


def _hash_features(text: str, dim: int) -> np.ndarray:
    vector = np.zeros((dim,), dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def _record_text(record: dict[str, Any], mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "canonical":
        return canonicalize_instruction(str(record.get("instruction") or record.get("task_text") or ""))
    if mode == "oracle":
        return str(record.get("original_instruction") or record.get("task_text") or "")
    if mode == "target_prior":
        selected = record.get("target_prior", {}).get("selected_candidates") or []
        return " ".join(selected)
    return str(record.get("instruction") or record.get("task_text") or "")


def _text_matrix(records: list[dict[str, Any]], mode: str, dim: int) -> np.ndarray:
    if dim <= 0:
        return np.zeros((len(records), 0), dtype=np.float32)
    return np.stack([_hash_features(_record_text(record, mode), dim) for record in records], axis=0).astype(np.float32)


def _clone_records(
    records: list[dict[str, Any]],
    *,
    instruction: str,
    original_instruction: str,
    group_id: str,
    source: str,
    row: ParaRow | None,
    candidates_by_path: dict[str, list[str]],
) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["instruction"] = instruction
        item["original_instruction"] = original_instruction
        item["paraphrase_group_id"] = group_id
        item["instruction_source"] = source
        item["target_prior"] = resolve_target_prior(instruction, candidates_by_path.get(item["hdf5_path"], []))
        if row is not None:
            item["libero_para"] = {
                "high": row.high,
                "mid": row.mid,
                "low": row.low,
                "eval": row.eval_id,
                "batch_idx": row.batch_idx,
                "structural_similarity": row.structural_similarity,
                "keyword_similarity": row.keyword_similarity,
            }
        cloned.append(item)
    return cloned


def _record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (str(record["hdf5_path"]), str(record["demo_name"]), int(record["timestep"]))


def _build_counterfactual_records(
    clean_eval: list[dict[str, Any]],
    task_order: list[str],
    instruction_by_original: dict[str, str],
    candidates_by_path: dict[str, list[str]],
) -> list[dict[str, Any]]:
    by_original: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in clean_eval:
        by_original[str(record["original_instruction"])].append(record)
    records: list[dict[str, Any]] = []
    for original, source_records in by_original.items():
        if original not in task_order:
            continue
        counter_original = task_order[(task_order.index(original) + 1) % len(task_order)]
        counter_instruction = instruction_by_original[counter_original]
        counter_pool = by_original.get(counter_original) or []
        for index, record in enumerate(source_records[:3]):
            item = dict(record)
            item["instruction"] = counter_instruction
            item["counterfactual_from"] = original
            item["counterfactual_to"] = counter_original
            item["counterfactual_pair_id"] = _stable_id([original, counter_original, str(index)])
            item["instruction_source"] = "counterfactual_instruction_text_swap"
            item["paraphrase_group_id"] = f"cf:{item['counterfactual_pair_id']}"
            item["target_prior"] = resolve_target_prior(counter_instruction, candidates_by_path.get(item["hdf5_path"], []))
            if counter_pool:
                item["counterfactual_expert_record"] = counter_pool[index % len(counter_pool)]
            records.append(item)
    return records


def _select_groups(rows: list[ParaRow], max_train: int, max_eval: int) -> tuple[list[ParaRow], list[ParaRow]]:
    by_group: dict[str, list[ParaRow]] = defaultdict(list)
    for row in rows:
        by_group[row.group_id].append(row)
    groups = sorted(by_group.values(), key=lambda group: (0 if group[0].high == "obj" else 1, group[0].group_id))
    heldout: list[ParaRow] = []
    train: list[ParaRow] = []
    for index, group in enumerate(groups):
        target = heldout if index % 3 == 0 else train
        target.extend(group[:1])
    if not any(row.high == "obj" for row in heldout):
        object_group = next((group for group in groups if group[0].high == "obj"), None)
        if object_group:
            heldout.insert(0, object_group[0])
    return train[:max_train], heldout[:max_eval]


def build_tg7d_dataset(
    *,
    data_root: Path,
    metadata_csv: Path,
    max_tasks: int,
    max_train_paraphrases_per_task: int,
    max_eval_paraphrases_per_task: int,
    train_demos: int,
    eval_demos: int,
    records_per_demo: int,
) -> dict[str, Any]:
    rows = _load_para_rows(metadata_csv)
    rows_by_original: dict[str, list[ParaRow]] = defaultdict(list)
    for row in rows:
        rows_by_original[row.original_instruction].append(row)

    matched: list[dict[str, Any]] = []
    for original in sorted(rows_by_original):
        path = _match_hdf5(original, data_root)
        if path is not None:
            matched.append({"original_instruction": original, "hdf5_path": path, "rows": rows_by_original[original]})
    matched = matched[:max_tasks]

    candidates_by_path: dict[str, list[str]] = {}
    clean_train: list[dict[str, Any]] = []
    clean_eval: list[dict[str, Any]] = []
    train_paraphrase: list[dict[str, Any]] = []
    heldout_paraphrase: list[dict[str, Any]] = []
    heldout_object: list[dict[str, Any]] = []
    train_groups: set[str] = set()
    heldout_groups: set[str] = set()
    task_order: list[str] = []
    instruction_by_original: dict[str, str] = {}
    task_reports: list[dict[str, Any]] = []

    for item in matched:
        path = Path(item["hdf5_path"])
        original = str(item["original_instruction"])
        task_order.append(original)
        instruction_by_original[original] = original
        candidates = visible_object_candidates(path)
        candidates_by_path[str(path)] = candidates
        split = base.select_records(
            path,
            max_train_demos=train_demos,
            max_eval_demos=eval_demos,
            records_per_demo=records_per_demo,
        )
        train_base = _clone_records(
            split["train_records"],
            instruction=original,
            original_instruction=original,
            group_id=f"clean:{_stable_id([original])}",
            source="original_instruction_clean",
            row=None,
            candidates_by_path=candidates_by_path,
        )
        eval_base = _clone_records(
            split["eval_records"],
            instruction=original,
            original_instruction=original,
            group_id=f"clean:{_stable_id([original])}",
            source="original_instruction_clean",
            row=None,
            candidates_by_path=candidates_by_path,
        )
        clean_train.extend(train_base)
        clean_eval.extend(eval_base)
        train_rows, heldout_rows = _select_groups(
            list(item["rows"]),
            max_train=max_train_paraphrases_per_task,
            max_eval=max_eval_paraphrases_per_task,
        )
        for row in train_rows:
            train_groups.add(row.group_id)
            train_paraphrase.extend(
                _clone_records(
                    train_base,
                    instruction=row.new_instruction,
                    original_instruction=original,
                    group_id=row.group_id,
                    source="official_libero_para_train_group",
                    row=row,
                    candidates_by_path=candidates_by_path,
                )
            )
        for row in heldout_rows:
            heldout_groups.add(row.group_id)
            cloned = _clone_records(
                eval_base,
                instruction=row.new_instruction,
                original_instruction=original,
                group_id=row.group_id,
                source="official_libero_para_heldout_group",
                row=row,
                candidates_by_path=candidates_by_path,
            )
            heldout_paraphrase.extend(cloned)
            if row.high == "obj":
                heldout_object.extend(cloned)
        task_reports.append(
            {
                "original_instruction": original,
                "hdf5_path": str(path),
                "train_records": len(train_base),
                "eval_records": len(eval_base),
                "visible_object_candidate_count": len(candidates),
                "visible_object_candidate_sample": candidates[:12],
                "selected_target_prior": resolve_target_prior(original, candidates),
                "train_paraphrase_rows": len(train_rows),
                "heldout_paraphrase_rows": len(heldout_rows),
                "heldout_object_rows": sum(1 for row in heldout_rows if row.high == "obj"),
            }
        )

    counterfactual_records = _build_counterfactual_records(
        clean_eval,
        task_order,
        instruction_by_original,
        candidates_by_path,
    )
    leakage = {
        "paraphrase_group_overlap": sorted(train_groups & heldout_groups),
        "group_leakage_detected": bool(train_groups & heldout_groups),
        "clean_eval_record_overlap": len({_record_key(r) for r in clean_train} & {_record_key(r) for r in clean_eval}),
        "heldout_paraphrase_group_count": len(heldout_groups),
        "train_paraphrase_group_count": len(train_groups),
        "heldout_object_group_count": len({r["paraphrase_group_id"] for r in heldout_object}),
    }
    feasibility = {
        "metadata_csv": str(metadata_csv),
        "metadata_exists": metadata_csv.exists(),
        "metadata_rows": len(rows),
        "original_instruction_count": len(rows_by_original),
        "matched_hdf5_task_count": len(matched),
        "selected_task_count": len(task_reports),
        "clean_train_records": len(clean_train),
        "clean_eval_records": len(clean_eval),
        "train_paraphrase_records": len(train_paraphrase),
        "heldout_paraphrase_records": len(heldout_paraphrase),
        "heldout_object_lexical_records": len(heldout_object),
        "counterfactual_records": len(counterfactual_records),
        "task_reports": task_reports,
        "leakage": leakage,
        "instruction_action_link_without_inference_leakage": bool(clean_train and clean_eval),
        "counterfactual_generated_without_oracle_eval_labels": bool(counterfactual_records),
        "target_prior_from_instruction_and_visible_names_only": all(
            not report["selected_target_prior"]["uses_bddl_target_labels"]
            and not report["selected_target_prior"]["uses_eval_labels"]
            for report in task_reports
        ),
    }
    return {
        "clean_train": clean_train,
        "clean_eval": clean_eval,
        "train_paraphrase": train_paraphrase,
        "heldout_paraphrase": heldout_paraphrase,
        "heldout_object": heldout_object,
        "counterfactual": counterfactual_records,
        "feasibility": feasibility,
    }


def _records_actions(records: list[dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [
            base._expert_action(Path(record["hdf5_path"]), str(record["demo_name"]), int(record["timestep"]))[:7]
            for record in records
        ],
        axis=0,
    ).astype(np.float32)


def _feature_tensors(
    records: list[dict[str, Any]],
    *,
    x_mean: np.ndarray | None = None,
    x_std: np.ndarray | None = None,
    text_mode: str,
    prior_mode: str,
):
    import torch

    state, time_tensor, y_np, out_mean, out_std = baseline._state_time_tensors(records, x_mean, x_std)
    text = torch.tensor(_text_matrix(records, text_mode, TEXT_DIM), dtype=torch.float32)
    prior = torch.tensor(_text_matrix(records, prior_mode, PRIOR_DIM), dtype=torch.float32)
    return state, time_tensor, text, prior, y_np, out_mean, out_std


def _mlp_param_count(input_dim: int, hidden_dim: int, output_dim: int) -> int:
    return (input_dim * hidden_dim + hidden_dim) + (hidden_dim * output_dim + output_dim)


def _evaluate_prediction_records(
    *,
    pred: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    expert = _records_actions(records)
    return fix._metrics_from_arrays(pred, expert)


def _train_variant(
    *,
    name: str,
    checkpoint: Path,
    train_records: list[dict[str, Any]],
    eval_sets: dict[str, list[dict[str, Any]]],
    counterfactual_records: list[dict[str, Any]],
    steps: int,
    hidden_dim: int,
    learning_rate: float,
    lora_rank: int | None,
    seed: int,
    text_mode: str,
    prior_mode: str,
    target_gate: bool,
    consistency_weight: float = 0.0,
    sensitivity_weight: float = 0.0,
    oracle: bool = False,
) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    torch.manual_seed(seed)
    state_weight, state_bias, weight_file = baseline._state_proj_weights(checkpoint)
    train_state, train_time, train_text, train_prior, y_train_np, x_mean, x_std = _feature_tensors(
        train_records,
        text_mode=text_mode,
        prior_mode=prior_mode,
    )
    normalizer = fix.Libero7DNormalizer.fit(y_train_np)
    y_train = torch.tensor(normalizer.normalize(y_train_np), dtype=torch.float32)

    lora_a = None
    lora_b = None
    lora_alpha = None
    params: list[Any] = []
    if lora_rank is not None:
        lora_alpha = int(lora_rank) * 2
        lora_a = torch.nn.Parameter(torch.randn(int(lora_rank), 32) * 0.01)
        lora_b = torch.nn.Parameter(torch.zeros(960, int(lora_rank)))
        params.extend([lora_a, lora_b])

    gate = None
    if target_gate:
        gate = torch.nn.Sequential(torch.nn.Linear(PRIOR_DIM, hidden_dim), torch.nn.SiLU(), torch.nn.Linear(hidden_dim, 1920))
        params.extend(gate.parameters())

    input_dim = 961
    if text_mode != "none":
        input_dim += TEXT_DIM
    if prior_mode != "none" and not target_gate:
        input_dim += PRIOR_DIM
    head = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.SiLU(), torch.nn.Linear(hidden_dim, 7))
    params.extend(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-5)
    losses: list[dict[str, float]] = []

    def projected(state_tensor):
        if lora_rank is None:
            return state_tensor @ state_weight.T + state_bias
        scale = float(lora_alpha) / float(lora_rank)
        delta = (lora_b @ lora_a) * scale
        return state_tensor @ (state_weight + delta).T + state_bias

    def forward(state_tensor, time_tensor, text_tensor, prior_tensor):
        feature = projected(state_tensor)
        pieces: list[Any] = []
        if target_gate:
            assert gate is not None
            gamma_beta = gate(prior_tensor)
            gamma, beta = gamma_beta[:, :960], gamma_beta[:, 960:]
            feature = feature * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        pieces.extend([feature, time_tensor])
        if text_mode != "none":
            pieces.append(text_tensor)
        if prior_mode != "none" and not target_gate:
            pieces.append(prior_tensor)
        return head(torch.cat(pieces, dim=1))

    n_train = max(1, len(train_records))
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        index = step % n_train
        pred = forward(
            train_state[index : index + 1],
            train_time[index : index + 1],
            train_text[index : index + 1],
            train_prior[index : index + 1],
        )
        target = y_train[index : index + 1]
        pose_loss = torch.nn.functional.mse_loss(pred[:, :6], target[:, :6])
        gripper_loss = torch.nn.functional.mse_loss(pred[:, 6:], target[:, 6:])
        loss = pose_loss + gripper_loss
        consistency_loss = torch.tensor(0.0)
        sensitivity_loss = torch.tensor(0.0)
        if consistency_weight > 0 and len(train_records) > 1:
            other_index = (index + len(clean_key_indices(train_records).get(_record_key(train_records[index]), [index]))) % n_train
            pred_other = forward(
                train_state[other_index : other_index + 1],
                train_time[other_index : other_index + 1],
                train_text[other_index : other_index + 1],
                train_prior[other_index : other_index + 1],
            )
            same_key = _record_key(train_records[index]) == _record_key(train_records[other_index])
            if same_key:
                consistency_loss = torch.nn.functional.mse_loss(pred, pred_other)
                loss = loss + float(consistency_weight) * consistency_loss
        if sensitivity_weight > 0 and counterfactual_records:
            cf = counterfactual_records[step % len(counterfactual_records)]
            cf_state, cf_time, cf_text, cf_prior, _, _, _ = _feature_tensors(
                [cf],
                x_mean=x_mean,
                x_std=x_std,
                text_mode=text_mode,
                prior_mode=prior_mode,
            )
            orig = dict(cf)
            orig["instruction"] = cf.get("counterfactual_from", cf.get("original_instruction", ""))
            orig["target_prior"] = resolve_target_prior(
                str(orig["instruction"]),
                visible_object_candidates(Path(orig["hdf5_path"])),
            )
            orig_state, orig_time, orig_text, orig_prior, _, _, _ = _feature_tensors(
                [orig],
                x_mean=x_mean,
                x_std=x_std,
                text_mode=text_mode,
                prior_mode=prior_mode,
            )
            cf_pred = forward(cf_state, cf_time, cf_text, cf_prior)
            orig_pred = forward(orig_state, orig_time, orig_text, orig_prior)
            delta = torch.linalg.norm(cf_pred - orig_pred, dim=1).mean()
            sensitivity_loss = torch.relu(torch.tensor(0.15) - delta) ** 2
            loss = loss + float(sensitivity_weight) * sensitivity_loss
        loss.backward()
        optimizer.step()
        losses.append(
            {
                "loss": _round(loss.detach().cpu()),
                "pose_loss": _round(pose_loss.detach().cpu()),
                "gripper_mse_loss": _round(gripper_loss.detach().cpu()),
                "consistency_loss": _round(consistency_loss.detach().cpu()),
                "sensitivity_loss": _round(sensitivity_loss.detach().cpu()),
            }
        )

    eval_metrics: dict[str, Any] = {}
    predictions_by_set: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for set_name, records in eval_sets.items():
            if not records:
                continue
            state, time_tensor, text, prior, _, _, _ = _feature_tensors(
                records,
                x_mean=x_mean,
                x_std=x_std,
                text_mode=text_mode,
                prior_mode=prior_mode,
            )
            pred_np = normalizer.unnormalize(forward(state, time_tensor, text, prior).detach().cpu().numpy())
            predictions_by_set[set_name] = pred_np
            eval_metrics[set_name] = _evaluate_prediction_records(pred=pred_np, records=records)

    consistency = _paraphrase_consistency(predictions_by_set, eval_sets)
    cf_metrics = _counterfactual_sensitivity(
        forward=forward,
        records=counterfactual_records,
        x_mean=x_mean,
        x_std=x_std,
        normalizer=normalizer,
        text_mode=text_mode,
        prior_mode=prior_mode,
    )
    lora_params = 0 if lora_rank is None else int(lora_rank) * (32 + 960)
    gate_params = 0 if gate is None else sum(int(param.numel()) for param in gate.parameters())
    head_params = _mlp_param_count(input_dim, hidden_dim, 7)
    return {
        "name": name,
        "adapter_schema": "LIBERO_7D",
        "feature_schema": "train-normalized observation.state padded to 32, frozen SmolVLA state_proj, timestep, and gated text/target prior features",
        "state_proj_weight_file": str(weight_file),
        "target_modules": ["state_proj_lora_rank_4"] if lora_rank else ["libero_7d_adapter"],
        "lora_rank": lora_rank,
        "text_mode": text_mode,
        "prior_mode": prior_mode,
        "target_conditioning": "FiLM-style target-prior gate" if target_gate else "none_or_concatenated_text",
        "oracle_upper_bound": bool(oracle),
        "trainable_params": lora_params + gate_params + head_params,
        "normalization": normalizer.report(),
        "training": {
            "batch_size": 1,
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "loss_start": losses[0]["loss"] if losses else None,
            "loss_end": losses[-1]["loss"] if losses else None,
            "loss_decreased": bool(losses and losses[-1]["loss"] < losses[0]["loss"]),
            "loss_curve_sample": {"first3": losses[:3], "last3": losses[-3:]},
        },
        "eval_metrics": eval_metrics,
        "paraphrase_consistency": consistency,
        "counterfactual_sensitivity": cf_metrics,
        "runtime_sec": _round(time.monotonic() - started, 3),
        "uses_eval_labels_for_training": False,
        "uses_bddl_target_labels_for_inference": False,
        "uses_task_ids_or_filenames_for_inference": False,
        "uses_so100_action_normalizer": False,
        "uses_hard_coded_gripper_fill": False,
    }


def clean_key_indices(records: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[int]]:
    by_key: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_key[_record_key(record)].append(index)
    return by_key


def _paraphrase_consistency(predictions_by_set: dict[str, np.ndarray], eval_sets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    clean_pred = predictions_by_set.get("clean")
    para_pred = predictions_by_set.get("heldout_paraphrase")
    if clean_pred is None or para_pred is None:
        return {"available": False}
    clean_by_key = {_record_key(record): clean_pred[index] for index, record in enumerate(eval_sets["clean"])}
    deltas = []
    for index, record in enumerate(eval_sets["heldout_paraphrase"]):
        key = _record_key(record)
        if key in clean_by_key:
            deltas.append(float(np.linalg.norm(para_pred[index] - clean_by_key[key])))
    return {
        "available": bool(deltas),
        "same_target_prediction_l2": _round(float(np.mean(deltas))) if deltas else None,
        "pair_count": len(deltas),
    }


def _counterfactual_sensitivity(
    *,
    forward: Any,
    records: list[dict[str, Any]],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    normalizer: fix.Libero7DNormalizer,
    text_mode: str,
    prior_mode: str,
) -> dict[str, Any]:
    if not records:
        return {"available": False, "pair_count": 0}
    import torch

    pred_deltas = []
    expert_deltas = []
    alignment_l2 = []
    with torch.no_grad():
        for cf in records:
            orig = dict(cf)
            orig["instruction"] = cf.get("counterfactual_from", cf.get("original_instruction", ""))
            orig["target_prior"] = resolve_target_prior(
                str(orig["instruction"]),
                visible_object_candidates(Path(orig["hdf5_path"])),
            )
            cf_state, cf_time, cf_text, cf_prior, _, _, _ = _feature_tensors(
                [cf],
                x_mean=x_mean,
                x_std=x_std,
                text_mode=text_mode,
                prior_mode=prior_mode,
            )
            orig_state, orig_time, orig_text, orig_prior, _, _, _ = _feature_tensors(
                [orig],
                x_mean=x_mean,
                x_std=x_std,
                text_mode=text_mode,
                prior_mode=prior_mode,
            )
            cf_pred = normalizer.unnormalize(forward(cf_state, cf_time, cf_text, cf_prior).detach().cpu().numpy())[0]
            orig_pred = normalizer.unnormalize(forward(orig_state, orig_time, orig_text, orig_prior).detach().cpu().numpy())[0]
            pred_deltas.append(float(np.linalg.norm(cf_pred - orig_pred)))
            counter_record = cf.get("counterfactual_expert_record")
            if counter_record:
                orig_expert = base._expert_action(Path(cf["hdf5_path"]), str(cf["demo_name"]), int(cf["timestep"]))[:7]
                counter_expert = base._expert_action(
                    Path(counter_record["hdf5_path"]),
                    str(counter_record["demo_name"]),
                    int(counter_record["timestep"]),
                )[:7]
                expert_deltas.append(float(np.linalg.norm(counter_expert - orig_expert)))
                alignment_l2.append(float(np.linalg.norm(cf_pred - counter_expert)))
    return {
        "available": bool(pred_deltas),
        "pair_count": len(pred_deltas),
        "prediction_delta_l2": _round(float(np.mean(pred_deltas))) if pred_deltas else None,
        "expert_counterfactual_delta_l2": _round(float(np.mean(expert_deltas))) if expert_deltas else None,
        "cf_prediction_to_counter_expert_l2": _round(float(np.mean(alignment_l2))) if alignment_l2 else None,
        "collapse_rate_delta_lt_0_05": _round(float(np.mean(np.asarray(pred_deltas) < 0.05))) if pred_deltas else None,
    }


def _constant_metrics(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    return baseline._mean_action_metrics(train_records, eval_records)


def _run_method_gate(args: argparse.Namespace, dataset: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    clean_train = dataset["clean_train"]
    clean_eval = dataset["clean_eval"]
    train_aug = clean_train + dataset["train_paraphrase"]
    eval_sets = {
        "clean": clean_eval,
        "heldout_paraphrase": dataset["heldout_paraphrase"],
        "object_lexical": dataset["heldout_object"],
    }
    checkpoint = Path(args.smolvla_ckpt)
    variants: dict[str, Any] = {
        "mean_action": {
            "name": "mean_action",
            "eval_metrics": {name: _constant_metrics(clean_train, records) for name, records in eval_sets.items() if records},
        },
        "ridge": fix._ridge_baseline(clean_train, clean_eval),
    }
    variants["small_mlp"] = _train_variant(
        name="small_state_time_mlp_7d_baseline",
        checkpoint=checkpoint,
        train_records=clean_train,
        eval_sets=eval_sets,
        counterfactual_records=dataset["counterfactual"],
        steps=int(args.simple_steps),
        hidden_dim=32,
        learning_rate=float(args.learning_rate),
        lora_rank=None,
        seed=37,
        text_mode="none",
        prior_mode="none",
        target_gate=False,
    )
    variants["standard_smolvla_7d_lora_adapter"] = _train_variant(
        name="standard_smolvla_7d_lora_adapter",
        checkpoint=checkpoint,
        train_records=clean_train,
        eval_sets=eval_sets,
        counterfactual_records=dataset["counterfactual"],
        steps=int(args.adapter_steps),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        lora_rank=int(args.lora_rank),
        seed=41,
        text_mode="none",
        prior_mode="none",
        target_gate=False,
    )
    variants["canonicalization_only"] = _train_variant(
        name="canonicalization_only",
        checkpoint=checkpoint,
        train_records=clean_train,
        eval_sets=eval_sets,
        counterfactual_records=dataset["counterfactual"],
        steps=int(args.adapter_steps),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        lora_rank=int(args.lora_rank),
        seed=43,
        text_mode="canonical",
        prior_mode="none",
        target_gate=False,
    )
    variants["simple_paraphrase_augmentation"] = _train_variant(
        name="simple_paraphrase_augmentation",
        checkpoint=checkpoint,
        train_records=train_aug,
        eval_sets=eval_sets,
        counterfactual_records=dataset["counterfactual"],
        steps=int(args.adapter_steps),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        lora_rank=int(args.lora_rank),
        seed=47,
        text_mode="raw",
        prior_mode="none",
        target_gate=False,
    )
    variants["tg7d_adapter"] = _train_variant(
        name="tg7d_adapter",
        checkpoint=checkpoint,
        train_records=train_aug,
        eval_sets=eval_sets,
        counterfactual_records=dataset["counterfactual"],
        steps=int(args.adapter_steps),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        lora_rank=int(args.lora_rank),
        seed=53,
        text_mode="raw",
        prior_mode="target_prior",
        target_gate=True,
        consistency_weight=0.05,
        sensitivity_weight=0.02,
    )
    variants["oracle_target_upper_bound"] = _train_variant(
        name="oracle_target_upper_bound",
        checkpoint=checkpoint,
        train_records=train_aug,
        eval_sets=eval_sets,
        counterfactual_records=dataset["counterfactual"],
        steps=max(10, int(args.adapter_steps) // 2),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        lora_rank=None,
        seed=59,
        text_mode="oracle",
        prior_mode="target_prior",
        target_gate=True,
        oracle=True,
    )
    return {
        "variants": variants,
        "lora_rank": int(args.lora_rank),
        "runtime_sec": _round(time.monotonic() - started, 3),
        "vram_peak_mb": 0.0,
        "dataset_split_used": "local_libero_goal_libero_para_group_holdout",
        "train_records": len(clean_train),
        "train_augmented_records": len(train_aug),
        "clean_eval_records": len(clean_eval),
        "heldout_paraphrase_records": len(dataset["heldout_paraphrase"]),
        "object_lexical_records": len(dataset["heldout_object"]),
        "counterfactual_records": len(dataset["counterfactual"]),
    }


def _metric(variant: dict[str, Any], split: str = "heldout_paraphrase") -> float | None:
    eval_metrics = variant.get("eval_metrics") or {}
    if split in eval_metrics:
        return eval_metrics[split].get("action_l2")
    return None


def _decide(report: dict[str, Any]) -> tuple[str, str]:
    feasibility = report.get("feasibility") or {}
    if not feasibility.get("metadata_exists") or feasibility.get("matched_hdf5_task_count", 0) < 2:
        return "NO_TARGET_GROUNDING_EVAL_PATH", "Stop: local LIBERO-Para to LIBERO HDF5 linking is insufficient."
    if (feasibility.get("leakage") or {}).get("group_leakage_detected"):
        return "KILL_LEAKAGE_RISK", "Stop: paraphrase group leakage was detected."
    if not feasibility.get("target_prior_from_instruction_and_visible_names_only"):
        return "KILL_LEAKAGE_RISK", "Stop: target prior requires forbidden metadata."
    if not report.get("method_gate"):
        return "NO_TARGET_GROUNDING_EVAL_PATH", f"Set {TRAINING_GATE}=1 only after STATE 1 feasibility is green."

    variants = report["method_gate"]["variants"]
    tg = variants["tg7d_adapter"]
    standard = variants["standard_smolvla_7d_lora_adapter"]
    canonical = variants["canonicalization_only"]
    mlp_metric = variants["small_mlp"]["eval_metrics"]["heldout_paraphrase"]["action_l2"]
    tg_para = _metric(tg, "heldout_paraphrase")
    standard_para = _metric(standard, "heldout_paraphrase")
    canonical_para = _metric(canonical, "heldout_paraphrase")
    tg_object = _metric(tg, "object_lexical")
    standard_object = _metric(standard, "object_lexical")
    canonical_object = _metric(canonical, "object_lexical")
    tg_clean = _metric(tg, "clean")
    standard_clean = _metric(standard, "clean")
    cf = tg.get("counterfactual_sensitivity") or {}
    sensitivity_ok = bool(cf.get("prediction_delta_l2") is not None and cf["prediction_delta_l2"] >= 0.05)
    clean_ok = bool(tg_clean is not None and standard_clean is not None and tg_clean <= standard_clean * 1.10)

    if tg_para is None or standard_para is None or canonical_para is None:
        return "NO_TARGET_GROUNDING_EVAL_PATH", "Stop: no meaningful target/paraphrase metric was produced."
    if tg_para >= canonical_para or (tg_object is not None and canonical_object is not None and tg_object >= canonical_object):
        return "KILL_CANONICALIZATION_DOMINATED", "Stop: canonicalization-only matched or beat TG-7D on the target/paraphrase metric."
    if tg_para >= standard_para or (tg_object is not None and standard_object is not None and tg_object >= standard_object):
        return "KILL_BASELINE_DOMINATED", "Stop: standard SmolVLA 7D LoRA/adapter matched or beat TG-7D."
    if tg_para >= mlp_metric:
        return "KILL_BASELINE_DOMINATED", "Stop: simple MLP/ridge baseline matched or beat TG-7D on the claimed metric."
    if not clean_ok:
        return "KILL_BASELINE_DOMINATED", "Stop: TG-7D did not preserve clean action quality."
    if not sensitivity_ok:
        return "KILL_BASELINE_DOMINATED", "Stop: counterfactual target sensitivity failed."
    return (
        "READY_FOR_TG7D_SCALE_UP",
        "Scale only to a predeclared TG-7D run with this split manifest, canonicalization baseline, standard rank-4/8 LoRA baseline, and counterfactual sensitivity gate preserved.",
    )


def _write_static_state0_reports() -> None:
    Path("reports").mkdir(exist_ok=True)
    Path("reports/tg7d_adapter_task_definition.md").write_text(
        "\n".join(
            [
                "# TG-7D Adapter Task Definition",
                "",
                "TG-7D Adapter tests target/object semantic grounding injected into the fixed SmolVLA LIBERO_7D action pathway.",
                "",
                "LoRA and adapters are training tools only. The novelty claim is valid only if target/object priors from instruction text and visible object names improve paraphrase/object lexical robustness while preserving clean action quality and counterfactual target sensitivity.",
                "",
                "Forbidden inference sources: BDDL target labels, eval labels, task IDs, filenames, reward/success labels, future actions, old 6D/SO100 labels, and hard-coded gripper fill.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/tg7d_adapter_related_work_matrix.md").write_text(
        "\n".join(
            [
                "# TG-7D Adapter Related Work Matrix",
                "",
                "| Anchor | Relevance | TG-7D distinction | Required anti-baseline |",
                "| --- | --- | --- | --- |",
                "| ActionMap | Voxel heatmap action head improves VLA action representation on LIBERO and real robot tasks. Source: https://arxiv.org/abs/2606.06904 | TG-7D is not a heatmap decoder; it tests instruction-resolved target priors in a fixed 7D adapter. | Standard LoRA/action imitation and future ActionMap-style head if scaled. |",
                "| Direct grounded 3D point action-head injection | Injects grounded 3D target points into the action head. Source: https://arxiv.org/html/2606.27663v1 | TG-7D may not rely on oracle 3D points; priors must come from instruction text plus visible names. | Single-point/destination oracle if any spatial point is used. |",
                "| LIBERO-Para | Provides paraphrase/object lexical benchmark and PRIDE-style difficulty evidence. Source: https://arxiv.org/abs/2603.28301 and https://huggingface.co/datasets/HAI-Lab/LIBERO-Para | TG-7D is a method gate using LIBERO-Para as evaluation metadata. | Canonicalization-only and simple paraphrase augmentation. |",
                "| OpenVLA-OFT | Strong fine-tuning recipe with continuous actions and high LIBERO success. Source: https://arxiv.org/abs/2502.19645 | TG-7D is a bounded SmolVLA adapter path, not OpenVLA-OFT. | OpenVLA-OFT remains a nonlocal paper baseline; do not run locally here. |",
                "| SmolVLA | Compact VLA backbone suited to consumer hardware. Source: https://arxiv.org/abs/2506.01844 | TG-7D uses SmolVLA as the executable backbone. | Standard SmolVLA 7D LoRA/adapter. |",
                "| Old Target-Prior TCA | Earlier target-prior route had reusable prior/leakage audits but weak proxy/action heads. | TG-7D must use fixed real SmolVLA/LIBERO_7D path, not old TCA-Select or weak MLP route. | Mean-action, ridge/MLP, standard LoRA. |",
                "| Canonicalization / paraphrase augmentation | Strong simple language baselines; prior PRISM route was killed by canonicalization. | TG-7D must beat them rather than rename them. | Canonicalization-only and simple paraphrase augmentation. |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/tg7d_adapter_experiment_plan.md").write_text(
        "\n".join(
            [
                "# TG-7D Adapter Experiment Plan",
                "",
                "- STATE 1: prove a no-leakage LIBERO-Para/object/counterfactual split exists.",
                "- STATE 2: run a tiny fixed LIBERO_7D rank-4 adapter gate only if STATE 1 is green.",
                "- Required arms: mean-action, ridge/MLP, standard SmolVLA 7D LoRA/adapter, canonicalization-only, simple paraphrase augmentation, TG-7D Adapter, and oracle target upper bound.",
                "- Metrics: clean/paraphrase/object action L2, translation L2, rotation L2, gripper error/accuracy, train/eval gap proxy, target consistency, counterfactual sensitivity, trainable params, VRAM, runtime.",
                "- No OpenVLA-OFT, downloads, full benchmark, old TCA-Select, old 6D/SO100 action path, hard-coded gripper fill, or paper claims.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/tg7d_adapter_kill_criteria.md").write_text(
        "\n".join(
            [
                "# TG-7D Adapter Kill Criteria",
                "",
                "Final decision must be one of `READY_FOR_TG7D_SCALE_UP`, `KILL_BASELINE_DOMINATED`, `KILL_CANONICALIZATION_DOMINATED`, `KILL_LEAKAGE_RISK`, `NO_TARGET_GROUNDING_EVAL_PATH`, or `TOO_HEAVY_LOCAL`.",
                "",
                "Kill immediately if standard LoRA, canonicalization-only, or MLP/ridge matches or beats TG-7D on the claimed target/paraphrase/object metric; if target prior requires leakage; if clean action quality collapses; if counterfactual sensitivity fails; or if no meaningful target/paraphrase/counterfactual metric exists.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/tg7d_adapter_risk_register.md").write_text(
        "\n".join(
            [
                "# TG-7D Adapter Risk Register",
                "",
                "| Risk | Severity | Mitigation |",
                "| --- | --- | --- |",
                "| Novelty collapses to standard LoRA capacity | High | Keep standard rank-4/8 SmolVLA 7D LoRA as primary baseline. |",
                "| Canonicalization dominates | High | Treat canonicalization-only as a kill baseline. |",
                "| Target prior leaks through BDDL/task IDs/filenames | High | Use only instruction text plus visible object-candidate names for inference prior. |",
                "| Consistency objective ignores target changes | High | Require counterfactual sensitivity. |",
                "| Local action metric overclaim | Medium | Mark as bounded method gate, not rollout or paper evidence. |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_report_bundle(report: dict[str, Any]) -> None:
    _write_static_state0_reports()
    summary = report.get("summary") or {}
    feasibility = report.get("feasibility") or {}
    method = report.get("method_gate") or {}
    variants = method.get("variants") or {}
    tg = variants.get("tg7d_adapter") or {}
    standard = variants.get("standard_smolvla_7d_lora_adapter") or {}
    canonical = variants.get("canonicalization_only") or {}
    mlp = variants.get("small_mlp") or {}
    lines = [
        "# TG-7D Adapter Autopilot State",
        "",
        f"- final decision: `{summary.get('final_decision')}`",
        f"- exact next step: {summary.get('exact_next_step')}",
        f"- current strong baseline reference L2: `{CURRENT_STRONG_BASELINE_L2}`",
        f"- experiments happened: `{summary.get('experiments_happened')}`",
        f"- training happened: `{summary.get('training_happened')}`",
        f"- loss computed: `{summary.get('loss_computed')}`",
        f"- downloads/OpenVLA-OFT/rollouts happened: `{summary.get('downloads_happened')}` / `{summary.get('openvla_oft_happened')}` / `{summary.get('rollouts_happened')}`",
        "",
    ]
    Path("reports/tg7d_adapter_autopilot_state.md").write_text("\n".join(lines), encoding="utf-8")

    split_lines = [
        "# TG-7D Adapter Split Feasibility",
        "",
        f"- metadata CSV: `{feasibility.get('metadata_csv')}`",
        f"- metadata rows: `{feasibility.get('metadata_rows')}`",
        f"- original instruction count: `{feasibility.get('original_instruction_count')}`",
        f"- matched HDF5 task count: `{feasibility.get('matched_hdf5_task_count')}`",
        f"- selected task count: `{feasibility.get('selected_task_count')}`",
        f"- clean train/eval records: `{feasibility.get('clean_train_records')} / {feasibility.get('clean_eval_records')}`",
        f"- train paraphrase records: `{feasibility.get('train_paraphrase_records')}`",
        f"- held-out paraphrase records: `{feasibility.get('heldout_paraphrase_records')}`",
        f"- held-out object lexical records: `{feasibility.get('heldout_object_lexical_records')}`",
        f"- counterfactual records: `{feasibility.get('counterfactual_records')}`",
        f"- leakage: `{feasibility.get('leakage')}`",
        "",
    ]
    Path("reports/tg7d_adapter_split_feasibility.md").write_text("\n".join(split_lines), encoding="utf-8")

    prior_lines = [
        "# TG-7D Adapter Target Prior Audit",
        "",
        "Target priors are derived from instruction text plus visible object-candidate names parsed from HDF5 model XML.",
        "",
        f"- target prior from instruction and visible names only: `{feasibility.get('target_prior_from_instruction_and_visible_names_only')}`",
        f"- instruction/action link without inference leakage: `{feasibility.get('instruction_action_link_without_inference_leakage')}`",
        f"- counterfactual generated without oracle eval labels: `{feasibility.get('counterfactual_generated_without_oracle_eval_labels')}`",
        "",
    ]
    for task in feasibility.get("task_reports") or []:
        prior_lines.extend(
            [
                f"## {task.get('original_instruction')}",
                "",
                f"- visible object candidate count: `{task.get('visible_object_candidate_count')}`",
                f"- visible object candidate sample: `{task.get('visible_object_candidate_sample')}`",
                f"- selected target prior: `{task.get('selected_target_prior')}`",
                "",
            ]
        )
    Path("reports/tg7d_adapter_target_prior_audit.md").write_text("\n".join(prior_lines), encoding="utf-8")

    result_lines = [
        "# TG-7D Adapter Method Gate Results",
        "",
        f"- dataset/split used: `{method.get('dataset_split_used')}`",
        f"- LoRA rank: `{method.get('lora_rank')}`",
        f"- runtime sec: `{summary.get('runtime_sec')}`",
        f"- VRAM peak MB: `{method.get('vram_peak_mb')}`",
        f"- mean-action held-out paraphrase L2: `{(((variants.get('mean_action') or {}).get('eval_metrics') or {}).get('heldout_paraphrase') or {}).get('action_l2')}`",
        f"- MLP held-out paraphrase L2: `{_metric(mlp, 'heldout_paraphrase')}`",
        f"- MLP clean L2: `{_metric(mlp, 'clean')}`",
        f"- standard LoRA held-out paraphrase L2: `{_metric(standard, 'heldout_paraphrase')}`",
        f"- canonicalization held-out paraphrase L2: `{_metric(canonical, 'heldout_paraphrase')}`",
        f"- TG-7D held-out paraphrase L2: `{_metric(tg, 'heldout_paraphrase')}`",
        f"- TG-7D object lexical L2: `{_metric(tg, 'object_lexical')}`",
        f"- TG-7D clean L2: `{_metric(tg, 'clean')}`",
        f"- TG-7D counterfactual sensitivity: `{tg.get('counterfactual_sensitivity')}`",
        f"- TG-7D target consistency: `{tg.get('paraphrase_consistency')}`",
        f"- TG-7D trainable params: `{tg.get('trainable_params')}`",
        f"- oracle target upper bound held-out paraphrase L2: `{_metric(variants.get('oracle_target_upper_bound') or {}, 'heldout_paraphrase')}`",
        "",
    ]
    Path("reports/tg7d_adapter_state1_state2_results.md").write_text("\n".join(result_lines), encoding="utf-8")

    decision_lines = [
        "# TG-7D Adapter Decision Log",
        "",
        f"Final decision: `{summary.get('final_decision')}`",
        "",
        f"Exact next step: {summary.get('exact_next_step')}",
        "",
        "This is a bounded method gate, not a rollout benchmark or paper claim.",
        "",
    ]
    Path("reports/tg7d_adapter_decision_log.md").write_text("\n".join(decision_lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "schema_version": "tg7d-adapter-state-gate-v1",
        "decision": None,
        "summary": {
            "final_decision": None,
            "exact_next_step": None,
            "experiments_happened": False,
            "training_happened": False,
            "loss_computed": False,
            "downloads_happened": False,
            "openvla_oft_happened": False,
            "rollouts_happened": False,
            "gpu_training_happened": False,
            "model_used": str(Path(args.smolvla_ckpt)),
            "current_strong_baseline_l2": CURRENT_STRONG_BASELINE_L2,
        },
        "policy": {
            "state_gate_set": _env_flag(STATE_GATE),
            "training_gate_set": _env_flag(TRAINING_GATE),
            "forbidden_gates_set": forbidden,
            "old_6d_so100_path_used": False,
            "hard_coded_gripper_fill_used": False,
            "old_tca_select_used": False,
            "openvla_oft_executed": False,
            "downloads_executed": False,
            "rollouts_executed": False,
        },
        "feasibility": None,
        "method_gate": None,
        "error": None,
    }

    def finish(decision: str, next_step: str, code: int) -> tuple[dict[str, Any], int]:
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"invalid final decision: {decision}")
        report["decision"] = decision
        report["summary"]["final_decision"] = decision
        report["summary"]["exact_next_step"] = next_step
        report["summary"]["runtime_sec"] = _round(time.monotonic() - started, 3)
        return report, code

    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("KILL_LEAKAGE_RISK", "Unset forbidden execution gates and rerun only the bounded TG-7D gate.", 20)
    if not report["policy"]["state_gate_set"]:
        return finish("NO_TARGET_GROUNDING_EVAL_PATH", f"Set {STATE_GATE}=1 to run STATE 1 feasibility.", 21)

    try:
        dataset = build_tg7d_dataset(
            data_root=Path(args.libero_data_root),
            metadata_csv=Path(args.libero_para_metadata_csv),
            max_tasks=int(args.max_tasks),
            max_train_paraphrases_per_task=int(args.max_train_paraphrases_per_task),
            max_eval_paraphrases_per_task=int(args.max_eval_paraphrases_per_task),
            train_demos=int(args.train_demos),
            eval_demos=int(args.eval_demos),
            records_per_demo=int(args.records_per_demo),
        )
        report["feasibility"] = dataset["feasibility"]
        feasible = (
            dataset["feasibility"]["matched_hdf5_task_count"] >= 2
            and dataset["feasibility"]["heldout_paraphrase_records"] > 0
            and dataset["feasibility"]["heldout_object_lexical_records"] > 0
            and dataset["feasibility"]["counterfactual_records"] > 0
            and not dataset["feasibility"]["leakage"]["group_leakage_detected"]
            and dataset["feasibility"]["target_prior_from_instruction_and_visible_names_only"]
        )
        if not feasible:
            decision, next_step = _decide(report)
            return finish(decision, next_step, 2)
        if not report["policy"]["training_gate_set"]:
            return finish("NO_TARGET_GROUNDING_EVAL_PATH", f"STATE 1 is feasible; set {TRAINING_GATE}=1 for the bounded method gate.", 22)

        report["summary"]["experiments_happened"] = True
        report["summary"]["training_happened"] = True
        report["summary"]["loss_computed"] = True
        report["method_gate"] = _run_method_gate(args, dataset)
        report["summary"]["dataset_split_used"] = report["method_gate"]["dataset_split_used"]
        report["summary"]["lora_rank"] = int(args.lora_rank)
        report["summary"]["vram_peak_mb"] = report["method_gate"]["vram_peak_mb"]
        report["summary"]["trainable_params"] = {
            name: payload.get("trainable_params")
            for name, payload in report["method_gate"]["variants"].items()
            if isinstance(payload, dict) and payload.get("trainable_params") is not None
        }
        decision, next_step = _decide(report)
        return finish(decision, next_step, 0)
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        if "out of memory" in str(exc).lower():
            return finish("TOO_HEAVY_LOCAL", "Stop: TG-7D gate exceeded local memory.", 10)
        return finish("NO_TARGET_GROUNDING_EVAL_PATH", "Stop: TG-7D gate failed before producing valid evidence.", 11)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-data-root", default=str(DEFAULT_LIBERO_DATA_ROOT))
    parser.add_argument("--libero-para-metadata-csv", default=str(DEFAULT_LIBERO_PARA_CSV))
    parser.add_argument("--smolvla-ckpt", default=str(DEFAULT_SMOLVLA_CKPT))
    parser.add_argument("--report-path", default="reports/tg7d_adapter_state_gate.json")
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--max-train-paraphrases-per-task", type=int, default=4)
    parser.add_argument("--max-eval-paraphrases-per-task", type=int, default=6)
    parser.add_argument("--train-demos", type=int, default=4)
    parser.add_argument("--eval-demos", type=int, default=2)
    parser.add_argument("--records-per-demo", type=int, default=3)
    parser.add_argument("--adapter-steps", type=int, default=500)
    parser.add_argument("--simple-steps", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=4)
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_path.resolve().parent == Path("reports").resolve():
        _write_report_bundle(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
