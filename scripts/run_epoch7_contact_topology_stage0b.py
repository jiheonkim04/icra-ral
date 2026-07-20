"""Run the frozen pre-method contact-topology Stage 0B headroom gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "reports"
    / "epoch6_contact_transition_topology"
    / "problem_verification_protocol.json"
)
CONTRACT_PATH = (
    REPO_ROOT
    / "reports"
    / "epoch7_contact_transition_topology"
    / "stage0b_execution_contract.json"
)
EXPECTED_PROTOCOL_SHA256 = "7FA28AAEEAC9886F36DD5CCD059CA7AC4CD65B21FABFBBCA4AFFA53B0A256240"
EXPECTED_CONTRACT_SHA256 = "D50FE6287B68BEF93F292F6AD9C207740F1B56DD599D6B1004B5E984D2764A20"
EXPECTED_STAGE0A_RESULT_SHA256 = "3D18F9D7C6FA2E6311E4667853D1247FFC515B1BCD3325671A46BB0994160E06"
ROOT_SEED = 620260721
SEEDS = (620260721, 620260722, 620260723)
ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
TYPED_BINS = (
    "free-free:birth",
    "free-free:death",
    "free-articulated:birth",
    "free-articulated:death",
    "free-static:birth",
    "free-static:death",
    "articulated-articulated:birth",
    "articulated-articulated:death",
    "articulated-static:birth",
    "articulated-static:death",
)
ALLOWED_DATASETS = {
    "actions",
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
    "obs/ee_pos",
    "obs/ee_ori",
    "obs/gripper_states",
    "obs/joint_states",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite preserved artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resource_snapshot() -> dict[str, Any]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "captured_at": utc_now(),
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
        "disk_free_bytes": disk.free,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
    }


def require_safe_resources(snapshot: Mapping[str, Any]) -> None:
    if int(snapshot["swap_used_bytes"]) != 0:
        raise RuntimeError("Stage 0B requires zero WSL swap use")
    if int(snapshot["disk_free_bytes"]) < 60_000_000_000:
        raise RuntimeError("Stage 0B requires at least 60 GB free disk")


def task_token(task: Mapping[str, Any]) -> str:
    return f"{task['suite']}_task{int(task['task_id'])}"


def frozen_tasks(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for partition in ("development_train", "development_tune", "validation"):
        for task in protocol["task_split"][partition]:
            row = dict(task)
            row["partition"] = partition
            result.append(row)
    if len(result) != 10 or len({task_token(row) for row in result}) != 10:
        raise RuntimeError("frozen Stage 0B task panel is invalid")
    return result


def validate_inputs(stage0a_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen scientific protocol hash mismatch")
    if sha256_file(CONTRACT_PATH) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen Stage 0B execution-contract hash mismatch")
    result_path = stage0a_dir / "stage0a_result.json"
    if sha256_file(result_path) != EXPECTED_STAGE0A_RESULT_SHA256:
        raise RuntimeError("Stage 0A authorization result hash mismatch")
    stage0a = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        stage0a["final_decision"] != "CONTACT_TOPOLOGY_LABEL_GATE_GO"
        or not stage0a["stage0b_authorized"]
        or stage0a["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("Stage 0A did not authorize Stage 0B")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    tasks = frozen_tasks(protocol)
    for task in tasks:
        token = task_token(task)
        metadata_path = stage0a_dir / f"task_{token}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        npz_path = stage0a_dir / f"task_{token}.npz"
        if (
            metadata["status"] != "TASK_CONTACT_EXTRACTION_PASS"
            or metadata["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
            or metadata["npz_sha256"] != sha256_file(npz_path)
        ):
            raise RuntimeError(f"Stage 0A task artifact failed integrity: {token}")
    return protocol, tasks


def read_allowed(group: h5py.Group, name: str, accesses: list[str]) -> np.ndarray:
    if name not in ALLOWED_DATASETS:
        raise RuntimeError(f"forbidden Stage 0B dataset access requested: {name}")
    accesses.append(name)
    node: Any = group
    for part in name.split("/"):
        node = node[part]
    return np.asarray(node)


def resize_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        cv2.resize(first, (64, 64), interpolation=cv2.INTER_AREA),
        cv2.resize(second, (64, 64), interpolation=cv2.INTER_AREA),
    )


def boundary_mask(typed: np.ndarray, radius: int = 2) -> np.ndarray:
    exact = np.any(typed != 0, axis=1)
    result = np.zeros(len(typed), dtype=bool)
    for index in np.flatnonzero(exact):
        result[max(0, index - radius) : min(len(typed), index + radius + 1)] = True
    return result


def load_rows(
    stage0a_dir: Path, tasks: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    images: list[np.ndarray] = []
    base: list[np.ndarray] = []
    gripper_change: list[float] = []
    stage: list[np.ndarray] = []
    typed_rows: list[np.ndarray] = []
    arm_targets: list[np.ndarray] = []
    boundary_rows: list[bool] = []
    partitions: list[str] = []
    task_tokens: list[str] = []
    groups: list[str] = []
    time_indices: list[int] = []
    accesses: list[str] = []
    source_artifacts: list[dict[str, Any]] = []
    for task in tasks:
        token = task_token(task)
        metadata_path = stage0a_dir / f"task_{token}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        npz_path = stage0a_dir / f"task_{token}.npz"
        with np.load(npz_path, allow_pickle=False) as payload:
            typed_all = np.asarray(payload["typed_transitions"], dtype=np.uint8)
            offsets = np.asarray(payload["demo_offsets"], dtype=np.int64)
            demo_ids = np.asarray(payload["demo_ids"], dtype=np.int32)
            state_indices = np.asarray(payload["state_indices"], dtype=np.int32)
        source_artifacts.append(
            {
                "token": token,
                "metadata_sha256": sha256_file(metadata_path),
                "npz_sha256": sha256_file(npz_path),
                "dataset_file": metadata["dataset_file"],
            }
        )
        with h5py.File(metadata["dataset_file"], "r") as handle:
            for demo_position, demo_id in enumerate(demo_ids):
                left, right = int(offsets[demo_position]), int(offsets[demo_position + 1])
                typed = typed_all[left:right]
                expected_indices = np.arange(right - left, dtype=np.int32)
                if not np.array_equal(state_indices[left:right], expected_indices):
                    raise RuntimeError(f"state alignment mismatch: {token}/demo_{demo_id}")
                demo = handle["data"][f"demo_{int(demo_id)}"]
                actions = read_allowed(demo, "actions", accesses).astype(np.float32)
                agent = read_allowed(demo, "obs/agentview_rgb", accesses)
                wrist = read_allowed(demo, "obs/eye_in_hand_rgb", accesses)
                ee_pos = read_allowed(demo, "obs/ee_pos", accesses).astype(np.float32)
                ee_ori = read_allowed(demo, "obs/ee_ori", accesses).astype(np.float32)
                gripper = read_allowed(demo, "obs/gripper_states", accesses).astype(np.float32)
                joints = read_allowed(demo, "obs/joint_states", accesses).astype(np.float32)
                length = len(typed)
                arrays = (actions, agent, wrist, ee_pos, ee_ori, gripper, joints)
                if any(len(array) != length for array in arrays) or length < 4:
                    raise RuntimeError(f"observation/action alignment mismatch: {token}/demo_{demo_id}")
                local_boundary = boundary_mask(typed)
                for t in range(3, length):
                    agent_pair = resize_pair(agent[t - 1], agent[t])
                    wrist_pair = resize_pair(wrist[t - 1], wrist[t])
                    image = np.concatenate((*agent_pair, *wrist_pair), axis=2).transpose(2, 0, 1)
                    robot = np.concatenate((ee_pos[t - 1], ee_ori[t - 1], gripper[t - 1], joints[t - 1]))
                    history = np.concatenate((actions[t - 2], actions[t - 3]))
                    base.append(
                        np.concatenate(
                            (robot, history, np.asarray([t / (length - 1)], dtype=np.float32))
                        ).astype(np.float32)
                    )
                    images.append(np.ascontiguousarray(image, dtype=np.uint8))
                    gripper_change.append(
                        float((actions[t - 1, 6] >= 0) != (actions[t - 2, 6] >= 0))
                    )
                    stage_one_hot = np.zeros(4, dtype=np.float32)
                    stage_one_hot[min(3, int(4 * t / length))] = 1.0
                    stage.append(stage_one_hot)
                    typed_rows.append(typed[t].astype(np.float32))
                    arm_targets.append(actions[t - 1, :6].astype(np.float32))
                    boundary_rows.append(bool(local_boundary[t]))
                    partitions.append(str(task["partition"]))
                    task_tokens.append(token)
                    groups.append(f"{token}/demo_{int(demo_id)}")
                    time_indices.append(t)
    result = {
        "images": np.stack(images),
        "base": np.stack(base),
        "gripper_change": np.asarray(gripper_change, dtype=np.float32)[:, None],
        "stage": np.stack(stage),
        "typed": np.stack(typed_rows),
        "arm_targets": np.stack(arm_targets),
        "boundary": np.asarray(boundary_rows, dtype=bool),
        "partition": np.asarray(partitions),
        "task": np.asarray(task_tokens),
        "group": np.asarray(groups),
        "t": np.asarray(time_indices, dtype=np.int32),
    }
    if result["images"].shape[1:] != (12, 64, 64):
        raise RuntimeError("visual tensor shape mismatch")
    for name in ("base", "typed", "arm_targets"):
        if not np.isfinite(result[name]).all():
            raise RuntimeError(f"nonfinite Stage 0B row tensor: {name}")
    manifest = {
        "schema_version": "epoch7.contact_topology.stage0b_rows.v1",
        "created_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "rows": len(images),
        "rows_by_partition": dict(Counter(partitions)),
        "rows_by_task": dict(Counter(task_tokens)),
        "oracle_boundary_rows": int(np.sum(result["boundary"])),
        "oracle_boundary_rows_by_partition": {
            partition: int(np.sum(result["boundary"] & (result["partition"] == partition)))
            for partition in ("development_train", "development_tune", "validation")
        },
        "typed_positive_counts": {
            name: int(result["typed"][:, index].sum()) for index, name in enumerate(TYPED_BINS)
        },
        "dataset_accesses": dict(Counter(accesses)),
        "forbidden_dataset_access_count": 0,
        "reward_success_done_read": False,
        "simulator_actions_executed": 0,
        "success_check_calls": 0,
        "policy_inference_calls": 0,
        "source_artifacts": source_artifacts,
    }
    return result, manifest


def safe_ap(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    positives = int(target.sum())
    if positives == 0:
        return 0.0
    if positives == len(target):
        return 1.0
    return float(average_precision_score(target, score))


class ProbeDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        nonvisual: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        include_images: bool = True,
    ) -> None:
        self.images = images
        self.nonvisual = nonvisual
        self.targets = targets
        self.indices = indices
        self.include_images = include_images

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        image = (
            torch.from_numpy(self.images[row]).to(torch.float32).div_(127.5).sub_(1.0)
            if self.include_images
            else torch.empty(0, dtype=torch.float32)
        )
        nonvisual = torch.from_numpy(self.nonvisual[row])
        target = torch.from_numpy(self.targets[row])
        return image, nonvisual, target


class VisualProbe(nn.Module):
    def __init__(self, nonvisual_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + nonvisual_dim, 64), nn.ReLU(), nn.Linear(64, 11)
        )

    def forward(self, image: torch.Tensor, nonvisual: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(image).flatten(1)
        return self.fusion(torch.cat((encoded, nonvisual), dim=1))


class NonvisualProbe(nn.Module):
    def __init__(self, nonvisual_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(nonvisual_dim, 64), nn.ReLU(), nn.Linear(64, 11)
        )

    def forward(self, _image: torch.Tensor, nonvisual: torch.Tensor) -> torch.Tensor:
        return self.network(nonvisual)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def predict_probe(
    model: nn.Module, dataset: ProbeDataset, device: torch.device
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for image, nonvisual, _target in loader:
            logits = model(image.to(device), nonvisual.to(device))
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks)


def train_probe(
    kind: str,
    seed: int,
    rows: Mapping[str, np.ndarray],
    nonvisual: np.ndarray,
    targets: np.ndarray,
    indices: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    set_seed(seed)
    model: nn.Module
    if kind == "visual":
        model = VisualProbe(nonvisual.shape[1])
    elif kind == "nonvisual":
        model = NonvisualProbe(nonvisual.shape[1])
    else:
        raise ValueError(kind)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    include_images = kind == "visual"
    train_dataset = ProbeDataset(
        rows["images"], nonvisual, targets, indices["train"], include_images
    )
    tune_dataset = ProbeDataset(
        rows["images"], nonvisual, targets, indices["tune"], include_images
    )
    validation_dataset = ProbeDataset(
        rows["images"], nonvisual, targets, indices["validation"], include_images
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    curve: list[dict[str, Any]] = []
    for epoch in range(1, 21):
        model.train()
        losses: list[float] = []
        for image, feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(image.to(device), feature.to(device))
            loss = loss_function(logits, target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        tune_score = predict_probe(model, tune_dataset, device)
        tune_ap = safe_ap(targets[indices["tune"], 0], tune_score[:, 0])
        curve.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "development_tune_any_transition_auprc": tune_ap,
            }
        )
        if tune_ap > best_score + 1e-12:
            best_score = tune_ap
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("probe checkpoint selection failed")
    model.load_state_dict(best_state)
    model.to(device)
    validation_score = predict_probe(model, validation_dataset, device)
    report = {
        "kind": kind,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selected_development_tune_any_transition_auprc": best_score,
        "training_curve": curve,
        "finite_validation_prediction_fraction": float(np.isfinite(validation_score).mean()),
    }
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report, validation_score


def visual_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    validation_indices: np.ndarray,
    validation_tasks: np.ndarray,
    supported: np.ndarray,
) -> dict[str, Any]:
    target = targets[validation_indices]
    typed_ap = [safe_ap(target[:, index + 1], scores[:, index + 1]) for index in supported]
    task_metrics: dict[str, float] = {}
    for token in sorted(set(validation_tasks.tolist())):
        mask = validation_tasks == token
        task_metrics[token] = safe_ap(target[mask, 0], scores[mask, 0])
    return {
        "any_transition_auprc": safe_ap(target[:, 0], scores[:, 0]),
        "supported_typed_bin_macro_ap": float(np.mean(typed_ap)),
        "supported_typed_bin_ap": {
            TYPED_BINS[index]: typed_ap[position] for position, index in enumerate(supported)
        },
        "per_task_any_transition_auprc": task_metrics,
    }


def run_visual_gate(
    output_dir: Path,
    rows: Mapping[str, np.ndarray],
    manifest_sha256: str,
) -> dict[str, Any]:
    partition = rows["partition"]
    indices = {
        "train": np.flatnonzero(partition == "development_train"),
        "tune": np.flatnonzero(partition == "development_tune"),
        "validation": np.flatnonzero(partition == "validation"),
    }
    train_mean = rows["base"][indices["train"]].mean(axis=0)
    train_std = rows["base"][indices["train"]].std(axis=0)
    train_std[train_std < 1e-8] = 1.0
    nonvisual = ((rows["base"] - train_mean) / train_std).astype(np.float32)
    any_target = np.any(rows["typed"] != 0, axis=1).astype(np.float32)[:, None]
    targets = np.concatenate((any_target, rows["typed"].astype(np.float32)), axis=1)
    supported = np.flatnonzero(rows["typed"].sum(axis=0) >= 10)
    if len(supported) < 3:
        raise RuntimeError("Stage 0A typed support was not preserved")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_reports: list[dict[str, Any]] = []
    validation_tasks = rows["task"][indices["validation"]]
    for seed in SEEDS:
        seed_path = output_dir / f"visual_seed_{seed}.json"
        if seed_path.exists():
            saved = json.loads(seed_path.read_text(encoding="utf-8"))
            if saved.get("contract_sha256") != EXPECTED_CONTRACT_SHA256 or saved.get(
                "row_manifest_sha256"
            ) != manifest_sha256:
                raise RuntimeError(f"preserved visual seed artifact has wrong provenance: {seed}")
            seed_reports.append(saved)
            continue
        report: dict[str, Any] = {
            "schema_version": "epoch7.contact_topology.visual_seed.v1",
            "completed_at": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "row_manifest_sha256": manifest_sha256,
            "seed": seed,
            "device": str(device),
        }
        for kind in ("visual", "nonvisual"):
            training, scores = train_probe(
                kind, seed, rows, nonvisual, targets, indices, device
            )
            training["validation"] = visual_metrics(
                scores, targets, indices["validation"], validation_tasks, supported
            )
            report[kind] = training
        report["resources_after"] = resource_snapshot()
        require_safe_resources(report["resources_after"])
        write_json(seed_path, report)
        seed_reports.append(report)
        print(f"visual seed {seed} complete", flush=True)
    visual_any = np.asarray(
        [row["visual"]["validation"]["any_transition_auprc"] for row in seed_reports]
    )
    nonvisual_any = np.asarray(
        [row["nonvisual"]["validation"]["any_transition_auprc"] for row in seed_reports]
    )
    visual_typed = np.asarray(
        [row["visual"]["validation"]["supported_typed_bin_macro_ap"] for row in seed_reports]
    )
    nonvisual_typed = np.asarray(
        [row["nonvisual"]["validation"]["supported_typed_bin_macro_ap"] for row in seed_reports]
    )
    validation_target = targets[indices["validation"], 0]
    train_prevalence = float(targets[indices["train"], 0].mean())
    prevalence_ap = safe_ap(validation_target, np.full(len(validation_target), train_prevalence))
    task_medians: dict[str, dict[str, float]] = {}
    tasks_beating = 0
    for token in sorted(set(validation_tasks.tolist())):
        visual_values = [
            row["visual"]["validation"]["per_task_any_transition_auprc"][token]
            for row in seed_reports
        ]
        nonvisual_values = [
            row["nonvisual"]["validation"]["per_task_any_transition_auprc"][token]
            for row in seed_reports
        ]
        visual_median = float(np.median(visual_values))
        nonvisual_median = float(np.median(nonvisual_values))
        if visual_median > nonvisual_median:
            tasks_beating += 1
        task_medians[token] = {
            "visual_any_transition_auprc_median": visual_median,
            "nonvisual_any_transition_auprc_median": nonvisual_median,
            "visual_minus_nonvisual": visual_median - nonvisual_median,
        }
    visual_any_median = float(np.median(visual_any))
    nonvisual_any_median = float(np.median(nonvisual_any))
    visual_typed_median = float(np.median(visual_typed))
    nonvisual_typed_median = float(np.median(nonvisual_typed))
    metrics = {
        "development_train_any_transition_prevalence": train_prevalence,
        "validation_any_transition_prevalence": float(validation_target.mean()),
        "prevalence_predictor_validation_auprc": prevalence_ap,
        "visual_any_transition_auprc_median": visual_any_median,
        "nonvisual_any_transition_auprc_median": nonvisual_any_median,
        "any_transition_auprc_over_prevalence_ratio": visual_any_median / prevalence_ap,
        "any_transition_auprc_over_nonvisual": visual_any_median - nonvisual_any_median,
        "visual_supported_typed_bin_macro_ap_median": visual_typed_median,
        "nonvisual_supported_typed_bin_macro_ap_median": nonvisual_typed_median,
        "supported_typed_bin_macro_ap_over_nonvisual": visual_typed_median
        - nonvisual_typed_median,
        "validation_tasks_beating_nonvisual_any_transition": tasks_beating,
        "validation_task_medians": task_medians,
        "supported_typed_bins": [TYPED_BINS[index] for index in supported],
        "seed_results": seed_reports,
    }
    gates = {
        "any_transition_auprc_over_prevalence_ratio": metrics[
            "any_transition_auprc_over_prevalence_ratio"
        ]
        >= 2.0,
        "any_transition_auprc_over_nonvisual": metrics[
            "any_transition_auprc_over_nonvisual"
        ]
        >= 0.1,
        "supported_typed_bin_macro_ap_over_nonvisual": metrics[
            "supported_typed_bin_macro_ap_over_nonvisual"
        ]
        >= 0.05,
        "validation_tasks_beating_nonvisual_any_transition": tasks_beating >= 3,
    }
    return {"metrics": metrics, "gates_passed": gates, "all_gates_passed": all(gates.values())}


def standardize_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def nrmse(target_standard: np.ndarray, prediction_standard: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target_standard - prediction_standard))))


def fit_ridge_family(
    x_train: np.ndarray,
    x_tune: np.ndarray,
    x_validation: np.ndarray,
    y_train_standard: np.ndarray,
    y_tune_standard: np.ndarray,
    y_validation_standard: np.ndarray,
) -> dict[str, Any]:
    mean, scale = standardize_fit(x_train)
    train = (x_train - mean) / scale
    tune = (x_tune - mean) / scale
    validation = (x_validation - mean) / scale
    scores: list[dict[str, float]] = []
    best_alpha = ALPHAS[0]
    best_score = float("inf")
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(train, y_train_standard)
        score = nrmse(y_tune_standard, model.predict(tune))
        scores.append({"alpha": alpha, "development_tune_nrmse": score})
        if score < best_score - 1e-12:
            best_alpha, best_score = alpha, score
    model = Ridge(alpha=best_alpha)
    model.fit(train, y_train_standard)
    prediction = model.predict(validation)
    return {
        "selected_alpha": best_alpha,
        "development_tune_nrmse": best_score,
        "alpha_scores": scores,
        "validation_prediction": prediction,
        "validation_nrmse": nrmse(y_validation_standard, prediction),
    }


def shuffled_within_groups(
    values: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = values.copy()
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        result[indices] = values[rng.permutation(indices)]
    return result


def run_oracle_gate(rows: Mapping[str, np.ndarray]) -> dict[str, Any]:
    selected = rows["boundary"]
    partition = rows["partition"][selected]
    task = rows["task"][selected]
    group = rows["group"][selected]
    base = rows["base"][selected].astype(np.float64)
    gripper_stage = np.concatenate(
        (base, rows["gripper_change"][selected], rows["stage"][selected]), axis=1
    )
    typed = rows["typed"][selected].astype(np.float64)
    binary = np.any(typed != 0, axis=1).astype(np.float64)[:, None]
    binary_control = np.concatenate((base, binary), axis=1)
    target = rows["arm_targets"][selected].astype(np.float64)
    split = {
        "train": np.flatnonzero(partition == "development_train"),
        "tune": np.flatnonzero(partition == "development_tune"),
        "validation": np.flatnonzero(partition == "validation"),
    }
    if any(len(indices) == 0 for indices in split.values()):
        raise RuntimeError("oracle boundary subset has an empty partition")
    y_mean, y_scale = standardize_fit(target[split["train"]])
    target_standard = (target - y_mean) / y_scale

    def fit(features: np.ndarray) -> dict[str, Any]:
        return fit_ridge_family(
            features[split["train"]],
            features[split["tune"]],
            features[split["validation"]],
            target_standard[split["train"]],
            target_standard[split["tune"]],
            target_standard[split["validation"]],
        )

    real = {
        "base": fit(base),
        "gripper_stage_control": fit(gripper_stage),
        "binary_control": fit(binary_control),
    }
    if (
        real["gripper_stage_control"]["development_tune_nrmse"]
        <= real["binary_control"]["development_tune_nrmse"]
    ):
        strongest_name = "gripper_stage_control"
        strongest_features = gripper_stage
    else:
        strongest_name = "binary_control"
        strongest_features = binary_control
    full_features = np.concatenate((strongest_features, typed), axis=1)
    real["full"] = fit(full_features)
    validation_tasks = task[split["validation"]]
    strongest_prediction = real[strongest_name]["validation_prediction"]
    full_prediction = real["full"]["validation_prediction"]
    validation_target = target_standard[split["validation"]]
    per_task: dict[str, dict[str, float]] = {}
    tasks_lower = 0
    for token in sorted(set(validation_tasks.tolist())):
        mask = validation_tasks == token
        strongest_nrmse = nrmse(validation_target[mask], strongest_prediction[mask])
        full_nrmse = nrmse(validation_target[mask], full_prediction[mask])
        if full_nrmse < strongest_nrmse:
            tasks_lower += 1
        per_task[token] = {
            "strongest_control_nrmse": strongest_nrmse,
            "full_nrmse": full_nrmse,
            "relative_reduction": (strongest_nrmse - full_nrmse) / strongest_nrmse,
        }
    shuffle_rows: list[dict[str, Any]] = []
    all_predictions = [
        real["base"]["validation_prediction"],
        real["gripper_stage_control"]["validation_prediction"],
        real["binary_control"]["validation_prediction"],
        full_prediction,
    ]
    for shuffle_index in range(50):
        rng = np.random.default_rng(ROOT_SEED + 10_000 + shuffle_index)
        shuffled = typed.copy()
        for partition_name, indices in split.items():
            shuffled[indices] = shuffled_within_groups(
                typed[indices], group[indices], rng
            )
        shuffled_features = np.concatenate((strongest_features, shuffled), axis=1)
        fitted = fit(shuffled_features)
        all_predictions.append(fitted["validation_prediction"])
        shuffle_rows.append(
            {
                "shuffle_index": shuffle_index,
                "seed": ROOT_SEED + 10_000 + shuffle_index,
                "selected_alpha": fitted["selected_alpha"],
                "development_tune_nrmse": fitted["development_tune_nrmse"],
                "validation_nrmse": fitted["validation_nrmse"],
            }
        )
    strongest_nrmse = float(real[strongest_name]["validation_nrmse"])
    full_nrmse = float(real["full"]["validation_nrmse"])
    full_gain = strongest_nrmse - full_nrmse
    relative_reduction = full_gain / strongest_nrmse
    shuffle_gains = np.asarray(
        [strongest_nrmse - row["validation_nrmse"] for row in shuffle_rows]
    )
    shuffle_fraction = float(np.median(shuffle_gains) / full_gain) if full_gain > 0 else None
    finite_values = np.concatenate([prediction.ravel() for prediction in all_predictions])
    finite_fraction = float(np.isfinite(finite_values).mean())
    for family in real.values():
        family.pop("validation_prediction")
    metrics = {
        "boundary_rows": int(np.sum(selected)),
        "boundary_rows_by_partition": {
            name: int(len(indices)) for name, indices in split.items()
        },
        "strongest_control": strongest_name,
        "models": real,
        "strongest_control_validation_nrmse": strongest_nrmse,
        "full_validation_nrmse": full_nrmse,
        "aggregate_arm_nrmse_relative_reduction": relative_reduction,
        "validation_tasks_with_lower_arm_nrmse": tasks_lower,
        "validation_per_task": per_task,
        "shuffle_rows": shuffle_rows,
        "shuffled_topology_fraction_of_full_gain": shuffle_fraction,
        "finite_prediction_fraction": finite_fraction,
    }
    gates = {
        "aggregate_arm_nrmse_relative_reduction": relative_reduction >= 0.05,
        "validation_tasks_with_lower_arm_nrmse": tasks_lower >= 3,
        "shuffled_topology_fraction_of_full_gain": shuffle_fraction is not None
        and shuffle_fraction <= 0.2,
        "finite_prediction_fraction": finite_fraction >= 1.0,
    }
    return {"metrics": metrics, "gates_passed": gates, "all_gates_passed": all(gates.values())}


def adjudicate(visual: Mapping[str, Any], oracle: Mapping[str, Any]) -> str:
    if visual["all_gates_passed"] and oracle["all_gates_passed"]:
        return "CONTACT_TOPOLOGY_PREMETHOD_STAGE0_GO"
    visual_metrics_row = visual["metrics"]
    oracle_metrics_row = oracle["metrics"]
    visual_equivalent = (
        visual_metrics_row["any_transition_auprc_over_nonvisual"] < 0.1
        and visual_metrics_row["supported_typed_bin_macro_ap_over_nonvisual"] < 0.05
    )
    oracle_equivalent = (
        oracle_metrics_row["aggregate_arm_nrmse_relative_reduction"] <= 0
    )
    if visual_equivalent or oracle_equivalent:
        return "STAGE0_TRIVIAL_EQUIVALENCE"
    return "STAGE0_NO_HEADROOM"


def run(stage0a_dir: Path, output_dir: Path) -> dict[str, Any]:
    resources_before = resource_snapshot()
    require_safe_resources(resources_before)
    _protocol, tasks = validate_inputs(stage0a_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "row_manifest.json"
    rows, manifest = load_rows(stage0a_dir, tasks)
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = dict(observed)
        comparable.pop("created_at", None)
        expected = dict(manifest)
        expected.pop("created_at", None)
        if comparable != expected:
            raise RuntimeError("preserved Stage 0B row manifest mismatch")
    else:
        write_json(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    print(f"loaded {manifest['rows']} aligned rows", flush=True)
    visual = run_visual_gate(output_dir, rows, manifest_sha256)
    oracle = run_oracle_gate(rows)
    resources_after = resource_snapshot()
    require_safe_resources(resources_after)
    result = {
        "schema_version": "epoch7.contact_topology.stage0b_result.v1",
        "completed_at": utc_now(),
        "execution_type": "DISCOVERY_PREMETHOD_PREDICTABILITY_AND_ACTION_HEADROOM",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "stage0a_result_sha256": EXPECTED_STAGE0A_RESULT_SHA256,
        "row_manifest_sha256": manifest_sha256,
        "resources_before": resources_before,
        "resources_after": resources_after,
        "visual_probe": visual,
        "oracle_action_headroom": oracle,
        "scientific_firewall": {
            "forbidden_dataset_access_count": 0,
            "reward_success_done_read": False,
            "simulator_actions_executed": 0,
            "success_check_calls": 0,
            "policy_inference_calls": 0,
            "vla_training_happened": False,
            "policy_rollout_happened": False,
        },
        "final_decision": adjudicate(visual, oracle),
        "method_contract_authorized": bool(
            visual["all_gates_passed"] and oracle["all_gates_passed"]
        ),
        "vla_training_authorized": False,
        "policy_rollout_authorized": False,
        "paper_generation_authorized": False,
    }
    write_json(output_dir / "stage0b_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0a-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.stage0a_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
