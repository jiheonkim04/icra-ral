"""Mechanism-faithful local RL4IL prior port.

This module is intentionally scoped to the external-prior comparator. It is
not VLA training, not Ours, and not an official RL4IL reproduction. The local
port preserves the RL4IL retrieval/imputation structure while replacing the
released scripts' constant scalar labels with the preregistered action-sequence
oracle supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import time
import traceback
from dataclasses import dataclass
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from transformers import CLIPModel, CLIPProcessor

try:  # Windows-side unit validation lacks the POSIX resource module.
    import resource
except ImportError:  # pragma: no cover - WSL empirical runtime has resource
    resource = None  # type: ignore[assignment]

from tca_map.rl4il_prior.action_oracle import (
    ActionOracleConfig,
    pairwise_action_distance_matrix,
    resample_action_sequence,
)


IMPLEMENTATION_LABEL = "MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT"
IDENTITY_BASE = 20260711
SEED = 42
NUM_FRAMES = 8
CAM_DIM = 512
LANG_DIM = 512
FULL_DIM = CAM_DIM + CAM_DIM + LANG_DIM
ACTION_DIM = 7
ACTION_DESC_STEPS = 64
ACTION_DESC_DIM = ACTION_DESC_STEPS * ACTION_DIM
K_APPROX = 20
K_GRAPH = 5
MAX_BFS_DEPTH = 6
MAX_NODE2_RL = 200
FUSION_TOPK = 32
IMP_SOFT_TOPK = 32


PANEL = [
    {
        "suite": "libero_goal",
        "task_id": 0,
        "instruction": "open the middle drawer of the cabinet",
        "hdf5": "open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        "identities": [20260733, 20260734, 20260735],
    },
    {
        "suite": "libero_object",
        "task_id": 0,
        "instruction": "pick up the alphabet soup and place it in the basket",
        "hdf5": "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
        "identities": [20260733, 20260734, 20260735],
    },
    {
        "suite": "libero_spatial",
        "task_id": 5,
        "instruction": "pick up the black bowl on the ramekin and place it on the plate",
        "hdf5": "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "identities": [20260731, 20260732, 20260735],
    },
]


@dataclass(frozen=True)
class TrainBudget:
    prediction_ppo_epochs: int = 1
    prediction_fusion_epochs: int = 1
    imputation_ppo_epochs: int = 1
    soft_imputation_epochs: int = 1
    minibatch_size: int = 16
    lr: float = 3e-4
    imp_lr: float = 3e-4
    fusion_lr: float = 3e-4
    soft_imp_lr: float = 3e-4


@dataclass
class DemoRecord:
    key: str
    index: int
    instruction: str
    cam0_frames: list[np.ndarray]
    cam1_frames: list[np.ndarray]
    actions: np.ndarray
    action_descriptor: np.ndarray


def safe_task_name(suite: str, task_id: int) -> str:
    return f"{suite}_task{int(task_id)}"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def memory_report() -> dict[str, Any]:
    report: dict[str, Any] = {}
    if resource is not None:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        report["ru_maxrss_kib"] = int(ru.ru_maxrss)
    status_path = pathlib.Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                key, value = line.split(":", 1)
                report[key.lower()] = value.strip()
    meminfo_path = pathlib.Path("/proc/meminfo")
    if meminfo_path.exists():
        for line in meminfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                report[key.lower()] = value.strip()
    return report


def cuda_report() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "pid": int(os.getpid()),
        "device": torch.cuda.get_device_name(0),
        "max_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "max_reserved_mib": float(torch.cuda.max_memory_reserved() / (1024 * 1024)),
    }


def params_vector(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in module.parameters()])


def count_trainable(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def grad_norm(module: nn.Module) -> float:
    vals = []
    for param in module.parameters():
        if param.grad is not None:
            vals.append(float(param.grad.detach().norm().cpu()))
    if not vals:
        return 0.0
    return float(math.sqrt(sum(v * v for v in vals)))


class FrozenCLIPEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 encoder matching the selected RL4IL prior."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.to(device)

    @torch.no_grad()
    def encode_image(self, frames_list: list[list[np.ndarray]]) -> np.ndarray:
        device = next(self.model.parameters()).device
        all_feats = []
        for frames in frames_list:
            if len(frames) == 0:
                all_feats.append(torch.zeros(CAM_DIM, device=device))
                continue
            if len(frames) >= NUM_FRAMES:
                idxs = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
            else:
                repeat = (NUM_FRAMES + len(frames) - 1) // len(frames)
                idxs = np.tile(np.arange(len(frames)), repeat)[:NUM_FRAMES]
            pil_frames = [Image.fromarray(np.asarray(frames[i], dtype=np.uint8)) for i in idxs]
            inputs = self.processor(images=pil_frames, return_tensors="pt", padding=True).to(device)
            feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            all_feats.append(feats.mean(dim=0))
        return torch.stack(all_feats).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        feats = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return feats.detach().cpu().numpy().astype(np.float32)


class ScoringMLP(nn.Module):
    def __init__(self, state_dim: int, cand_dim: int, hidden: int = 256):
        super().__init__()
        self.qe = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.ce = nn.Sequential(nn.Linear(cand_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.sh = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, cand_feats: torch.Tensor) -> torch.Tensor:
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
            cand_feats = cand_feats.unsqueeze(0)
        _, k, _ = cand_feats.shape
        hq = self.qe(state)
        hq_exp = hq.unsqueeze(1).expand(-1, k, -1)
        hc = self.ce(cand_feats)
        scores = self.sh(torch.cat([hq_exp, hc, hq_exp * hc], dim=-1)).squeeze(-1)
        return scores.squeeze(0) if squeeze else scores


class ActionFusionHead(nn.Module):
    """Soft cross-attention head over RL-ranked candidate action descriptors."""

    def __init__(self, emb_dim: int = FULL_DIM, action_desc_dim: int = ACTION_DESC_DIM, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.heads = int(heads)
        self.d_head = int(hidden) // int(heads)
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.q_proj = nn.Linear(emb_dim, hidden, bias=False)
        self.k_proj = nn.Linear(emb_dim, hidden, bias=False)
        self.ctx_proj = nn.Linear(hidden, hidden)
        self.refine = nn.Sequential(
            nn.Linear(hidden + action_desc_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_desc_dim),
        )
        self.scale = self.d_head**-0.5

    def forward(self, q_emb: torch.Tensor, cand_embs: torch.Tensor, cand_action_desc: torch.Tensor) -> torch.Tensor:
        k_count = cand_embs.shape[0]
        q = self.q_proj(q_emb).view(self.heads, self.d_head)
        k = self.k_proj(cand_embs).view(k_count, self.heads, self.d_head)
        weights = F.softmax(torch.einsum("hd,khd->hk", q, k) * self.scale, dim=-1).mean(0)
        attended_action = (weights.unsqueeze(-1) * cand_action_desc).sum(0)
        k_flat = k.reshape(k_count, self.heads * self.d_head)
        context = F.relu(self.ctx_proj((weights.unsqueeze(-1) * k_flat).sum(0)))
        return self.refine(torch.cat([context, attended_action], dim=-1))


class SoftImputationHead(nn.Module):
    def __init__(self, d_full: int = FULL_DIM, d_m: int = CAM_DIM, hidden: int = 64, heads: int = 2):
        super().__init__()
        self.heads = int(heads)
        self.d_head = int(hidden) // int(heads)
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.q_proj = nn.Linear(d_full, hidden, bias=False)
        self.k_proj = nn.Linear(d_m, hidden, bias=False)
        self.refine = nn.Sequential(nn.Linear(d_m + hidden, hidden), nn.ReLU(), nn.Linear(hidden, d_m))
        self.scale = self.d_head**-0.5

    def forward(self, q_partial: torch.Tensor, donor_embs: torch.Tensor) -> torch.Tensor:
        k_count = donor_embs.shape[0]
        q = self.q_proj(q_partial).view(self.heads, self.d_head)
        k = self.k_proj(donor_embs).view(k_count, self.heads, self.d_head)
        weights = F.softmax(torch.einsum("hd,khd->hk", q, k) * self.scale, dim=-1).mean(0)
        attended = (weights.unsqueeze(-1) * donor_embs).sum(0)
        k_flat = k.reshape(k_count, self.heads * self.d_head)
        context = F.relu((weights.unsqueeze(-1) * k_flat).sum(0))
        return self.refine(torch.cat([attended, context], dim=-1))


def demo_index(key: str) -> int:
    if not key.startswith("demo_"):
        raise ValueError(f"unexpected demo key {key!r}")
    return int(key.split("_", 1)[1])


def sample_frames(frames: np.ndarray) -> list[np.ndarray]:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.shape[0] >= NUM_FRAMES:
        idxs = np.linspace(0, frames.shape[0] - 1, NUM_FRAMES, dtype=int)
    else:
        repeat = (NUM_FRAMES + frames.shape[0] - 1) // frames.shape[0]
        idxs = np.tile(np.arange(frames.shape[0]), repeat)[:NUM_FRAMES]
    return [frames[i].copy() for i in idxs]


def load_task_demos(dataset_root: pathlib.Path, task: dict[str, Any]) -> list[DemoRecord]:
    path = dataset_root / str(task["suite"]) / str(task["hdf5"])
    config = ActionOracleConfig(resample_steps=ACTION_DESC_STEPS, length_penalty_weight=0.01)
    demos: list[DemoRecord] = []
    with h5py.File(path, "r") as h:
        for key in sorted(h["data"].keys(), key=demo_index):
            group = h["data"][key]
            obs = group["obs"]
            cam0 = obs["agentview_rgb"][:] if "agentview_rgb" in obs else obs["agentview_image"][:]
            if "eye_in_hand_rgb" in obs:
                cam1 = obs["eye_in_hand_rgb"][:]
            elif "robot0_eye_in_hand_rgb" in obs:
                cam1 = obs["robot0_eye_in_hand_rgb"][:]
            else:
                raise KeyError(f"{path} {key} has no in-hand RGB stream")
            actions = np.asarray(group["actions"], dtype=np.float32)
            demos.append(
                DemoRecord(
                    key=key,
                    index=demo_index(key),
                    instruction=str(task["instruction"]),
                    cam0_frames=sample_frames(cam0),
                    cam1_frames=sample_frames(cam1),
                    actions=actions,
                    action_descriptor=resample_action_sequence(actions, steps=config.resample_steps).reshape(-1),
                )
            )
    return demos


def extract_demo_embeddings(demos: list[DemoRecord], clip: FrozenCLIPEncoder, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c0_list: list[np.ndarray] = []
    c1_list: list[np.ndarray] = []
    text_list: list[np.ndarray] = []
    for start in range(0, len(demos), batch_size):
        batch = demos[start : start + batch_size]
        c0_list.append(clip.encode_image([demo.cam0_frames for demo in batch]))
        c1_list.append(clip.encode_image([demo.cam1_frames for demo in batch]))
        text_list.append(clip.encode_text([demo.instruction for demo in batch]))
    return np.concatenate(c0_list), np.concatenate(c1_list), np.concatenate(text_list)


def compute_stats(raw: list[np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, arr in enumerate(raw):
        mu = arr.mean(axis=0)
        sigma = arr.std(axis=0, ddof=1)
        sigma = np.where(sigma < 1e-8, np.ones_like(sigma), sigma)
        stats[idx] = (mu.astype(np.float32), sigma.astype(np.float32))
    return stats


def partial_embedding(raw_mods: list[np.ndarray], present: np.ndarray, stats: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    dims = [CAM_DIM, CAM_DIM, LANG_DIM]
    present_count = max(1, int(present.sum()))
    pieces = []
    for idx, (row, dim) in enumerate(zip(raw_mods, dims)):
        if not bool(present[idx]):
            pieces.append(np.zeros(dim, dtype=np.float32))
            continue
        mu, sigma = stats[idx]
        z = (row.astype(np.float32) - mu) / sigma
        pieces.append((z / float(dim * present_count) ** 0.5).astype(np.float32))
    return np.concatenate(pieces).astype(np.float32)


def full_embeddings(raw: list[np.ndarray], stats: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    present = np.ones(3, dtype=bool)
    return np.stack(
        [partial_embedding([raw[0][i], raw[1][i], raw[2][i]], present, stats) for i in range(raw[0].shape[0])]
    ).astype(np.float32)


def build_knn_graph(emb: np.ndarray, k: int = K_GRAPH) -> list[list[tuple[int, float]]]:
    kq = min(k + 1, emb.shape[0])
    idx = NearestNeighbors(n_neighbors=kq, metric="euclidean", n_jobs=-1).fit(emb)
    dists, inds = idx.kneighbors(emb)
    graph: list[list[tuple[int, float]]] = []
    for row_i in range(emb.shape[0]):
        edges = []
        for dist, ind in zip(dists[row_i], inds[row_i]):
            if int(ind) == row_i:
                continue
            edges.append((int(ind), float(dist)))
        graph.append(edges)
    return graph


def bfs(seeds: Iterable[tuple[int, float]], graph: list[list[tuple[int, float]]], target: int | None = None) -> list[tuple[int, float, int]]:
    import heapq

    visited: dict[int, tuple[float, int]] = {}
    heap: list[tuple[float, int, int]] = []
    for idx, dist in seeds:
        if idx not in visited:
            visited[idx] = (float(dist), 0)
            heapq.heappush(heap, (float(dist), int(idx), 0))
    sequence: list[tuple[int, float, int]] = []
    while heap:
        g_dist, node, depth = heapq.heappop(heap)
        if g_dist > visited.get(node, (float("inf"), 0))[0] + 1e-9:
            continue
        sequence.append((int(node), float(g_dist), int(depth)))
        if target is not None and len(sequence) >= target:
            break
        if depth >= MAX_BFS_DEPTH and len(sequence) >= MAX_NODE2_RL:
            break
        for nb, edge_weight in graph[node]:
            new_dist = g_dist + edge_weight
            if new_dist < visited.get(nb, (float("inf"), 0))[0]:
                visited[nb] = (new_dist, depth + 1)
                heapq.heappush(heap, (new_dist, int(nb), depth + 1))
    return sequence


def seed_neighbors(q: np.ndarray, emb: np.ndarray, exclude: int | None = None) -> list[tuple[int, float]]:
    kq = min(K_APPROX + 1, emb.shape[0])
    idx = NearestNeighbors(n_neighbors=kq, metric="euclidean", n_jobs=-1).fit(emb)
    dists, inds = idx.kneighbors(q.reshape(1, -1))
    seeds = []
    for ind, dist in zip(inds[0], dists[0]):
        if exclude is not None and int(ind) == int(exclude):
            continue
        if float(dist) <= 1e-12:
            continue
        seeds.append((int(ind), float(dist)))
    return seeds[:K_APPROX]


def build_candidate_sets(emb: np.ndarray, action_distance: np.ndarray, rng: np.random.RandomState) -> tuple[list[list[tuple[int, float, int]] | None], list[int | None], np.ndarray, list[list[tuple[int, float]]]]:
    graph = build_knn_graph(emb)
    sets: list[list[tuple[int, float, int]] | None] = []
    oracles: list[int | None] = []
    sizes = np.zeros(emb.shape[0], dtype=np.float32)
    for source in range(emb.shape[0]):
        seeds = seed_neighbors(emb[source], emb, exclude=source)
        sequence = [item for item in bfs(seeds, graph) if item[0] != source]
        if len(sequence) < 2:
            sets.append(None)
            oracles.append(None)
            continue
        oracle_node = min((node for node, _, _ in sequence), key=lambda node: (float(action_distance[source, node]), node))
        perm = rng.permutation(len(sequence))
        sequence = [sequence[int(pos)] for pos in perm]
        oracle_pos = int([node for node, _, _ in sequence].index(int(oracle_node)))
        sets.append(sequence)
        oracles.append(oracle_pos)
        sizes[source] = float(len(sequence))
    return sets, oracles, sizes, graph


def policy_features(q: np.ndarray, sequence: list[tuple[int, float, int]], emb: np.ndarray, action_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cand_emb = np.stack([emb[node] for node, _, _ in sequence])
    dists = np.linalg.norm(cand_emb - q.reshape(1, -1), axis=1)
    max_dist = float(dists.max()) + 1e-9
    max_depth = max(depth for _, _, depth in sequence) + 1e-9
    sorted_by_dist = np.argsort(dists)
    rank = {int(pos): rank_i for rank_i, pos in enumerate(sorted_by_dist)}
    state = np.concatenate([q, [float(dists.var())], [float(len(sequence))]]).astype(np.float32)
    cfs = []
    max_len = float(max(float(action_lengths.max()), 1.0))
    for pos, (node, _graph_dist, depth) in enumerate(sequence):
        cfs.append(
            np.concatenate(
                [
                    emb[node],
                    [
                        float(dists[pos] / max_dist),
                        float(depth / max_depth),
                        float(rank[pos] / max(len(sequence) - 1, 1)),
                        float(action_lengths[node] / max_len),
                    ],
                ]
            ).astype(np.float32)
        )
    return state, np.stack(cfs).astype(np.float32)


def ppo_update(
    policy: ScoringMLP,
    optimizer: torch.optim.Optimizer,
    batch: list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]],
    *,
    clip: float = 0.2,
    entropy_coef: float = 0.0001,
) -> tuple[float, float]:
    losses = []
    for state, cand_feats, action, old_lp, advantage in batch:
        scores = policy(state, cand_feats)
        log_pi = F.log_softmax(scores, dim=-1)
        pi = log_pi.exp()
        ratio = torch.exp(log_pi[int(action)] - old_lp.detach())
        entropy = -(pi * log_pi).sum()
        losses.append(
            -torch.min(ratio * advantage, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage)
            - entropy_coef * entropy
        )
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = grad_norm(policy)
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu()), float(norm)


def checkpoint_module(path: pathlib.Path, module: nn.Module) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)
    reloaded = type(module)(*getattr(module, "_constructor_args", ()))  # type: ignore[misc]
    state = torch.load(path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(state)
    return {"path": str(path), "sha256": sha256_file(path), "reload_ok": True}


def train_policy_module(
    name: str,
    emb: np.ndarray,
    action_desc: np.ndarray,
    action_lengths: np.ndarray,
    action_dist: np.ndarray,
    budget: TrainBudget,
    out_path: pathlib.Path,
    rng: np.random.RandomState,
) -> tuple[ScoringMLP, dict[str, Any], list[list[tuple[int, float, int]] | None], np.ndarray, list[list[tuple[int, float]]]]:
    sets, oracles, sizes, graph = build_candidate_sets(emb, action_dist, rng)
    policy = ScoringMLP(FULL_DIM + 2, FULL_DIM + 4).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    policy._constructor_args = (FULL_DIM + 2, FULL_DIM + 4)  # type: ignore[attr-defined]
    before = params_vector(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=budget.lr)
    losses: list[float] = []
    rewards: list[float] = []
    grad_norms: list[float] = []
    steps = 0
    device = next(policy.parameters()).device

    valid = [i for i, seq in enumerate(sets) if seq is not None and len(seq) >= 2 and oracles[i] is not None]
    for epoch in range(int(budget.prediction_ppo_epochs)):
        rng.shuffle(valid)
        rollout = []
        policy.eval()
        for i in valid:
            sequence = sets[i]
            oracle_pos = oracles[i]
            if sequence is None or oracle_pos is None:
                continue
            state_np, cand_np = policy_features(emb[i], sequence, emb, action_lengths)
            state = torch.tensor(state_np, dtype=torch.float32, device=device)
            cands = torch.tensor(cand_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = policy(state, cands)
                log_pi = F.log_softmax(scores, dim=-1)
                action = int(torch.multinomial(log_pi.exp(), 1).item())
                old_lp = log_pi[action].detach()
            distances = [float(action_dist[i, node]) for node, _, _ in sequence]
            rank_order = np.argsort(distances)
            rank_map = {int(pos): rank for rank, pos in enumerate(rank_order)}
            reward = float(rank_map[int(oracle_pos)] - rank_map[int(action)]) / max(len(sequence) - 1, 1)
            rollout.append((state, cands, action, old_lp, torch.tensor(reward, dtype=torch.float32, device=device)))
            rewards.append(float(reward))
        for start in range(0, len(rollout), int(budget.minibatch_size)):
            batch = rollout[start : start + int(budget.minibatch_size)]
            if not batch:
                continue
            loss, norm = ppo_update(policy, optimizer, batch)
            losses.append(loss)
            grad_norms.append(norm)
            steps += 1

    checkpoint = checkpoint_module(out_path, policy)
    after = params_vector(policy)
    delta = float(torch.norm(after - before).item())
    metrics = {
        "component": name,
        "trainable_parameter_count": count_trainable(policy),
        "valid_training_points": len(valid),
        "optimizer_steps": int(steps),
        "first_loss": None if not losses else float(losses[0]),
        "final_loss": None if not losses else float(losses[-1]),
        "first_reward": None if not rewards else float(rewards[0]),
        "final_reward": None if not rewards else float(rewards[-1]),
        "mean_reward": None if not rewards else float(np.mean(rewards)),
        "gradient_norms": grad_norms,
        "finite_nonzero_gradients": bool(any(np.isfinite(v) and abs(v) > 0.0 for v in grad_norms)),
        "weight_delta_l2": delta,
        "weights_changed": bool(delta > 0.0),
        "checkpoint": checkpoint,
        "action_sequence_oracle_used_for_supervision": True,
    }
    return policy, metrics, sets, sizes, graph


def top_policy_candidates(
    policy: ScoringMLP,
    q: np.ndarray,
    sequence: list[tuple[int, float, int]],
    emb: np.ndarray,
    action_lengths: np.ndarray,
    topk: int = FUSION_TOPK,
) -> list[int]:
    device = next(policy.parameters()).device
    state_np, cand_np = policy_features(q, sequence, emb, action_lengths)
    with torch.no_grad():
        scores = policy(
            torch.tensor(state_np, dtype=torch.float32, device=device),
            torch.tensor(cand_np, dtype=torch.float32, device=device),
        )
    order = scores.argsort(descending=True).detach().cpu().numpy()[: min(topk, len(sequence))]
    return [int(sequence[int(pos)][0]) for pos in order]


def train_action_fusion(
    name: str,
    policy: ScoringMLP,
    emb: np.ndarray,
    action_desc: np.ndarray,
    action_lengths: np.ndarray,
    sets: list[list[tuple[int, float, int]] | None],
    budget: TrainBudget,
    out_path: pathlib.Path,
) -> tuple[ActionFusionHead, dict[str, Any]]:
    device = next(policy.parameters()).device
    head = ActionFusionHead().to(device)
    head._constructor_args = ()  # type: ignore[attr-defined]
    before = params_vector(head)
    optimizer = torch.optim.Adam(head.parameters(), lr=budget.fusion_lr)
    losses: list[float] = []
    grad_norms: list[float] = []
    steps = 0
    valid = [i for i, seq in enumerate(sets) if seq is not None and len(seq) >= 2]
    policy.eval()
    for _epoch in range(int(budget.prediction_fusion_epochs)):
        for start in range(0, len(valid), int(budget.minibatch_size)):
            batch_ids = valid[start : start + int(budget.minibatch_size)]
            batch_losses = []
            for i in batch_ids:
                sequence = sets[i]
                if sequence is None:
                    continue
                top_ids = top_policy_candidates(policy, emb[i], sequence, emb, action_lengths)
                cand_emb = torch.tensor(emb[top_ids], dtype=torch.float32, device=device)
                cand_desc = torch.tensor(action_desc[top_ids], dtype=torch.float32, device=device)
                q = torch.tensor(emb[i], dtype=torch.float32, device=device)
                target = torch.tensor(action_desc[i], dtype=torch.float32, device=device)
                pred = head(q, cand_emb, cand_desc)
                batch_losses.append(F.mse_loss(pred, target))
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = grad_norm(head)
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(norm)
            steps += 1
    checkpoint = checkpoint_module(out_path, head)
    delta = float(torch.norm(params_vector(head) - before).item())
    return head, {
        "component": name,
        "trainable_parameter_count": count_trainable(head),
        "valid_training_points": len(valid),
        "optimizer_steps": int(steps),
        "first_loss": None if not losses else float(losses[0]),
        "final_loss": None if not losses else float(losses[-1]),
        "gradient_norms": grad_norms,
        "finite_nonzero_gradients": bool(any(np.isfinite(v) and abs(v) > 0.0 for v in grad_norms)),
        "weight_delta_l2": delta,
        "weights_changed": bool(delta > 0.0),
        "checkpoint": checkpoint,
    }


def imp_features(q_partial: np.ndarray, q_m: np.ndarray, sequence: list[tuple[int, float, int]], donor_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cand = np.stack([donor_emb[node] for node, _, _ in sequence])
    dists = np.linalg.norm(cand - q_m.reshape(1, -1), axis=1)
    max_dist = float(dists.max()) + 1e-9
    max_depth = max(depth for _, _, depth in sequence) + 1e-9
    sorted_by_dist = np.argsort(dists)
    rank = {int(pos): rank_i for rank_i, pos in enumerate(sorted_by_dist)}
    state = np.concatenate([q_partial, cand.mean(axis=0), [float(dists.var())], [float(len(sequence))]]).astype(np.float32)
    cfs = []
    for pos, (node, _dist, depth) in enumerate(sequence):
        cfs.append(
            np.concatenate(
                [
                    donor_emb[node],
                    [
                        float(dists[pos] / max_dist),
                        float(depth / max_depth),
                        float(rank[pos] / max(len(sequence) - 1, 1)),
                    ],
                ]
            ).astype(np.float32)
        )
    return state, np.stack(cfs).astype(np.float32)


def train_imputation_policy(
    raw: list[np.ndarray],
    stats: dict[int, tuple[np.ndarray, np.ndarray]],
    budget: TrainBudget,
    out_path: pathlib.Path,
    rng: np.random.RandomState,
) -> tuple[ScoringMLP, dict[str, Any], list[list[tuple[int, float]]]]:
    donor = raw[1]
    graph = build_knn_graph(donor)
    policy = ScoringMLP(FULL_DIM + CAM_DIM + 2, CAM_DIM + 3).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    policy._constructor_args = (FULL_DIM + CAM_DIM + 2, CAM_DIM + 3)  # type: ignore[attr-defined]
    before = params_vector(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=budget.imp_lr)
    device = next(policy.parameters()).device
    losses: list[float] = []
    rewards: list[float] = []
    grad_norms: list[float] = []
    steps = 0
    present = np.array([True, False, True], dtype=bool)
    train_items = list(range(donor.shape[0]))
    for _epoch in range(int(budget.imputation_ppo_epochs)):
        rng.shuffle(train_items)
        rollout = []
        policy.eval()
        for i in train_items:
            q_partial = partial_embedding([raw[0][i], raw[1][i], raw[2][i]], present, stats)
            q_m = donor[i]
            seeds = seed_neighbors(q_m, donor, exclude=i)
            sequence = [item for item in bfs(seeds, graph) if item[0] != i]
            if len(sequence) < 2:
                continue
            oracle_node = min((node for node, _, _ in sequence), key=lambda node: (float(np.sum((donor[node] - q_m) ** 2)), node))
            perm = rng.permutation(len(sequence))
            sequence = [sequence[int(pos)] for pos in perm]
            oracle_pos = int([node for node, _, _ in sequence].index(int(oracle_node)))
            state_np, cand_np = imp_features(q_partial, q_m, sequence, donor)
            state = torch.tensor(state_np, dtype=torch.float32, device=device)
            cands = torch.tensor(cand_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = policy(state, cands)
                log_pi = F.log_softmax(scores, dim=-1)
                action = int(torch.multinomial(log_pi.exp(), 1).item())
                old_lp = log_pi[action].detach()
            l2s = [float(np.sum((donor[node] - q_m) ** 2)) for node, _, _ in sequence]
            rank_order = np.argsort(l2s)
            rank_map = {int(pos): rank for rank, pos in enumerate(rank_order)}
            reward = float(rank_map[int(oracle_pos)] - rank_map[int(action)]) / max(len(sequence) - 1, 1)
            rollout.append((state, cands, action, old_lp, torch.tensor(reward, dtype=torch.float32, device=device)))
            rewards.append(float(reward))
        for start in range(0, len(rollout), int(budget.minibatch_size)):
            batch = rollout[start : start + int(budget.minibatch_size)]
            if not batch:
                continue
            loss, norm = ppo_update(policy, optimizer, batch)
            losses.append(loss)
            grad_norms.append(norm)
            steps += 1
    checkpoint = checkpoint_module(out_path, policy)
    delta = float(torch.norm(params_vector(policy) - before).item())
    return policy, {
        "component": "imputation_policy_mod1",
        "trainable_parameter_count": count_trainable(policy),
        "optimizer_steps": int(steps),
        "first_loss": None if not losses else float(losses[0]),
        "final_loss": None if not losses else float(losses[-1]),
        "first_reward": None if not rewards else float(rewards[0]),
        "final_reward": None if not rewards else float(rewards[-1]),
        "mean_reward": None if not rewards else float(np.mean(rewards)),
        "gradient_norms": grad_norms,
        "finite_nonzero_gradients": bool(any(np.isfinite(v) and abs(v) > 0.0 for v in grad_norms)),
        "weight_delta_l2": delta,
        "weights_changed": bool(delta > 0.0),
        "checkpoint": checkpoint,
    }, graph


def select_imputation_donors(policy: ScoringMLP, q_partial: np.ndarray, q_m_seed: np.ndarray, donor: np.ndarray, graph: list[list[tuple[int, float]]]) -> list[int]:
    seeds = seed_neighbors(q_m_seed, donor, exclude=None)
    sequence = bfs(seeds, graph)
    if not sequence:
        return list(range(min(IMP_SOFT_TOPK, donor.shape[0])))
    state_np, cand_np = imp_features(q_partial, q_m_seed, sequence, donor)
    device = next(policy.parameters()).device
    with torch.no_grad():
        scores = policy(
            torch.tensor(state_np, dtype=torch.float32, device=device),
            torch.tensor(cand_np, dtype=torch.float32, device=device),
        )
    order = scores.argsort(descending=True).detach().cpu().numpy()[: min(IMP_SOFT_TOPK, len(sequence))]
    return [int(sequence[int(pos)][0]) for pos in order]


def train_soft_imputation(
    policy: ScoringMLP,
    raw: list[np.ndarray],
    stats: dict[int, tuple[np.ndarray, np.ndarray]],
    graph: list[list[tuple[int, float]]],
    budget: TrainBudget,
    out_path: pathlib.Path,
) -> tuple[SoftImputationHead, dict[str, Any]]:
    device = next(policy.parameters()).device
    donor = raw[1]
    head = SoftImputationHead().to(device)
    head._constructor_args = ()  # type: ignore[attr-defined]
    before = params_vector(head)
    optimizer = torch.optim.Adam(head.parameters(), lr=budget.soft_imp_lr)
    losses: list[float] = []
    grad_norms: list[float] = []
    steps = 0
    present = np.array([True, False, True], dtype=bool)
    policy.eval()
    for _epoch in range(int(budget.soft_imputation_epochs)):
        for start in range(0, donor.shape[0], int(budget.minibatch_size)):
            batch_losses = []
            for i in range(start, min(start + int(budget.minibatch_size), donor.shape[0])):
                q_partial = partial_embedding([raw[0][i], raw[1][i], raw[2][i]], present, stats)
                top_ids = select_imputation_donors(policy, q_partial, donor[i], donor, graph)
                pred = head(
                    torch.tensor(q_partial, dtype=torch.float32, device=device),
                    torch.tensor(donor[top_ids], dtype=torch.float32, device=device),
                )
                target = torch.tensor(donor[i], dtype=torch.float32, device=device)
                batch_losses.append(F.mse_loss(pred, target))
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = grad_norm(head)
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(norm)
            steps += 1
    checkpoint = checkpoint_module(out_path, head)
    delta = float(torch.norm(params_vector(head) - before).item())
    return head, {
        "component": "soft_imputation_head_mod1",
        "trainable_parameter_count": count_trainable(head),
        "optimizer_steps": int(steps),
        "first_loss": None if not losses else float(losses[0]),
        "final_loss": None if not losses else float(losses[-1]),
        "gradient_norms": grad_norms,
        "finite_nonzero_gradients": bool(any(np.isfinite(v) and abs(v) > 0.0 for v in grad_norms)),
        "weight_delta_l2": delta,
        "weights_changed": bool(delta > 0.0),
        "checkpoint": checkpoint,
    }


def impute_dataset(raw: list[np.ndarray], stats: dict[int, tuple[np.ndarray, np.ndarray]], policy: ScoringMLP, head: SoftImputationHead, graph: list[list[tuple[int, float]]]) -> np.ndarray:
    donor = raw[1]
    present = np.array([True, False, True], dtype=bool)
    full_present = np.ones(3, dtype=bool)
    rows = []
    zero = np.zeros(CAM_DIM, dtype=np.float32)
    device = next(policy.parameters()).device
    policy.eval()
    head.eval()
    for i in range(raw[0].shape[0]):
        q_partial = partial_embedding([raw[0][i], raw[1][i], raw[2][i]], present, stats)
        top_ids = select_imputation_donors(policy, q_partial, zero, donor, graph)
        with torch.no_grad():
            imputed = (
                head(
                    torch.tensor(q_partial, dtype=torch.float32, device=device),
                    torch.tensor(donor[top_ids], dtype=torch.float32, device=device),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        rows.append(partial_embedding([raw[0][i], imputed, raw[2][i]], full_present, stats))
    return np.stack(rows).astype(np.float32)


def save_bundle(path: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {"path": str(path), "sha256": sha256_file(path)}


def load_action_library(dataset_root: pathlib.Path, task: dict[str, Any]) -> dict[str, np.ndarray]:
    path = dataset_root / str(task["suite"]) / str(task["hdf5"])
    actions: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h:
        for key in sorted(h["data"].keys(), key=demo_index):
            actions[key] = np.asarray(h["data"][key]["actions"], dtype=np.float32)
    return actions


def load_state_module(module: nn.Module, path: pathlib.Path, device: torch.device) -> nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    module.load_state_dict(state)
    module.to(device)
    module.eval()
    return module


def load_task_port(training_dir: pathlib.Path, task: dict[str, Any], device: torch.device) -> dict[str, Any]:
    task_name = safe_task_name(str(task["suite"]), int(task["task_id"]))
    task_dir = training_dir / task_name
    bundle_path = task_dir / "bundle.npz"
    bundle = np.load(bundle_path, allow_pickle=False)
    stats = {
        0: (bundle["stats_cam0_mu"], bundle["stats_cam0_sigma"]),
        1: (bundle["stats_cam1_mu"], bundle["stats_cam1_sigma"]),
        2: (bundle["stats_lang_mu"], bundle["stats_lang_sigma"]),
    }
    port = {
        "task_name": task_name,
        "task_dir": task_dir,
        "bundle_path": bundle_path,
        "bundle_sha256": sha256_file(bundle_path),
        "train_keys": [str(x) for x in bundle["train_keys"].tolist()],
        "train_clean_emb": bundle["train_clean_emb"].astype(np.float32),
        "train_mask1_emb": bundle["train_mask1_emb"].astype(np.float32),
        "train_raw_cam1": bundle["train_raw_cam1"].astype(np.float32),
        "action_desc": bundle["action_desc"].astype(np.float32),
        "action_lengths": bundle["action_lengths"].astype(np.float32),
        "stats": stats,
        "clean_graph": build_knn_graph(bundle["train_clean_emb"].astype(np.float32)),
        "mask1_graph": build_knn_graph(bundle["train_mask1_emb"].astype(np.float32)),
        "imp_graph": build_knn_graph(bundle["train_raw_cam1"].astype(np.float32)),
        "clean_policy": load_state_module(ScoringMLP(FULL_DIM + 2, FULL_DIM + 4), task_dir / "clean_retrieval_policy.pt", device),
        "clean_fusion": load_state_module(ActionFusionHead(), task_dir / "clean_action_fusion_head.pt", device),
        "mask1_policy": load_state_module(ScoringMLP(FULL_DIM + 2, FULL_DIM + 4), task_dir / "mask1_retrieval_policy.pt", device),
        "mask1_fusion": load_state_module(ActionFusionHead(), task_dir / "mask1_action_fusion_head.pt", device),
        "imp_policy": load_state_module(ScoringMLP(FULL_DIM + CAM_DIM + 2, CAM_DIM + 3), task_dir / "imputation_policy_mod1.pt", device),
        "soft_imp": load_state_module(SoftImputationHead(), task_dir / "soft_imputation_head_mod1.pt", device),
        "checkpoint_hashes": {
            "clean_retrieval_policy": sha256_file(task_dir / "clean_retrieval_policy.pt"),
            "clean_action_fusion_head": sha256_file(task_dir / "clean_action_fusion_head.pt"),
            "mask1_retrieval_policy": sha256_file(task_dir / "mask1_retrieval_policy.pt"),
            "mask1_action_fusion_head": sha256_file(task_dir / "mask1_action_fusion_head.pt"),
            "imputation_policy_mod1": sha256_file(task_dir / "imputation_policy_mod1.pt"),
            "soft_imputation_head_mod1": sha256_file(task_dir / "soft_imputation_head_mod1.pt"),
        },
    }
    return port


def obs_agent_rgb(obs: dict[str, Any]) -> np.ndarray:
    if "agentview_image" in obs:
        return np.asarray(obs["agentview_image"], dtype=np.uint8)
    if "agentview_rgb" in obs:
        return np.asarray(obs["agentview_rgb"], dtype=np.uint8)
    raise KeyError("live obs has no agentview image")


def obs_wrist_rgb(obs: dict[str, Any]) -> np.ndarray:
    if "robot0_eye_in_hand_image" in obs:
        return np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
    if "eye_in_hand_rgb" in obs:
        return np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
    if "robot0_eye_in_hand_rgb" in obs:
        return np.asarray(obs["robot0_eye_in_hand_rgb"], dtype=np.uint8)
    raise KeyError("live obs has no wrist/in-hand image")


def encode_live_query(
    clip: FrozenCLIPEncoder,
    port: dict[str, Any],
    obs: dict[str, Any],
    instruction: str,
    condition: str,
    counts: dict[str, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    cam0 = obs_agent_rgb(obs)
    c0 = clip.encode_image([[cam0]])
    counts["clip_image_forward_count"] = counts.get("clip_image_forward_count", 0) + 1
    lt = clip.encode_text([instruction])
    counts["clip_text_forward_count"] = counts.get("clip_text_forward_count", 0) + 1
    stats = port["stats"]
    if condition == "clean":
        cam1 = obs_wrist_rgb(obs)
        c1 = clip.encode_image([[cam1]])
        counts["clip_image_forward_count"] = counts.get("clip_image_forward_count", 0) + 1
        q = partial_embedding([c0[0], c1[0], lt[0]], np.ones(3, dtype=bool), stats)
        return q, {
            "mask1_imputation_used": False,
            "wrist_input_used": True,
            "agent_rgb_shape": list(cam0.shape),
            "wrist_rgb_shape": list(cam1.shape),
        }

    if condition != "mask_1_in_hand_dropout":
        raise ValueError(f"unknown rollout condition {condition!r}")

    present = np.array([True, False, True], dtype=bool)
    q_partial = partial_embedding([c0[0], np.zeros(CAM_DIM, dtype=np.float32), lt[0]], present, stats)
    zero = np.zeros(CAM_DIM, dtype=np.float32)
    top_ids = select_imputation_donors(port["imp_policy"], q_partial, zero, port["train_raw_cam1"], port["imp_graph"])
    counts["imputation_policy_forward_count"] = counts.get("imputation_policy_forward_count", 0) + 1
    device = next(port["soft_imp"].parameters()).device
    with torch.no_grad():
        imputed = (
            port["soft_imp"](
                torch.tensor(q_partial, dtype=torch.float32, device=device),
                torch.tensor(port["train_raw_cam1"][top_ids], dtype=torch.float32, device=device),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    counts["soft_imputation_forward_count"] = counts.get("soft_imputation_forward_count", 0) + 1
    q = partial_embedding([c0[0], imputed, lt[0]], np.ones(3, dtype=bool), stats)
    return q, {
        "mask1_imputation_used": True,
        "wrist_input_used": False,
        "imputation_donor_indices": [int(x) for x in top_ids],
        "agent_rgb_shape": list(cam0.shape),
        "wrist_rgb_shape": None,
    }


def retrieve_action_sequence(
    port: dict[str, Any],
    condition: str,
    q_emb: np.ndarray,
    action_library: dict[str, np.ndarray],
    counts: dict[str, int],
) -> tuple[str, np.ndarray, dict[str, Any]]:
    if condition == "clean":
        train_emb = port["train_clean_emb"]
        graph = port["clean_graph"]
        policy = port["clean_policy"]
        fusion = port["clean_fusion"]
    else:
        train_emb = port["train_mask1_emb"]
        graph = port["mask1_graph"]
        policy = port["mask1_policy"]
        fusion = port["mask1_fusion"]
    seeds = seed_neighbors(q_emb, train_emb, exclude=None)
    sequence = bfs(seeds, graph)
    if not sequence:
        nearest = int(NearestNeighbors(n_neighbors=1, metric="euclidean").fit(train_emb).kneighbors(q_emb.reshape(1, -1))[1][0][0])
        key = port["train_keys"][nearest]
        return key, action_library[key], {"fallback": "nearest_neighbor_no_bfs", "candidate_count": 1}
    top_ids = top_policy_candidates(policy, q_emb, sequence, train_emb, port["action_lengths"])
    counts["retrieval_policy_forward_count"] = counts.get("retrieval_policy_forward_count", 0) + 1
    device = next(fusion.parameters()).device
    with torch.no_grad():
        pred_desc = (
            fusion(
                torch.tensor(q_emb, dtype=torch.float32, device=device),
                torch.tensor(train_emb[top_ids], dtype=torch.float32, device=device),
                torch.tensor(port["action_desc"][top_ids], dtype=torch.float32, device=device),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    counts["action_fusion_forward_count"] = counts.get("action_fusion_forward_count", 0) + 1
    cand_desc = port["action_desc"][top_ids]
    descriptor_mse = np.mean((cand_desc - pred_desc.reshape(1, -1)) ** 2, axis=1)
    chosen_pos = int(np.argmin(descriptor_mse))
    chosen_id = int(top_ids[chosen_pos])
    key = port["train_keys"][chosen_id]
    return key, action_library[key], {
        "fallback": None,
        "candidate_count": int(len(sequence)),
        "top_candidate_indices": [int(x) for x in top_ids],
        "chosen_train_index": int(chosen_id),
        "chosen_demo_key": key,
        "chosen_descriptor_mse": float(descriptor_mse[chosen_pos]),
        "top_descriptor_mse": [float(x) for x in descriptor_mse[: min(5, len(descriptor_mse))]],
    }


def run_one_rollout_episode(
    env_cls: Any,
    bddl_file: pathlib.Path,
    init_state: np.ndarray,
    identity: int,
    port: dict[str, Any],
    action_library: dict[str, np.ndarray],
    clip: FrozenCLIPEncoder,
    task: dict[str, Any],
    condition: str,
    max_steps: int,
    settle_steps: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "execution_classification": "PRIOR_CLOSED_LOOP_ROLLOUT",
        "implementation_label": IMPLEMENTATION_LABEL,
        "suite": task["suite"],
        "task_id": int(task["task_id"]),
        "instruction": str(task["instruction"]),
        "reset_identity": int(identity),
        "condition": condition,
        "completed": False,
        "success": False,
        "exception": None,
        "module_forward_counts": {},
        "expert_action_replay_counted_as_prior_success": False,
        "live_inference_uses_target_or_future_action": False,
    }
    env = None
    started = time.monotonic()
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=128, camera_widths=128)
        env.seed(int(identity))
        env.reset()
        obs = env.set_init_state(np.asarray(init_state, dtype=np.float64))
        row["set_init_state_ok"] = True
        for _ in range(int(settle_steps)):
            obs, _reward, _done, _info = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        q_emb, query_meta = encode_live_query(clip, port, obs, str(task["instruction"]), condition, row["module_forward_counts"])
        selected_key, actions, retrieval_meta = retrieve_action_sequence(
            port, condition, q_emb, action_library, row["module_forward_counts"]
        )
        row["query"] = query_meta
        row["retrieval"] = retrieval_meta
        row["retrieved_demo_key"] = selected_key
        row["retrieved_action_shape"] = [int(x) for x in actions.shape]
        row["action_sequences_produced_through_frozen_rl4il_port"] = True
        final_reward = 0.0
        done_seen = False
        success_seen = False
        steps = 0
        for step, action in enumerate(actions[: int(max_steps)]):
            obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
            final_reward = float(reward)
            steps = step + 1
            info_success = bool(isinstance(info, dict) and info.get("success", False))
            try:
                check_success = bool(env.check_success())
            except Exception:
                check_success = False
            success_seen = bool(success_seen or info_success or check_success or float(reward) > 0.0)
            done_seen = bool(done_seen or done)
            if success_seen or done:
                break
        row.update(
            {
                "completed": True,
                "success": bool(success_seen or done_seen),
                "done_seen": bool(done_seen),
                "final_reward": float(final_reward),
                "steps": int(steps),
                "max_steps": int(max_steps),
                "settle_steps": int(settle_steps),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
    except Exception as exc:  # pragma: no cover - simulator boundary
        row["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-80:],
        }
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return row


def run_rollout(args: argparse.Namespace) -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_dir = pathlib.Path(args.run_dir)
    rollout_dir = run_dir / "prior_closed_loop_rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    status_path = rollout_dir / "prior_closed_loop_rollout_status.txt"
    heartbeat_path = rollout_dir / "prior_closed_loop_rollout_heartbeat.txt"
    status_path.write_text("starting\n", encoding="utf-8")
    heartbeat_path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": "2026-07-18.epoch5_rl4il_action_oracle_prior_closed_loop_rollout.v1",
        "execution_classification": "PRIOR_CLOSED_LOOP_ROLLOUT",
        "implementation_label": IMPLEMENTATION_LABEL,
        "not_official_rl4il_reproduction": True,
        "not_vla_training": True,
        "not_ours": True,
        "run_dir": str(run_dir),
        "training_dir": str(args.training_dir),
        "rollout_dir": str(rollout_dir),
        "conditions": ["clean", "mask_1_in_hand_dropout"],
        "identity_base": IDENTITY_BASE,
        "max_steps": int(args.max_steps),
        "settle_steps": int(args.settle_steps),
        "episodes": [],
        "exceptions": [],
        "expert_action_replay_counted_as_prior_success": False,
        "live_inference_uses_target_or_future_action": False,
        "frozen_panel_preserved": True,
    }
    try:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark_dict
        from libero.libero.envs import OffScreenRenderEnv

        clip = FrozenCLIPEncoder(device)
        result["frozen_clip"] = {
            "model": "openai/clip-vit-base-patch32",
            "trainable_parameter_count": count_trainable(clip),
            "all_parameters_frozen": bool(count_trainable(clip) == 0),
        }
        benchmark = get_benchmark_dict()
        for task in PANEL:
            status_path.write_text(
                f"rollout {safe_task_name(str(task['suite']), int(task['task_id']))}\n",
                encoding="utf-8",
            )
            heartbeat_path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
            port = load_task_port(pathlib.Path(args.training_dir), task, device)
            action_library = load_action_library(pathlib.Path(args.dataset_root), task)
            task_suite = benchmark[str(task["suite"])]()
            libero_task = task_suite.get_task(int(task["task_id"]))
            bddl_file = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
            initial_states = task_suite.get_task_init_states(int(task["task_id"]))
            for identity in task["identities"]:
                index = int(identity) - IDENTITY_BASE
                for condition in ["clean", "mask_1_in_hand_dropout"]:
                    episode = run_one_rollout_episode(
                        OffScreenRenderEnv,
                        bddl_file,
                        np.asarray(initial_states[index]).reshape(-1),
                        int(identity),
                        port,
                        action_library,
                        clip,
                        task,
                        condition,
                        int(args.max_steps),
                        int(args.settle_steps),
                    )
                    episode["initial_state_index"] = int(index)
                    result["episodes"].append(episode)
                    if episode.get("exception"):
                        result["exceptions"].append(episode["exception"])
                    write_json(rollout_dir / f"{safe_task_name(str(task['suite']), int(task['task_id']))}_{identity}_{condition}.json", episode)
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - environment boundary
        result["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-100:],
            }
        )

    episodes = result["episodes"]
    clean_success = sum(1 for ep in episodes if ep.get("condition") == "clean" and ep.get("success"))
    dropout_success = sum(1 for ep in episodes if ep.get("condition") == "mask_1_in_hand_dropout" and ep.get("success"))
    clean_count = sum(1 for ep in episodes if ep.get("condition") == "clean")
    dropout_count = sum(1 for ep in episodes if ep.get("condition") == "mask_1_in_hand_dropout")
    completed_count = sum(1 for ep in episodes if ep.get("completed"))
    module_forward_count = sum(
        int(value)
        for ep in episodes
        for value in (ep.get("module_forward_counts") or {}).values()
    )
    valid = bool(
        not result["exceptions"]
        and len(episodes) == 18
        and clean_count == 9
        and dropout_count == 9
        and completed_count == 18
        and module_forward_count > 0
        and all(bool(ep.get("action_sequences_produced_through_frozen_rl4il_port")) for ep in episodes)
        and all(not bool(ep.get("live_inference_uses_target_or_future_action")) for ep in episodes)
        and all(not bool(ep.get("expert_action_replay_counted_as_prior_success")) for ep in episodes)
    )
    if not valid:
        decision = "RL4IL_ACTION_ORACLE_PRIOR_EVALUATION_INVALID"
    elif dropout_success == 0:
        decision = "RL4IL_ACTION_ORACLE_PRIOR_NO_LOCAL_IMPROVEMENT"
    elif dropout_success >= 9:
        decision = "RL4IL_ACTION_ORACLE_PRIOR_SATURATES_CONDITION"
    else:
        decision = "RL4IL_ACTION_ORACLE_PRIOR_LOCAL_RESIDUAL_ESTABLISHED"
    result["aggregate"] = {
        "episode_count": len(episodes),
        "completed_episode_count": int(completed_count),
        "clean_episode_count": int(clean_count),
        "dropout_episode_count": int(dropout_count),
        "clean_success_count": int(clean_success),
        "dropout_success_count": int(dropout_success),
        "module_forward_count": int(module_forward_count),
        "valid": bool(valid),
    }
    result["decision"] = decision
    result["success"] = bool(valid)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    result["cuda"] = cuda_report()
    result["system_ram"] = memory_report()
    result_path = rollout_dir / "prior_closed_loop_rollout_result.json"
    write_json(result_path, result)
    write_rollout_markdown(rollout_dir / "prior_closed_loop_rollout_result.md", result)
    shutil.copy2(result_path, pathlib.Path("reports/rl4il_action_oracle_prior_closed_loop_rollout_result.json"))
    shutil.copy2(rollout_dir / "prior_closed_loop_rollout_result.md", pathlib.Path("reports/rl4il_action_oracle_prior_closed_loop_rollout_result.md"))
    heartbeat_path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    status_path.write_text(("complete\n" if valid else "failed\n"), encoding="utf-8")
    (rollout_dir / "prior_closed_loop_rollout_exit_code.txt").write_text(("0\n" if valid else "2\n"), encoding="utf-8")
    return 0 if valid else 2


def write_rollout_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    agg = result.get("aggregate", {})
    lines = [
        "# RL4IL Action-Oracle Prior Closed-Loop Rollout Result",
        "",
        f"- Execution classification: `{result.get('execution_classification')}`",
        f"- Implementation label: `{result.get('implementation_label')}`",
        f"- Decision: `{result.get('decision')}`",
        f"- Valid: `{agg.get('valid')}`",
        f"- Clean successes: `{agg.get('clean_success_count')}/{agg.get('clean_episode_count')}`",
        f"- mask_1 successes: `{agg.get('dropout_success_count')}/{agg.get('dropout_episode_count')}`",
        f"- Module forward count: `{agg.get('module_forward_count')}`",
        f"- Peak VRAM MiB: `{(result.get('cuda') or {}).get('max_allocated_mib')}`",
        "",
        "This is an external RL4IL retrieval/imputation prior rollout, not Ours and not VLA fine-tuning.",
        "",
        "| suite/task | identity | condition | success | steps | retrieved demo | imputation |",
        "|---|---:|---|---|---:|---|---|",
    ]
    for ep in result.get("episodes", []):
        query = ep.get("query") or {}
        lines.append(
            f"| `{ep.get('suite')}/task{ep.get('task_id')}` | {ep.get('reset_identity')} | `{ep.get('condition')}` | "
            f"`{ep.get('success')}` | {ep.get('steps')} | `{ep.get('retrieved_demo_key')}` | `{query.get('mask1_imputation_used')}` |"
        )
    if result.get("exceptions"):
        lines += ["", "## Exceptions", "", "```json", json.dumps(result["exceptions"], indent=2), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_one_task(task: dict[str, Any], dataset_root: pathlib.Path, out_dir: pathlib.Path, clip: FrozenCLIPEncoder, budget: TrainBudget) -> dict[str, Any]:
    started = time.monotonic()
    task_name = safe_task_name(str(task["suite"]), int(task["task_id"]))
    task_dir = out_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED + int(task["task_id"]))
    demos = load_task_demos(dataset_root, task)
    train_demos = [demo for demo in demos if demo.index < 40]
    val_demos = [demo for demo in demos if demo.index >= 40]
    c0, c1, lt = extract_demo_embeddings(train_demos, clip)
    raw = [c0, c1, lt]
    stats = compute_stats(raw)
    clean_emb = full_embeddings(raw, stats)
    action_desc = np.stack([demo.action_descriptor for demo in train_demos]).astype(np.float32)
    action_lengths = np.asarray([demo.actions.shape[0] for demo in train_demos], dtype=np.float32)
    action_dist = pairwise_action_distance_matrix([demo.actions for demo in train_demos])

    clean_policy, clean_policy_metrics, clean_sets, _clean_sizes, clean_graph = train_policy_module(
        "clean_retrieval_policy",
        clean_emb,
        action_desc,
        action_lengths,
        action_dist,
        budget,
        task_dir / "clean_retrieval_policy.pt",
        rng,
    )
    clean_fusion, clean_fusion_metrics = train_action_fusion(
        "clean_action_fusion_head",
        clean_policy,
        clean_emb,
        action_desc,
        action_lengths,
        clean_sets,
        budget,
        task_dir / "clean_action_fusion_head.pt",
    )
    imp_policy, imp_policy_metrics, imp_graph = train_imputation_policy(raw, stats, budget, task_dir / "imputation_policy_mod1.pt", rng)
    soft_imp, soft_imp_metrics = train_soft_imputation(
        imp_policy,
        raw,
        stats,
        imp_graph,
        budget,
        task_dir / "soft_imputation_head_mod1.pt",
    )
    mask1_emb = impute_dataset(raw, stats, imp_policy, soft_imp, imp_graph)
    mask1_policy, mask1_policy_metrics, mask1_sets, _mask1_sizes, mask1_graph = train_policy_module(
        "mask1_retrieval_policy",
        mask1_emb,
        action_desc,
        action_lengths,
        action_dist,
        budget,
        task_dir / "mask1_retrieval_policy.pt",
        rng,
    )
    mask1_fusion, mask1_fusion_metrics = train_action_fusion(
        "mask1_action_fusion_head",
        mask1_policy,
        mask1_emb,
        action_desc,
        action_lengths,
        mask1_sets,
        budget,
        task_dir / "mask1_action_fusion_head.pt",
    )

    bundle = save_bundle(
        task_dir / "bundle.npz",
        {
            "train_keys": np.asarray([demo.key for demo in train_demos]),
            "train_indices": np.asarray([demo.index for demo in train_demos], dtype=np.int32),
            "train_clean_emb": clean_emb,
            "train_mask1_emb": mask1_emb,
            "train_raw_cam0": raw[0],
            "train_raw_cam1": raw[1],
            "train_raw_lang": raw[2],
            "action_desc": action_desc,
            "action_lengths": action_lengths,
            "stats_cam0_mu": stats[0][0],
            "stats_cam0_sigma": stats[0][1],
            "stats_cam1_mu": stats[1][0],
            "stats_cam1_sigma": stats[1][1],
            "stats_lang_mu": stats[2][0],
            "stats_lang_sigma": stats[2][1],
        },
    )
    metadata = {
        "task": task,
        "task_name": task_name,
        "train_demo_count": len(train_demos),
        "validation_demo_count": len(val_demos),
        "train_demo_indices": [int(demo.index) for demo in train_demos],
        "validation_demo_indices": [int(demo.index) for demo in val_demos],
        "bundle": bundle,
        "clean_graph_node_count": len(clean_graph),
        "mask1_graph_node_count": len(mask1_graph),
        "imputation_graph_node_count": len(imp_graph),
        "component_metrics": [
            clean_policy_metrics,
            clean_fusion_metrics,
            imp_policy_metrics,
            soft_imp_metrics,
            mask1_policy_metrics,
            mask1_fusion_metrics,
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(task_dir / "task_training_metadata.json", metadata)
    del clean_policy, clean_fusion, imp_policy, soft_imp, mask1_policy, mask1_fusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def run_train(args: argparse.Namespace) -> int:
    run_dir = pathlib.Path(args.run_dir)
    train_dir = run_dir / "prior_module_training"
    train_dir.mkdir(parents=True, exist_ok=True)
    status_path = train_dir / "prior_module_training_status.txt"
    heartbeat_path = train_dir / "prior_module_training_heartbeat.txt"
    status_path.write_text("starting\n", encoding="utf-8")
    heartbeat_path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "schema_version": "2026-07-18.epoch5_rl4il_action_oracle_prior_module_training.v1",
        "execution_classification": "PRIOR_MODULE_TRAINING",
        "implementation_label": IMPLEMENTATION_LABEL,
        "not_official_rl4il_reproduction": True,
        "not_vla_training": True,
        "not_ours": True,
        "run_dir": str(run_dir),
        "training_dir": str(train_dir),
        "dataset_root": str(args.dataset_root),
        "pid": int(os.getpid()),
        "cuda_pid": int(os.getpid()) if torch.cuda.is_available() else None,
        "budget": TrainBudget().__dict__,
        "deviations_from_official_release": [
            "constant scalar labels are replaced with action-sequence-oracle supervision",
            "local port is task-panel-specific rather than a full 10-task suite reproduction",
            "fusion head predicts an action-sequence descriptor for retrieval instead of the released scalar label head",
            "only mask_1 in-hand-camera imputation is trained because the frozen condition is wrist/in-hand dropout",
        ],
        "action_sequence_oracle_usage": "training supervision only; live inference receives camera/language observations and never receives target/future action sequences",
        "module_forward_counts": {},
        "tasks": [],
        "exceptions": [],
    }
    started = time.monotonic()
    try:
        clip = FrozenCLIPEncoder(device)
        clip_trainable = count_trainable(clip)
        result["frozen_clip"] = {
            "model": "openai/clip-vit-base-patch32",
            "trainable_parameter_count": clip_trainable,
            "all_parameters_frozen": bool(clip_trainable == 0),
        }
        for task in PANEL:
            heartbeat_path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
            status_path.write_text(
                f"training {safe_task_name(str(task['suite']), int(task['task_id']))}\n",
                encoding="utf-8",
            )
            task_result = train_one_task(task, pathlib.Path(args.dataset_root), train_dir, clip, TrainBudget())
            result["tasks"].append(task_result)
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - environment boundary
        result["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-80:],
            }
        )
    components = [comp for task in result["tasks"] for comp in task.get("component_metrics", [])]
    result["aggregate"] = {
        "task_count": len(result["tasks"]),
        "component_count": len(components),
        "trainable_parameter_count": int(sum(int(comp.get("trainable_parameter_count") or 0) for comp in components)),
        "optimizer_steps": int(sum(int(comp.get("optimizer_steps") or 0) for comp in components)),
        "finite_nonzero_gradients": bool(components and all(bool(comp.get("finite_nonzero_gradients")) for comp in components)),
        "weights_changed": bool(components and all(bool(comp.get("weights_changed")) for comp in components)),
        "checkpoint_reload_ok": bool(
            components
            and all(bool((comp.get("checkpoint") or {}).get("reload_ok")) for comp in components if comp.get("checkpoint"))
        ),
    }
    result["success"] = bool(
        not result["exceptions"]
        and bool((result.get("frozen_clip") or {}).get("all_parameters_frozen"))
        and result["aggregate"]["task_count"] == len(PANEL)
        and result["aggregate"]["trainable_parameter_count"] > 0
        and result["aggregate"]["optimizer_steps"] > 0
        and result["aggregate"]["finite_nonzero_gradients"]
        and result["aggregate"]["weights_changed"]
        and result["aggregate"]["checkpoint_reload_ok"]
    )
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    result["cuda"] = cuda_report()
    result["system_ram"] = memory_report()
    result_path = train_dir / "prior_module_training_result.json"
    write_json(result_path, result)
    write_training_markdown(train_dir / "prior_module_training_result.md", result)
    shutil.copy2(result_path, pathlib.Path("reports/rl4il_action_oracle_prior_module_training_result.json"))
    shutil.copy2(train_dir / "prior_module_training_result.md", pathlib.Path("reports/rl4il_action_oracle_prior_module_training_result.md"))
    heartbeat_path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    status_path.write_text(("complete\n" if result["success"] else "failed\n"), encoding="utf-8")
    (train_dir / "prior_module_training_exit_code.txt").write_text(("0\n" if result["success"] else "2\n"), encoding="utf-8")
    return 0 if result["success"] else 2


def write_training_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    agg = result.get("aggregate", {})
    lines = [
        "# RL4IL Action-Oracle Prior Module Training Result",
        "",
        f"- Execution classification: `{result.get('execution_classification')}`",
        f"- Implementation label: `{result.get('implementation_label')}`",
        f"- Success: `{result.get('success')}`",
        f"- Trainable prior parameters: `{agg.get('trainable_parameter_count')}`",
        f"- Optimizer steps: `{agg.get('optimizer_steps')}`",
        f"- Finite nonzero gradients: `{agg.get('finite_nonzero_gradients')}`",
        f"- Weights changed: `{agg.get('weights_changed')}`",
        f"- Checkpoint reload OK: `{agg.get('checkpoint_reload_ok')}`",
        f"- CUDA PID: `{result.get('cuda_pid')}`",
        f"- Peak VRAM MiB: `{(result.get('cuda') or {}).get('max_allocated_mib')}`",
        "",
        "This is external RL4IL retrieval/imputation prior-module training, not VLA training and not Ours.",
        "",
        "## Component summary",
        "",
        "| task | component | params | steps | first loss | final loss | grad nonzero | changed | checkpoint |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for task in result.get("tasks", []):
        for comp in task.get("component_metrics", []):
            ckpt = (comp.get("checkpoint") or {}).get("path")
            lines.append(
                f"| `{task.get('task_name')}` | `{comp.get('component')}` | {comp.get('trainable_parameter_count')} | "
                f"{comp.get('optimizer_steps')} | {comp.get('first_loss')} | {comp.get('final_loss')} | "
                f"`{comp.get('finite_nonzero_gradients')}` | `{comp.get('weights_changed')}` | `{ckpt}` |"
            )
    if result.get("exceptions"):
        lines += ["", "## Exceptions", "", "```json", json.dumps(result["exceptions"], indent=2), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--run-dir", required=True)
    train.add_argument("--dataset-root", default="/mnt/c/assets/data/libero")
    rollout = sub.add_parser("rollout")
    rollout.add_argument("--run-dir", required=True)
    rollout.add_argument("--training-dir", required=True)
    rollout.add_argument("--dataset-root", default="/mnt/c/assets/data/libero")
    rollout.add_argument("--max-steps", type=int, default=260)
    rollout.add_argument("--settle-steps", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "train":
        return run_train(args)
    if args.cmd == "rollout":
        return run_rollout(args)
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
