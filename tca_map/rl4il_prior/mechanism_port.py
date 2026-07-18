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
import resource
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
    ru = resource.getrusage(resource.RUSAGE_SELF)
    report: dict[str, Any] = {"ru_maxrss_kib": int(ru.ru_maxrss)}
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "train":
        return run_train(args)
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
