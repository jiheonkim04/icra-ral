"""Download/load smoke for the official OpenPI pi0.5 LIBERO checkpoint.

This is intentionally not a LIBERO rollout.  It verifies that the local
OpenPI source environment can download the official checkpoint, instantiate the
`pi05_libero` policy, and produce one action chunk for OpenPI's own random
LIBERO example.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--checkpoint",
        default="gs://openpi-assets/checkpoints/pi05_libero",
    )
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    started = time.time()
    payload: dict[str, object] = {
        "schema_version": 1,
        "command": "openpi_pi05_policy_smoke",
        "checkpoint": args.checkpoint,
        "config_name": args.config_name,
        "seed": args.seed,
        "openpi_data_home": os.environ.get("OPENPI_DATA_HOME"),
        "success": False,
        "exception": None,
    }

    try:
        import numpy as np

        np.random.seed(args.seed)

        import jax
        import torch

        from openpi.policies import libero_policy
        from openpi.policies import policy_config
        from openpi.training import config as train_config

        payload["jax_version"] = getattr(jax, "__version__", "unknown")
        payload["jax_devices"] = [str(device) for device in jax.devices()]
        payload["torch_version"] = getattr(torch, "__version__", "unknown")
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["torch_cuda_device_name"] = torch.cuda.get_device_name(0)

        cfg = train_config.get_config(args.config_name)
        payload["config"] = {
            "name": cfg.name,
            "model_class": type(cfg.model).__name__,
            "data_class": type(cfg.data).__name__,
            "weight_loader_class": type(cfg.weight_loader).__name__,
            "action_horizon": getattr(cfg.model, "action_horizon", None),
            "action_dim": getattr(cfg.model, "action_dim", None),
        }

        load_started = time.time()
        policy = policy_config.create_trained_policy(cfg, args.checkpoint)
        payload["load_seconds"] = round(time.time() - load_started, 3)
        payload["policy_class"] = type(policy).__name__

        example = libero_policy.make_libero_example()
        infer_started = time.time()
        result = policy.infer(example)
        payload["infer_seconds"] = round(time.time() - infer_started, 3)

        actions = np.asarray(result["actions"])
        payload["actions"] = {
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "finite": bool(np.isfinite(actions).all()),
            "min": float(np.min(actions)),
            "max": float(np.max(actions)),
            "mean": float(np.mean(actions)),
        }
        payload["success"] = True
    except Exception as exc:  # pragma: no cover - diagnostic script
        payload["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    payload["elapsed_seconds"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
