from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_official_smolvla_repro_lock.py"
LOCK_PATH = ROOT / "configs" / "official_smolvla_repro_lock.yaml"

spec = importlib.util.spec_from_file_location("check_official_smolvla_repro_lock", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_current_repro_lock_is_valid() -> None:
    data = module.validate_lock(module.load_lock(LOCK_PATH))
    assert data["final_decision"] == "LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED"
    assert data["model"]["revision_status"] == "REVISION_LOCKED"
    assert data["dataset"]["revision_status"] == "REVISION_LOCKED"


def test_rollout_ready_rejects_missing_lora_checkpoints() -> None:
    data = module.load_lock(LOCK_PATH)
    mutated = copy.deepcopy(data)
    mutated["final_decision"] = "ROLLOUT_PROTOCOL_READY"

    with pytest.raises(module.ReproLockError, match="missing or unproven LoRA checkpoints"):
        module.validate_lock(mutated)


def test_static_mix_is_not_adapter_soup_or_weight_merge() -> None:
    data = module.validate_lock(module.load_lock(LOCK_PATH))
    static_mix = data["baseline_names"]["validation_selected_action_space_static_mix"]
    assert static_mix["adapter_soup"] is False
    assert static_mix["adapter_weight_merge"] is False
