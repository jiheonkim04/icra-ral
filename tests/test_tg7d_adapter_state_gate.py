import os
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.tg7d_adapter import state_gate as tg


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "234_tg7d_adapter_state_gate.ps1"


def _write_demo(path: Path, *, offset: float, model_names: list[str]) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for demo_idx in range(8):
            demo = data.create_group(f"demo_{demo_idx}")
            actions = demo.create_dataset("actions", shape=(12, 7), dtype="f4")
            obs = demo.create_group("obs")
            obs.create_dataset("ee_states", shape=(12, 6), dtype="f4")
            for row in range(12):
                actions[row, :] = offset + row * 0.01 + demo_idx * 0.001
            xml_names = "\n".join(f'<body name="{name}_1_main"/>' for name in model_names)
            demo.attrs["model_file"] = f"<mujoco><worldbody>{xml_names}</worldbody></mujoco>"
            demo.attrs["num_samples"] = 12


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "libero"
    _write_demo(
        data_root / "libero_goal" / "put_the_bowl_on_the_plate_demo.hdf5",
        offset=0.1,
        model_names=["bowl", "plate", "stove"],
    )
    _write_demo(
        data_root / "libero_goal" / "turn_on_the_stove_demo.hdf5",
        offset=0.7,
        model_names=["bowl", "plate", "stove"],
    )
    metadata = tmp_path / "libero_para_metadata.csv"
    metadata.write_text(
        "\n".join(
            [
                "high,mid,low,eval,batch_idx,new_instruction,original_instruction,structural_similarity,keyword_similarity",
                "obj,lexical,synonym,0,0,set the dish on the plate,put the bowl on the plate,0.6,0.5",
                "act,lexical,synonym,0,1,place the bowl on the plate,put the bowl on the plate,0.8,0.8",
                "comp,lexical+structural,synonym+coordination,0,2,find the dish and set it on the plate,put the bowl on the plate,0.4,0.5",
                "obj,lexical,synonym,1,0,activate the cooktop,turn on the stove,0.6,0.5",
                "act,lexical,synonym,1,1,switch on the stove,turn on the stove,0.8,0.8",
                "comp,lexical+structural,synonym+coordination,1,2,make the cooktop heat up,turn on the stove,0.4,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data_root, metadata


def test_canonicalization_has_no_metadata_lookup():
    assert tg.canonicalize_instruction("gently activate the cooktop") == "turn on the stove"
    assert tg.canonicalize_instruction("set the dish on the plate") == "put the bowl on the plate"


def test_visible_prior_uses_instruction_and_candidates(tmp_path):
    path = tmp_path / "task_demo.hdf5"
    _write_demo(path, offset=0.1, model_names=["black_bowl", "plate", "robot0_link"])
    candidates = tg.visible_object_candidates(path)
    prior = tg.resolve_target_prior("put the black bowl on the plate", candidates)
    assert prior["uses_bddl_target_labels"] is False
    assert prior["uses_eval_labels"] is False
    assert any("bowl" in item for item in prior["selected_candidates"])


def test_dataset_split_has_no_group_leakage(tmp_path):
    data_root, metadata = _fixture(tmp_path)
    dataset = tg.build_tg7d_dataset(
        data_root=data_root,
        metadata_csv=metadata,
        max_tasks=2,
        max_train_paraphrases_per_task=2,
        max_eval_paraphrases_per_task=2,
        train_demos=2,
        eval_demos=1,
        records_per_demo=2,
    )
    feasibility = dataset["feasibility"]
    assert feasibility["matched_hdf5_task_count"] == 2
    assert feasibility["heldout_paraphrase_records"] > 0
    assert feasibility["counterfactual_records"] > 0
    assert feasibility["leakage"]["group_leakage_detected"] is False
    assert feasibility["target_prior_from_instruction_and_visible_names_only"] is True


def test_final_decisions_are_exact():
    assert tg.FINAL_DECISIONS == {
        "READY_FOR_TG7D_SCALE_UP",
        "KILL_BASELINE_DOMINATED",
        "KILL_CANONICALIZATION_DOMINATED",
        "KILL_LEAKAGE_RISK",
        "NO_TARGET_GROUNDING_EVAL_PATH",
        "TOO_HEAVY_LOCAL",
    }


def test_runner_requires_state_gate(tmp_path):
    powershell = pytest.importorskip("shutil").which("powershell")
    if powershell is None:
        pytest.skip("PowerShell not available")
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ReportPath",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if not key.startswith("ALLOW_TG7D")},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 21
    assert "NO_TARGET_GROUNDING_EVAL_PATH" in (result.stdout + result.stderr)
