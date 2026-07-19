from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR = REPO_ROOT / "scripts" / "monitor_a2c2_clean_host_smoke.ps1"
LAUNCHER = REPO_ROOT / "scripts" / "run_a2c2_clean_host_smoke_wsl.sh"
FROZEN_RUNNER = REPO_ROOT / "scripts" / "run_a2c2_problem_verification.py"
PANEL_MONITOR = REPO_ROOT / "scripts" / "monitor_a2c2_clean_host_panel.ps1"
PANEL_LAUNCHER = REPO_ROOT / "scripts" / "run_a2c2_clean_host_panel_wsl.sh"


def test_clean_host_monitor_freezes_minimum_memory_envelope() -> None:
    text = MONITOR.read_text(encoding="utf-8")

    assert "[ValidateSet(8, 10, 12, 14)]" in text
    assert "-gt 0.65" in text
    assert "-gt 0.82" in text
    assert '"python", "python3", "wsl"' in text
    assert "Background-heavy process became active" in text
    assert "A2C2_CLEAN_HOST_PAGEFILE_UNSTABLE" in text
    assert "Stale internal clean-host smoke report" in text
    assert "Stale host clean-host smoke report" in text


def test_clean_host_monitor_is_single_instance_and_no_prefetch() -> None:
    text = MONITOR.read_text(encoding="utf-8")

    for required in (
        "policy_processes = 1",
        "policy_instances = 1",
        "libero_environments = 1",
        "eval_batch_size = 1",
        "parallel_tasks = 1",
        "model_residency_count = 1",
        "model_checkpoint = $false",
        "training_data = $false",
        "environment = $false",
        "next_task = $false",
        "video = $false",
        "observation_history = $false",
    ):
        assert required in text

    assert "= false" not in text
    assert "= true" not in text


def test_clean_host_monitor_uses_exact_decision_vocabulary() -> None:
    text = MONITOR.read_text(encoding="utf-8")

    for decision in (
        "A2C2_RESOURCE_SMOKE_PASS",
        "A2C2_RESOURCE_SMOKE_FAIL_CAP_TOO_LOW",
        "A2C2_RESOURCE_SMOKE_FAIL_WINDOWS_CEILING",
        "A2C2_RESOURCE_SMOKE_FAIL_KERNEL_OOM",
        "A2C2_RESOURCE_SMOKE_FAIL_MEMORY_LEAK",
        "A2C2_RESOURCE_SMOKE_FAIL_UNRELATED_IMPLEMENTATION",
    ):
        assert decision in text


def test_clean_host_monitor_includes_post_shutdown_in_peak_accounting() -> None:
    text = MONITOR.read_text(encoding="utf-8")

    assert "$peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$postShutdown.pagefile_current_usage_mib)" in text
    assert "$peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$postShutdown.page_writes_per_sec)" in text
    assert "$pagefileGrowthMiB = $peakPagefileCurrentMiB - [double]$baseline.pagefile_current_usage_mib" in text


def test_clean_host_launcher_uses_accepted_environment_and_variable_outputs() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python" in text
    assert 'OUTPUT_JSON="$3"' in text
    assert 'OUTPUT_MD="$4"' in text
    assert "scripts/run_a2c2_resource_smoke.py" in text


def test_frozen_scientific_runner_remains_resource_mode_free() -> None:
    text = FROZEN_RUNNER.read_text(encoding="utf-8")

    assert '"resource-smoke"' not in text


def test_clean_host_panel_is_sequential_missing_key_resumable() -> None:
    monitor = PANEL_MONITOR.read_text(encoding="utf-8")
    launcher = PANEL_LAUNCHER.read_text(encoding="utf-8")

    for required in (
        'concurrency_mode = "SEQUENTIAL"',
        'prefetch_mode = "DISABLED"',
        "full_backbone_instances = 1",
        "simultaneous_full_backbones = 1",
        "eval_batch_size = 1",
        "parallel_tasks = 1",
        "live_environments = 1",
        "-gt 0.82",
        "--terminate Ubuntu-22.04",
    ):
        assert required in monitor
    assert "base_rollout_partial.json" in launcher
    assert "prior_rollout_partial.json" in launcher
    assert "scripts/run_a2c2_problem_verification.py" in launcher
    assert "--limit-episodes" not in launcher


def test_clean_host_panel_preserves_frozen_runner_hash_contract() -> None:
    monitor = PANEL_MONITOR.read_text(encoding="utf-8")

    assert "memory=12GB" in monitor
    assert "A7CC4F707936DBBCE335F298BF0E968804F956E4DBE71A17ECEF5775CC708445" in monitor
    assert "A2C2_BASE_CLOSED_LOOP_ACCEPTED" in monitor
    assert "A2C2_PRIOR_CLOSED_LOOP_ACCEPTED" in monitor
