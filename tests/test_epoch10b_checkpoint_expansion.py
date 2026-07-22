from pathlib import Path

import pytest

from scripts.run_epoch10b_checkpoint_expansion import (
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    NEW_SEEDS,
    RETAINED_STAGES,
    ExpansionError,
    _partition,
    _validate_output_root,
)


def test_frozen_split_is_whole_seed_disjoint_and_targets_twelve_lineages() -> None:
    assert DEVELOPMENT_SEEDS.isdisjoint(HOLDOUT_SEEDS)
    assert len(DEVELOPMENT_SEEDS | HOLDOUT_SEEDS) == 12
    assert set(NEW_SEEDS).issubset(DEVELOPMENT_SEEDS | HOLDOUT_SEEDS)
    assert RETAINED_STAGES == (30, 100)


def test_frozen_split_has_eight_development_and_four_holdout_lineages() -> None:
    assert len(DEVELOPMENT_SEEDS) == 8
    assert len(HOLDOUT_SEEDS) == 4
    assert all(_partition(seed) == "checkpoint_development_panel" for seed in DEVELOPMENT_SEEDS)
    assert all(_partition(seed) == "checkpoint_holdout_panel" for seed in HOLDOUT_SEEDS)


def test_unregistered_seed_is_rejected() -> None:
    with pytest.raises(ExpansionError, match="outside the frozen whole-seed split"):
        _partition(999999)


def test_expansion_root_cannot_overlap_protected_original(tmp_path: Path) -> None:
    original = tmp_path / "protected" / "rank4"
    original.mkdir(parents=True)
    with pytest.raises(ExpansionError, match="overlaps protected Epoch 10 root"):
        _validate_output_root(original / "nested", original)
