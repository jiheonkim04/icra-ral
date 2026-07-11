import numpy as np

from tca_map.smolvla.censored_credit_vla import (
    CensorRecord,
    fit_censor_credit,
    simple_temporal_ema,
    temporal_feature_dict,
    temporal_hold_blend,
    vla_corrector_jump_proxy,
)


def _action(value):
    return np.asarray([[value, 0.0, 0.0, value, 0.0, 0.0, -1.0]])


def test_fit_censor_credit_separates_delta_features():
    prev = _action(0.1)
    smooth = temporal_feature_dict(_action(0.12), previous_action=prev, step_fraction=0.2)
    jump = temporal_feature_dict(_action(0.9), previous_action=prev, step_fraction=0.2)
    model = fit_censor_credit([CensorRecord(smooth, 1.0), CensorRecord(jump, -1.0)])

    assert model.score(smooth) > model.score(jump)


def test_temporal_hold_blend_uses_previous_action_only_when_margin_is_low():
    current = _action(1.0)
    previous = _action(0.0)

    safe = temporal_hold_blend(current, previous_action=previous, margin=1.0)
    risky = temporal_hold_blend(current, previous_action=previous, margin=-1.0, hold_strength=0.75)

    assert np.allclose(safe, current)
    assert np.linalg.norm(risky) < np.linalg.norm(current)


def test_temporal_baselines_are_distinct():
    current = _action(1.0)
    previous = _action(0.0)

    ema = simple_temporal_ema(current, previous_action=previous, ema_strength=0.5)
    jump = vla_corrector_jump_proxy(current, previous_action=previous, jump_threshold=0.1)

    assert np.allclose(ema[:, :6], 0.5 * current[:, :6])
    assert np.allclose(jump, previous)
