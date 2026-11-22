"""Tests for the pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
from dense_armor.roles import RealtimePipeline, Transformer

from d_avatar import AvatarPipeline, build_pipeline, run
from d_avatar.synthetic import synthetic_face_68

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import synthetic_occlusion as ex


class _NoBudget(Transformer):
    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return x

    def learn_one(self, x: dict, y=None, t: float | None = None) -> '_NoBudget':
        return self


def test_realtime_pipeline_requires_budget():
    with pytest.raises(ValueError, match='budget_s'):
        RealtimePipeline(steps=[_NoBudget()], period_s=1 / 30, require_budget=True)


def test_budgets_over_period_are_refused():
    with pytest.raises(ValueError, match='exceeds period'):
        AvatarPipeline(period_s=1e-6)


def test_build_pipeline_passes_parameters():
    ap = build_pipeline(max_step=0.5, max_step_deg=7.0, shield_max_gap=9)
    assert ap.params_step.bound.max_step == 0.5
    assert ap.params_step.bound.per_name['head_yaw'] == 7.0
    assert ap.shield.max_gap == 9
    assert ap.vmc is None


def test_head_ramp_is_not_frozen():
    ap = AvatarPipeline(max_step_deg=5.0, use_shield=False)
    pts = synthetic_face_68()
    ap.step({'points': pts, 'confidence': np.ones(68), 't': 0.0})
    c = pts.mean(axis=0)
    a = np.radians(30.0)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    turned = (pts - c) @ r.T + c
    roll = [
        ap.step({'points': turned, 'confidence': np.ones(68), 't': (i + 1) / 30})['head_roll']
        for i in range(8)
    ]
    assert abs(roll[5] - 30.0) < 1e-6
    assert abs(roll[0] - 5.0) < 1e-6


def test_shield_reduces_error_under_occlusion():
    stream = ex.build_synthetic_stream()
    clean = ex.AvatarPipeline(use_shield=False)
    truth = []
    for frame in ex.LandmarkStream(
        np.tile(ex.synthetic_face_68(), (ex.N_FRAMES, 1, 1)),
        np.ones((ex.N_FRAMES, 68)),
        np.arange(ex.N_FRAMES) / 30.0,
        fps=30.0,
    ):
        truth.append(clean.step(frame)['eye_l'])
    with_s = [p['eye_l'] for p in ex.trace(stream, True)]
    without = [p['eye_l'] for p in ex.trace(stream, False)]
    err_with = max(abs(a - b) for a, b in zip(with_s, truth, strict=True))
    err_without = max(abs(a - b) for a, b in zip(without, truth, strict=True))
    assert err_with < 0.05
    assert err_without > 0.5


def test_run_without_python_osc_and_p99_within_budget():
    jumps, p99 = run(ex.build_synthetic_stream())
    assert all(j <= 5.0 + 1e-6 for j in jumps.values())
    assert all(
        j <= 0.1 + 1e-6 for k, j in jumps.items() if not k.startswith('head') and k != 'blink'
    )
    assert p99 <= AvatarPipeline().total_budget_s
