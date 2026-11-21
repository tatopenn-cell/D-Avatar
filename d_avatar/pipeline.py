"""
Pipeline: chains FaceShield to landmarks_to_params to RateBound to VmcSender
inside RealtimePipeline at 30 fps.

Source: dense_armor.roles.RealtimePipeline, dense_armor.core.signals.vision.landmarks.LandmarkStream
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dense_armor.roles import RealtimePipeline, Transformer
from dense_armor.core.signals.vision.landmarks import LandmarkStream

from d_avatar.shield import FaceShield
from d_avatar.params import landmarks_to_params, RateBound, BlinkCalibrator, ema
from d_avatar.vmc import VmcSender


def build_pipeline(
    period_s: float = 1.0 / 30.0,
    max_step: float = 0.1,
    shield_window: int = 7,
    shield_n_sigmas: float = 3.0,
    shield_max_gap: int = 5,
    vmc_host: str = "127.0.0.1",
    vmc_port: int = 39539,
) -> RealtimePipeline:
    """Build the real-time pipeline at 30 fps.

    Args:
        period_s: frame period (default 1/30 s).
        max_step: max parameter change per frame for RateBound.
        shield_window: FaceShield Hampel window.
        shield_n_sigmas: FaceShield Hampel threshold.
        shield_max_gap: FaceShield max predict-only frames.
        vmc_host: VMC destination host.
        vmc_port: VMC destination port.

    Returns:
        RealtimePipeline with require_budget=True.
    """
    # Components
    shield = FaceShield(
        window=shield_window,
        n_sigmas=shield_n_sigmas,
        max_gap=shield_max_gap,
    )

    # Create step instances with passed parameters using closures
    def make_shield_step(shield):
        class ShieldStep(Transformer):
            budget_s = shield.budget_s
            memory_class = shield.memory_class

            def transform_one(self, x: dict, t: float | None = None) -> dict:
                return shield.transform_one(x)

            def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> "ShieldStep":
                return self
        return ShieldStep()

    def make_params_step(max_step: float):
        class ParamsStep(Transformer):
            budget_s = 1e-3
            memory_class = "O(1)"

            def __init__(self, max_step: float = max_step) -> None:
                self._blink_cal = BlinkCalibrator()
                self._rate_bound = RateBound(max_step=max_step)
                self._prev: dict[str, float] = {}

            def transform_one(self, x: dict, t: float | None = None) -> dict:
                cleaned = x["points"]
                params = landmarks_to_params(cleaned)

                # Blink calibration (Larey §3.5.1)
                ear_l = (
                    np.linalg.norm(cleaned[37] - cleaned[41])
                    + np.linalg.norm(cleaned[38] - cleaned[40])
                ) / (2.0 * np.linalg.norm(cleaned[36] - cleaned[39]) + 1e-6)
                ear_r = (
                    np.linalg.norm(cleaned[43] - cleaned[47])
                    + np.linalg.norm(cleaned[44] - cleaned[46])
                ) / (2.0 * np.linalg.norm(cleaned[42] - cleaned[45]) + 1e-6)
                ear = (ear_l + ear_r) / 2.0
                self._blink_cal.update(ear)
                params["blink"] = 1.0 if self._blink_cal.is_blink(ear) else 0.0

                # EMA smoothing (Larey §3.2)
                for k, v in params.items():
                    if k in ("eye_l", "eye_r", "mouth_open", "brow_l", "brow_r", "gaze_x", "gaze_y"):
                        prev = self._prev.get(k, v)
                        params[k] = ema(prev, v, 0.3)
                        self._prev[k] = params[k]

                # Rate bound (Kartynnik §5)
                for k, v in params.items():
                    if k in (
                        "eye_l", "eye_r", "mouth_open", "brow_l", "brow_r",
                        "gaze_x", "gaze_y", "head_yaw", "head_pitch", "head_roll", "blink"
                    ):
                        params[k] = self._rate_bound.step(k, v)

                return params

            def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> "ParamsStep":
                return self
        return ParamsStep()

    def make_vmc_step(host: str, port: int):
        class VmcStep(Transformer):
            budget_s = 1e-3
            memory_class = "O(1)"

            def __init__(self) -> None:
                self._vmc = VmcSender(host=host, port=port)

            def transform_one(self, x: dict, t: float | None = None) -> dict:
                self._vmc.send(x)
                return x

            def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> "VmcStep":
                return self
        return VmcStep()

    # Create instances with passed parameters
    shield_step = make_shield_step(shield)
    params_step = make_params_step(max_step)
    vmc_step = make_vmc_step(vmc_host, vmc_port)

    pipeline = RealtimePipeline(
        steps=[shield_step, params_step, vmc_step],
        period_s=period_s,
        require_budget=True,
    )
    return pipeline


def run(stream: LandmarkStream) -> tuple[dict[str, float], float]:
    """Run pipeline on a LandmarkStream.

    Returns:
        (max_jumps_per_param, p99_latency_s)
    """
    pipeline = build_pipeline()
    max_jumps: dict[str, float] = {}
    latencies: list[float] = []
    prev_params: dict[str, float] | None = None

    import time

    for frame in stream:
        t0 = time.perf_counter()
        frame_dict = {
            "points": frame["points"],
            "confidence": frame["confidence"],
            "t": frame["t"],
        }
        # Pipeline steps expect dict with points/confidence/t
        shield_out = pipeline.steps[0].learn_one(frame_dict)
        shield_out = pipeline.steps[0].transform_one(frame_dict)
        params_out = pipeline.steps[1].learn_one(shield_out)
        params_out = pipeline.steps[1].transform_one(shield_out)
        vmc_out = pipeline.steps[2].learn_one(params_out)
        vmc_out = pipeline.steps[2].transform_one(params_out)

        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        # Track max jumps
        if prev_params is not None:
            for k, v in vmc_out.items():
                if k in prev_params:
                    jump = abs(v - prev_params[k])
                    if k not in max_jumps or jump > max_jumps[k]:
                        max_jumps[k] = jump
        prev_params = vmc_out.copy()

    # Compute p99
    if latencies:
        latencies.sort()
        p99_idx = int(0.99 * len(latencies))
        p99 = latencies[p99_idx]
    else:
        p99 = 0.0

    return max_jumps, p99
