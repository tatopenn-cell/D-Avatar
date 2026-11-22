"""
Pipeline: FaceShield, then landmarks_to_params with smoothing and rate bounds, then an
optional VMC sender.

``AvatarPipeline.step`` chains the steps frame by frame. ``AvatarPipeline.realtime`` is a
dense_armor ``RealtimePipeline`` built over the same steps: it refuses to build when a step
has no ``budget_s`` or when the budgets add up to more than the frame period.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN  # type: ignore[import-untyped]
from dense_armor.core.signals.vision.landmarks import LandmarkStream  # type: ignore[import-untyped]
from dense_armor.roles import RealtimePipeline, Transformer  # type: ignore[import-untyped]

from d_avatar.gaze import GazeCalibrator, iris_displacement
from d_avatar.params import (
    BlinkCalibrator,
    RateBound,
    ema,
    eye_aspect_ratios,
    landmarks_to_params,
)
from d_avatar.shield import FaceShield
from d_avatar.vmc import VmcSender

_UNIT_KEYS = ('eye_l', 'eye_r', 'mouth_open', 'brow_l', 'brow_r')
_HEAD_KEYS = ('head_yaw', 'head_pitch', 'head_roll')
_GAZE_KEYS = ('gaze_x', 'gaze_y')
_EMA_ALPHA = 0.3


class ParamsStep(Transformer):
    """Landmarks to parameters: calibrated eye opening, EMA on [0, 1] values,
    rate bounds, gaze from the iris, and drift-triggered recalibration.

    When a frame carries ``iris``, ``gaze_x`` and ``gaze_y`` are computed with
    :class:`d_avatar.gaze.GazeCalibrator`, smoothed like the other [0, 1]
    values and limited with the same ``max_step``. Frames without ``iris``
    return the parameters unchanged (no gaze keys).

    Drift detection: two ADWIN detectors monitor the eye aspect ratio (EAR)
    and the iris displacement magnitude. When either signals a drift
    (baseline shift, e.g. new person or changed distance), both
    ``BlinkCalibrator`` and ``GazeCalibrator`` are reset.

    Args:
        max_step: maximum change per frame of the [0, 1] parameters.
        max_step_deg: maximum change per frame of the head angles, in degrees.
        drift_delta: ADWIN confidence parameter (default 0.002).
        drift_max_window: ADWIN maximum window size (default 100).
    """

    budget_s: float | None = 6e-3
    memory_class: str = 'O(1)'

    def __init__(
        self,
        max_step: float = 0.1,
        max_step_deg: float = 5.0,
        drift_delta: float = 0.002,
        drift_max_window: int = 100,
    ) -> None:
        self.max_step = float(max_step)
        self.max_step_deg = float(max_step_deg)
        self.bound = RateBound(self.max_step, dict.fromkeys(_HEAD_KEYS, self.max_step_deg))
        self.calibrator = BlinkCalibrator()
        self.gaze_calibrator = GazeCalibrator()
        self._ema: dict[str, float] = {}
        self._gaze_ema: dict[str, float] = {}
        # ADWIN drift detectors for EAR and iris displacement magnitude
        self._ear_drift = ADWIN(delta=drift_delta, max_window=drift_max_window)
        self._iris_drift = ADWIN(delta=drift_delta, max_window=drift_max_window)
        self._drift_detected = False

    def _reset_calibrators(self) -> None:
        """Reset both calibrators when drift is detected."""
        self.calibrator = BlinkCalibrator()
        self.gaze_calibrator = GazeCalibrator()
        self._ema.clear()
        self._gaze_ema.clear()
        self._drift_detected = True

    def transform_one(self, x: dict, t: float | None = None) -> dict[str, Any]:
        pts = np.asarray(x['points'], dtype=float)
        ear_l, ear_r = eye_aspect_ratios(pts)
        ear = (ear_l + ear_r) / 2.0

        # Update EAR drift detector
        self._ear_drift.update(ear)
        if self._ear_drift.drift_detected:
            self._reset_calibrators()

        self.calibrator.update(ear)
        params: dict[str, float] = landmarks_to_params(pts, ear_open=self.calibrator.open_ear())
        for k in _UNIT_KEYS:
            smooth = ema(self._ema.get(k, params[k]), params[k], _EMA_ALPHA)
            self._ema[k] = smooth
            params[k] = self.bound.step(k, smooth)
        for k in _HEAD_KEYS:
            params[k] = self.bound.step(k, params[k])
        params['blink'] = 1.0 if self.calibrator.is_blink(ear) else 0.0

        if 'iris' in x and x['iris'] is not None:
            iris = np.asarray(x['iris'], dtype=float)
            dxs: list[float] = []
            dys: list[float] = []
            # (inner corner, outer corner, lid up, lid down) per iBUG eye.
            # iris[0] is the left eye (iBUG 36-41), iris[1] the right eye (42-47).
            for eye, (ci, co, lu, ld) in enumerate(((39, 36, 37, 40), (42, 45, 43, 46))):
                dx, dy = iris_displacement(iris[eye], pts[ci], pts[co], pts[lu], pts[ld])
                dxs.append(dx)
                dys.append(dy)
            dx = float(np.mean(dxs))
            dy = float(np.mean(dys))
            iris_mag = float(np.hypot(dx, dy))

            # Update iris displacement drift detector
            self._iris_drift.update(iris_mag)
            if self._iris_drift.drift_detected:
                self._reset_calibrators()

            self.gaze_calibrator.update(dx, dy)
            gx, gy = self.gaze_calibrator.gaze(dx, dy)
            for name, g in (('gaze_x', gx), ('gaze_y', gy)):
                smooth = ema(self._gaze_ema.get(name, g), g, _EMA_ALPHA)
                self._gaze_ema[name] = smooth
                params[name] = self.bound.step(name, smooth)
        else:
            self._drift_detected = False
        return params

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> ParamsStep:
        self.transform_one(x, t)
        return self


class VmcStep(Transformer):
    """Send the parameters of a frame over VMC and pass them on."""

    budget_s: float | None = 4e-3
    memory_class: str = 'O(1)'

    def __init__(self, host: str = '127.0.0.1', port: int = 39539) -> None:
        self.host = host
        self.port = int(port)
        self._vmc = VmcSender(host=host, port=port)

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        self._vmc.send(x)
        return x

    def close(self) -> None:
        """Close the UDP socket of the sender."""
        self._vmc.close()

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> VmcStep:
        self.transform_one(x, t)
        return self


class PassThrough(Transformer):
    """Shield replacement that returns the landmarks as they are (for comparisons)."""

    budget_s: float | None = 1e-3
    memory_class: str = 'O(1)'

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        pts = np.asarray(x['points'], dtype=float)
        return {'points': pts, 'cleaned': np.zeros(len(pts), dtype=int)}

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> PassThrough:
        return self


class AvatarPipeline:
    """Shield, parameters and optional VMC output, chained frame by frame.

    Args:
        period_s: frame period in seconds (default 0.05). The declared budgets are ten times
            the measured p99 of each step and add up to 41 ms without VMC and 45 ms with it.
        max_step: max change per frame of the [0, 1] parameters.
        max_step_deg: max change per frame of the head angles, in degrees.
        shield_window: FaceShield Hampel window.
        shield_n_sigmas: FaceShield Hampel threshold.
        shield_max_gap: FaceShield max predict-only frames.
        drift_delta: ADWIN confidence parameter (default 0.002).
        drift_max_window: ADWIN maximum window size (default 100).
        vmc_host: VMC destination host; None sends nothing and needs no python-osc.
        vmc_port: VMC destination port.
        use_shield: False replaces the shield with a pass-through.

    Raises:
        ValueError: if a step declares no budget_s or the budgets exceed period_s.
    """

    def __init__(  # noqa: PLR0913
        self,
        period_s: float = 0.05,
        max_step: float = 0.1,
        max_step_deg: float = 5.0,
        *,
        shield_window: int = 7,
        shield_n_sigmas: float = 3.0,
        shield_max_gap: int = 5,
        drift_delta: float = 0.002,
        drift_max_window: int = 100,
        vmc_host: str | None = None,
        vmc_port: int = 39539,
        use_shield: bool = True,
    ) -> None:
        self.shield: Transformer = (
            FaceShield(window=shield_window, n_sigmas=shield_n_sigmas, max_gap=shield_max_gap)
            if use_shield
            else PassThrough()
        )
        self.params_step = ParamsStep(max_step, max_step_deg, drift_delta, drift_max_window)
        self.vmc = VmcStep(vmc_host, vmc_port) if vmc_host is not None else None
        steps: list[Any] = [self.shield, self.params_step]
        if self.vmc is not None:
            steps.append(self.vmc)
        self.realtime = RealtimePipeline(steps=steps, period_s=period_s, require_budget=True)

    @property
    def total_budget_s(self) -> float | None:
        budget: float | None = self.realtime.total_budget_s
        return budget

    def step(self, frame: dict) -> dict:
        """Process one frame ``{"points", "confidence", "t"}`` and return its parameters.

        The iris (when present in the frame) is passed through the shield to the
        parameter step so gaze can be computed.
        """
        cleaned = self.shield.transform_one(frame)
        x = dict(cleaned)
        if 'iris' in frame:
            x['iris'] = frame['iris']
        params = self.params_step.transform_one(x, frame.get('t'))
        if self.vmc is not None:
            self.vmc.transform_one(params)
        return params


def build_pipeline(**kwargs: Any) -> AvatarPipeline:
    """Build an AvatarPipeline; the keyword arguments are those of AvatarPipeline."""
    return AvatarPipeline(**kwargs)


def run(
    stream: LandmarkStream, pipeline: AvatarPipeline | None = None
) -> tuple[dict[str, float], float]:
    """Run the pipeline on a LandmarkStream.

    Returns:
        (largest frame-to-frame jump of each parameter, p99 latency per frame in seconds)
    """
    ap = pipeline if pipeline is not None else AvatarPipeline()
    jumps: dict[str, float] = {}
    lat: list[float] = []
    prev: dict[str, float] | None = None
    for frame in stream:
        t0 = time.perf_counter()
        params = ap.step(frame)
        lat.append(time.perf_counter() - t0)
        if prev is not None:
            for k, v in params.items():
                jumps[k] = max(jumps.get(k, 0.0), abs(v - prev[k]))
        prev = params
    return jumps, float(np.percentile(lat, 99)) if lat else 0.0
