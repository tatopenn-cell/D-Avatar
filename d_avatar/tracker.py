"""Native face tracker: raw video frame -> 7 avatar parameters, no ML models.

Uses only Dense-Armor primitives:
- MovingObjectTracker: find the face box in the first seconds (largest moving object)
- FrameFeatures: per-cell Lucas-Kanade flow + intensity moments on facial ROIs
- MotionEnergy: blink energy spike on the eye ROIs
- GlobalMotion / Registration / Stabilizer: head pose from global camera motion
- RateBound: per-parameter max step for fluidity
"""

from __future__ import annotations

import numpy as np
from dense_armor.core.signals.vision.features import FrameFeatures
from dense_armor.core.signals.vision.motion import GlobalMotion, MotionEnergy
from dense_armor.core.signals.vision.registration import Registration, Stabilizer
from dense_armor.core.signals.vision.tracking import MovingObjectTracker

from d_avatar.params import RateBound

_HEAD_KEYS = ('head_yaw', 'head_pitch', 'head_roll')
_UNIT_KEYS = ('eye_l', 'eye_r', 'mouth_open')
_ALL_KEYS = _HEAD_KEYS + _UNIT_KEYS + ('blink',)
_BLINK_ENERGY_THRESHOLD = 0.3
_NDIM_RGB = 3


class NativeTracker:
    """Track a face from raw frames and return the 7 D-Avatar parameters.

    No pre-trained model: the face box is found by :class:`MovingObjectTracker`
    on the first frames (largest moving object), then fixed ROIs inside the box
    are used for eyes, mouth and head pose. :class:`RateBound` keeps every
    parameter fluid frame to frame.

    Args:
        max_step: max per-frame change of [0, 1] parameters.
        max_step_deg: max per-frame change of head angles, in degrees.
        calibrate_frames: frames to wait for face lock before trusting the box.
    """

    budget_s: float | None = 2e-2
    memory_class: str = 'O(1)'

    def __init__(
        self,
        max_step: float = 0.1,
        max_step_deg: float = 5.0,
        calibrate_frames: int = 90,
    ) -> None:
        self.max_step = float(max_step)
        self.max_step_deg = float(max_step_deg)
        self.calibrate_frames = int(calibrate_frames)
        self.bound = RateBound(max_step, dict.fromkeys(_HEAD_KEYS, max_step_deg))
        self.mot = MovingObjectTracker()
        self.features = FrameFeatures(flow_cells=(2, 2), flow_levels=2)
        self.me_l = MotionEnergy(blur=0, downsample=1)
        self.me_r = MotionEnergy(blur=0, downsample=1)
        self.global_motion = GlobalMotion()
        self.registration = Registration()
        self.stabilizer = Stabilizer()
        self._face_box: tuple[int, int, int, int] | None = None
        self._n_seen = 0

    # -- public API -------------------------------------------------------

    def __call__(self, frame: np.ndarray) -> dict[str, float]:
        """Process one frame (H, W) or (H, W, 3), float32 [0,1] or uint8.

        Returns the 7 parameters: head_yaw, head_pitch, head_roll (deg),
        mouth_open, eye_l, eye_r (0, 1), blink (0 or 1).
        """
        img = self._to_gray(frame)
        self._n_seen += 1

        # 1. Find / track the face box (largest moving object)
        tracks = self.mot.transform_one(img)
        box = self._pick_face_box(tracks)
        if box is not None:
            self._face_box = box
        if self._face_box is None or self._n_seen < self.calibrate_frames:
            return self._neutral()

        # 2. ROIs inside the face box
        x0, y0, x1, y1 = self._face_box
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        eye_h = max(1, int(h * 0.14))
        eye_w = max(1, int(w * 0.10))
        l_eye = (
            x0 + int(w * 0.20),
            y0 + int(h * 0.30),
            x0 + int(w * 0.20) + eye_w,
            y0 + int(h * 0.30) + eye_h,
        )
        r_eye = (
            x0 + int(w * 0.70) - eye_w,
            y0 + int(h * 0.30),
            x0 + int(w * 0.70),
            y0 + int(h * 0.30) + eye_h,
        )
        mouth = (
            x0 + int(w * 0.35),
            y0 + int(h * 0.70),
            x0 + int(w * 0.65),
            y0 + int(h * 0.70) + max(1, int(h * 0.08)),
        )

        # 3. Signals from primitives
        eye_l = self._eye_openness(img, l_eye)
        eye_r = self._eye_openness(img, r_eye)
        mouth_open = self._mouth_open(img, mouth)
        blink = self._blink(img, l_eye, r_eye)
        head_yaw, head_pitch, head_roll = self._head_pose(img)

        # 4. Rate bound for fluidity
        raw = {
            'head_yaw': head_yaw,
            'head_pitch': head_pitch,
            'head_roll': head_roll,
            'mouth_open': mouth_open,
            'eye_l': eye_l,
            'eye_r': eye_r,
            'blink': blink,
        }
        out = {k: self.bound.step(k, v) for k, v in raw.items()}

        # 5. Remember each eye ROI so MotionEnergy has a previous frame next call
        x0, y0, x1, y1 = self._face_box
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        eye_h = max(1, int(h * 0.14))
        eye_w = max(1, int(w * 0.10))
        l_eye = (
            x0 + int(w * 0.20),
            y0 + int(h * 0.30),
            x0 + int(w * 0.20) + eye_w,
            y0 + int(h * 0.30) + eye_h,
        )
        r_eye = (
            x0 + int(w * 0.70) - eye_w,
            y0 + int(h * 0.30),
            x0 + int(w * 0.70),
            y0 + int(h * 0.30) + eye_h,
        )
        self.me_l.learn_one(
            img[max(0, l_eye[1]) : max(0, l_eye[3]), max(0, l_eye[0]) : max(0, l_eye[2])]
        )
        self.me_r.learn_one(
            img[max(0, r_eye[1]) : max(0, r_eye[3]), max(0, r_eye[0]) : max(0, r_eye[2])]
        )
        return out

    def set_face_box(self, box: tuple[int, int, int, int]) -> None:
        """Fix the face ROI manually (used for calibration / tests)."""
        self._face_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

    # -- primitives per parameter ----------------------------------------

    def _eye_openness(self, img: np.ndarray, box: tuple[int, int, int, int]) -> float:
        """Eye openness = mean intensity of the eye ROI (open=bright, closed=dark)."""
        x0, y0, x1, y1 = box
        roi = img[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        if roi.size == 0:
            return 1.0
        feats = self.features.transform_one(roi)
        return float(np.clip(feats.get('mean', 0.5), 0.0, 1.0))

    def _mouth_open(self, img: np.ndarray, box: tuple[int, int, int, int]) -> float:
        """Mouth open = inverse mean intensity of the mouth ROI (open=dark inside)."""
        x0, y0, x1, y1 = box
        roi = img[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        if roi.size == 0:
            return 0.0
        feats = self.features.transform_one(roi)
        return float(np.clip(1.0 - feats.get('mean', 0.5), 0.0, 1.0))

    def _blink(
        self,
        img: np.ndarray,
        l_eye: tuple[int, int, int, int],
        r_eye: tuple[int, int, int, int],
    ) -> float:
        """Blink = MotionEnergy max spike on either eye ROI.

        Each eye has its own MotionEnergy instance; ``transform_one`` compares
        against the previous ROI (memorised by ``learn_one`` at the end of
        ``__call__``), so the first call always returns 0.
        """
        x0, y0, x1, y1 = l_eye
        roi_l = img[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        me_l = self.me_l.transform_one(roi_l)
        x0, y0, x1, y1 = r_eye
        roi_r = img[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        me_r = self.me_r.transform_one(roi_r)
        energy = float(me_l.get('motion_max', 0.0)) + float(me_r.get('motion_max', 0.0))
        return 1.0 if energy > _BLINK_ENERGY_THRESHOLD else 0.0

    def _head_pose(self, img: np.ndarray) -> tuple[float, float, float]:
        """Head pose from GlobalMotion (phase correlation shift -> yaw/pitch)."""
        gm = self.global_motion.transform_one(img)
        yaw = float(np.clip(gm.get('dx', 0.0) * 10.0, -90.0, 90.0))
        pitch = float(np.clip(gm.get('dy', 0.0) * 10.0, -90.0, 90.0))
        return yaw, pitch, 0.0

    # -- helpers ----------------------------------------------------------

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == _NDIM_RGB:
            out = np.asarray(
                0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2],
                dtype=np.float32,
            )
            return out  # type: ignore[no-any-return]
        return arr.astype(np.float32)  # type: ignore[no-any-return]

    def _pick_face_box(self, tracks: dict) -> tuple[int, int, int, int] | None:
        """Largest track box as the face candidate."""
        best: tuple[int, int, int, int] | None = None
        best_area = 0
        for tr in tracks.get('tracks', []):
            box = tr.get('box')
            if box is None:
                continue
            x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            area = (x1 - x0) * (y1 - y0)
            if area > best_area:
                best_area = area
                best = (x0, y0, x1, y1)
        return best

    def _neutral(self) -> dict[str, float]:
        return dict.fromkeys(_HEAD_KEYS, 0.0) | dict.fromkeys(_UNIT_KEYS, 1.0) | {'blink': 0.0}
