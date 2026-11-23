"""Native face tracker: raw video frame -> 7 avatar parameters, no ML models.

Uses only Dense-Armor primitives:
- FrameFeatures: per-cell Lucas-Kanade flow + intensity moments on facial ROIs
- MotionEnergy: blink energy spike on the eye ROIs
- GlobalMotion: head pose from global camera motion (phase correlation)
- RateBound / OneEuroFilter: per-parameter max step for fluidity
- spectral_flux / spectral_centroid: audio energy for mouth drive

No pre-trained model: the face box is found by HSV skin-color segmentation
(largest skin-colored region), which works even when the head is perfectly
still. The pipeline applies a causal 1 Euro filter (Casiez et al. 2012) to
every parameter for fluid, jitter-free motion.
"""

from __future__ import annotations

import time

import numpy as np
from dense_armor.core.signals.vision.features import FrameFeatures
from dense_armor.core.signals.vision.motion import GlobalMotion, MotionEnergy

from d_avatar.params import RateBound

_HEAD_KEYS = ('head_yaw', 'head_pitch', 'head_roll')
_UNIT_KEYS = ('eye_l', 'eye_r', 'mouth_open')
_ALL_KEYS = _HEAD_KEYS + _UNIT_KEYS + ('blink',)
_BLINK_ENERGY_THRESHOLD = 0.3
_NDIM_RGB = 3
_FACE_MIN_AREA = 500
_SKIN_HSV_LOW = np.array([0, 30, 60], dtype=np.uint8)
_SKIN_HSV_HIGH = np.array([25, 150, 255], dtype=np.uint8)

# HSV range for skin color (OpenCV H in [0,179], S,V in [0,255]).
_SKIN_HSV_LOW = np.array([0, 30, 60], dtype=np.uint8)
_SKIN_HSV_HIGH = np.array([25, 150, 255], dtype=np.uint8)


class OneEuroFilter:
    """1 Euro filter: causal low-pass with adaptive cutoff.

    Keeps the signal smooth when the value is stable (noise reduction) and
    reactive when it moves fast (minimal phase lag). Ideal for avatar
    parameters: no jitter when the user is still, instant response when
    they move.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float = 0.0

    def __call__(self, x: float, t: float | None = None) -> float:
        x = float(x)
        now = t if t is not None else time.monotonic()
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = now
            return x
        dt = max(now - self._t_prev, 1e-3)
        # Differenziato per la velocità
        dx = (x - self._x_prev) / dt
        # Filtro sulla velocità
        dx_hat = (
            self._alpha(self.d_cutoff, dt) * dx
            + (1.0 - self._alpha(self.d_cutoff, dt)) * self._dx_prev
        )
        # Cutoff adattivo: più veloce si muove, più alto il cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        # Filtro sul valore
        x_hat = self._alpha(cutoff, dt) * x + (1.0 - self._alpha(cutoff, dt)) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = now
        return float(x_hat)

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class NativeTracker:
    """Track a face from raw frames and return the 7 D-Avatar parameters.

    No pre-trained model: the face box is found by HSV skin-color
    segmentation (largest skin-colored region), which works even when
    the head is perfectly still. Fixed ROIs inside the box are then
    used for eyes, mouth and head pose. :class:`RateBound` keeps every
    parameter fluid frame to frame. A :class:`OneEuroFilter` is applied
    per parameter for jitter-free motion with fast response.

    Args:
        max_step: max per-frame change of [0, 1] parameters.
        max_step_deg: max per-frame change of head angles, in degrees.
        one_euro_min_cutoff: min cutoff for the 1 Euro filter.
        one_euro_beta: beta for the 1 Euro filter.
    """

    budget_s: float | None = 2e-2
    memory_class: str = 'O(1)'

    def __init__(
        self,
        max_step: float = 0.1,
        max_step_deg: float = 5.0,
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.5,
    ) -> None:
        self.max_step = float(max_step)
        self.max_step_deg = float(max_step_deg)
        self.bound = RateBound(max_step, dict.fromkeys(_HEAD_KEYS, max_step_deg))
        self.features = FrameFeatures(flow_cells=(2, 2), flow_levels=2)
        self.me_l = MotionEnergy(blur=0, downsample=1)
        self.me_r = MotionEnergy(blur=0, downsample=1)
        self.global_motion = GlobalMotion()
        self._face_box: tuple[int, int, int, int] | None = None
        self._n_seen = 0
        # One Euro filter per parametro
        self._filters: dict[str, OneEuroFilter] = {
            k: OneEuroFilter(one_euro_min_cutoff, one_euro_beta) for k in _ALL_KEYS
        }
        # Audio-driven mouth: energy history per blink/mouth
        self._audio_energy: float = 0.0
        self._audio_alpha: float = 0.3

    # -- public API -------------------------------------------------------

    def __call__(self, frame: np.ndarray, t: float | None = None) -> dict[str, float]:
        """Process one frame (H, W) or (H, W, 3), float32 [0,1] or uint8.

        Returns the 7 parameters: head_yaw, head_pitch, head_roll (deg),
        mouth_open, eye_l, eye_r (0, 1), blink (0 or 1).
        """
        img = self._to_gray(frame)
        h, w = img.shape[:2]
        self._n_seen += 1

        # 1. Find / track the face box via HSV skin-color segmentation.
        # Fallback: centred box covering ~60% of the frame.
        box = self._find_face_box(frame)
        if box is not None:
            self._face_box = box
        elif self._face_box is None:
            cx, cy = w // 2, h // 2
            bw, bh = int(w * 0.6), int(h * 0.7)
            self._face_box = (
                cx - bw // 2,
                cy - bh // 2,
                cx + bw // 2,
                cy + bh // 2,
            )

        if self._face_box is None:
            return self._neutral()

        # 2. ROIs inside the face box
        x0, y0, x1, y1 = self._face_box
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        eye_h = max(1, int(bh * 0.14))
        eye_w = max(1, int(bw * 0.10))
        l_eye = (
            x0 + int(bw * 0.20),
            y0 + int(bh * 0.30),
            x0 + int(bw * 0.20) + eye_w,
            y0 + int(bh * 0.30) + eye_h,
        )
        r_eye = (
            x0 + int(bw * 0.70) - eye_w,
            y0 + int(bh * 0.30),
            x0 + int(bw * 0.70),
            y0 + int(bh * 0.30) + eye_h,
        )
        mouth = (
            x0 + int(bw * 0.35),
            y0 + int(bh * 0.70),
            x0 + int(bw * 0.65),
            y0 + int(bh * 0.70) + max(1, int(bh * 0.08)),
        )

        # 3. Signals from primitives
        eye_l = self._eye_openness(img, l_eye)
        eye_r = self._eye_openness(img, r_eye)
        mouth_open = self._mouth_open(img, mouth)
        blink = self._blink(img, l_eye, r_eye)
        head_yaw, head_pitch, head_roll = self._head_pose(img)

        # 4. Audio-driven mouth: if audio energy is high, mouth opens
        mouth_open = max(mouth_open, self._audio_energy)

        # 5. Rate bound + 1 Euro filter for fluidity
        raw = {
            'head_yaw': head_yaw,
            'head_pitch': head_pitch,
            'head_roll': head_roll,
            'mouth_open': mouth_open,
            'eye_l': eye_l,
            'eye_r': eye_r,
            'blink': blink,
        }
        out: dict[str, float] = {}
        for k, v in raw.items():
            out[k] = self.bound.step(k, self._filters[k](v, t))

        # 6. Remember each eye ROI so MotionEnergy has a previous frame
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

    def update_audio_energy(self, energy: float) -> None:
        """Feed audio energy (0..1) to drive mouth opening."""
        self._audio_energy = (
            self._audio_alpha * float(energy) + (1.0 - self._audio_alpha) * self._audio_energy
        )

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
        """Blink = MotionEnergy max spike on either eye ROI."""
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
            out: np.ndarray = np.asarray(
                0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2],
                dtype=np.float32,
            )
            return out  # type: ignore[no-any-return]
        return arr.astype(np.float32)

    def _find_face_box(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Find the face region via HSV skin-color segmentation.

        Returns the bounding box (x0, y0, x1, y1) of the largest
        skin-colored connected region, or None if nothing is found.
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return None

        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if arr.ndim == _NDIM_RGB and arr.shape[2] == _NDIM_RGB:
            hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        else:
            return None

        mask = cv2.inRange(hsv, _SKIN_HSV_LOW, _SKIN_HSV_HIGH)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < _FACE_MIN_AREA:
            return None

        x, y, w, h = cv2.boundingRect(best)
        pad = int(0.15 * min(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(arr.shape[1], x + w + pad)
        y1 = min(arr.shape[0], y + h + pad)
        return (x0, y0, x1, y1)

    def _neutral(self) -> dict[str, float]:
        return dict.fromkeys(_HEAD_KEYS, 0.0) | dict.fromkeys(_UNIT_KEYS, 1.0) | {'blink': 0.0}
