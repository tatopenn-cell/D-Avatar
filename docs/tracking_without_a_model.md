# Tracking Without a Model: D-Avatar Parameters from Dense-Armor Primitives Only

This document maps every D-Avatar output parameter to **Dense-Armor primitives that operate on raw video frames without any pre-trained model**. No MediaPipe, no 68-point iBUG landmarks, no neural networks. The pipeline uses only classical computer vision primitives from Dense-Armor: frame differences, optical flow, background subtraction, motion energy, and patch memory.

**Default path = zero ML imports.** `mediapipe`, `torch`, `tensorflow`, `onnx` never imported. Tests run on synthetic frames from `CameraStream` or `ArrayStream`.

---

## 1. Parameter Map — Model-Free Primitives Only

| D-Avatar Parameter | Dense-Armor Primitive(s) | Image Region / Signal Source | Synthetic Test Limit (p99) | Paper / Equation Reference |
|---|---|---|---|---|
| `eye_l` (left eye openness) | `MovingObjectTracker` → `FrameFeatures._flow` (Lucas-Kanade per cell) on eye ROI | Left eye ROI: tracked box from `MovingObjectTracker` + `FrameFeatures` cells covering eye region | TBD (not yet implemented) | Lucas–Kanade 1981 (IJCAI); Ziani 2025 Eqs. 4–5 (per-cell LS); Dense-Armor `FrameFeatures._flow_cells_one_level` |
| `eye_r` (right eye openness) | `MovingObjectTracker` → `FrameFeatures._flow` on eye ROI | Right eye ROI: tracked box + `FrameFeatures` cells | TBD | Same as `eye_l` |
| `mouth_open` | `MovingObjectTracker` → `FrameFeatures._flow` on mouth ROI | Mouth ROI: tracked box + `FrameFeatures` cells | TBD | Same primitives |
| `brow_l` (left brow raise) | `MovingObjectTracker` → `FrameFeatures._flow` on brow ROI | Left brow ROI: tracked box + cells | TBD | Same primitives |
| `brow_r` (right brow raise) | `MovingObjectTracker` → `FrameFeatures._flow` on brow ROI | Right brow ROI: tracked box + cells | TBD | Same primitives |
| `head_yaw` | `GlobalMotion` (phase correlation) + `Registration` | Whole frame: global rotation estimate from phase correlation | TBD | Horn–Schunck 1981 (global flow); Dense-Armor `GlobalMotion._phase_shift` (FFT phase correlation) |
| `head_pitch` | `GlobalMotion` + `Registration` | Whole frame: global rotation | TBD | Same as `head_yaw` |
| `head_roll` | `GlobalMotion` + `Registration` | Whole frame: global rotation | TBD | Same as `head_yaw` |
| `blink` | `MotionEnergy` on eye ROI + `ADWIN` | Eye ROI: frame-difference energy spike = blink | TBD | MotionEnergy (filtered |diff|); ADWIN (Bifet & Gavalda 2007) |
| `gaze_x`, `gaze_y` | `DenseFlow` / `FrameFeatures._flow` on iris patch + `PatchMemory` | Iris patch: dense flow vector in eye region, matched against `PatchMemory` of neutral/extreme gazes | TBD | Dense optical flow (Horn–Schunck / Lucas–Kanade dense); PatchMemory (Dense-Armor `patches.py`) |

**Notes:**
- **No 68 iBUG points.** Facial regions (eyes, mouth, brows) are **tracked boxes** from `MovingObjectTracker`, not landmark coordinates.
- **No iris landmarks.** Gaze uses **dense flow in an iris patch** + `PatchMemory` comparison, not `iris_displacement` from MediaPipe 469–478.
- **Head pose** comes from **global camera motion** (`GlobalMotion` phase correlation), not from nose-bridge/eye-center geometry.
- **Blink** is a **motion energy spike** in the eye ROI, not an eye aspect ratio from landmarks.
- All primitives are **causal, online, fixed-memory** (window-bounded).

---

## 2. Dense-Armor Primitives — `usable select` / `card` Evidence

### 2.1 CameraStream — Frame Source
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/streams.py
```
```
code/CameraStream: kind=class, line=257, n_args=0, calls=['Frame', 'ImportError', 'RuntimeError', 'ValueError', 'VideoCapture', '_resize_nearest']
code/__iter__: kind=function, line=302, n_args=1, calls=['Frame', 'RuntimeError', 'VideoCapture', '_resize_nearest', '_to_float01', 'cap.isOpened']
```
Yields `Frame(array: float32 [0,1] RGB, t: float)`.

---

### 2.2 MovingObjectTracker — Track Face/Object Boxes Without Detector
```bash
python Dense-Armor/dense_armor/usable/launch.py select vision_features --budget-ms 5 --top 5
```
(returns 91 deep-learning classes; `MovingObjectTracker` is domain-specific in `vision/tracking.py`)

```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/tracking.py
```
```
code/MovingObjectTracker: kind=class, line=152, n_args=0, calls=['ValueError', '_Track', '_components', '_iou', '_to_gray', 'append']
code/__init__: kind=function, line=217, n_args=10, calls=['ValueError', 'callable', 'float', 'hasattr', 'int', 'kf.clone']
code/_background: kind=function, line=268, n_args=1, calls=['np.median', 'np.stack']
code/_detect: kind=function, line=271, n_args=2, calls=['_components', 'float', 'len', 'mask.copy', 'mask.sum', 'max']
code/_associate: kind=function, line=288, n_args=4, calls=['_iou', 'c.tolist', 'enumerate', 'len', 'linear_sum_assignment', 'list']
code/transform_one: kind=function, line=437, n_args=3, calls=['_to_gray', 'hasattr', 'isinstance', 'np.array', 'np.asarray', 'self._detect']
code/learn_one: kind=function, line=445, n_args=4, calls=['_to_gray', 'append', 'hasattr', 'isinstance', 'len', 'np.asarray']
```

**How it works (docstring):**
1. **Motion detector**: robust running background (median of last `history` frames), flags pixels with `|diff| > n_sigmas × robust_scale`. Morphological erosion + connected components → bounding boxes.
2. **Kalman filter per track**: constant-velocity (state = x, y, w, h, vx, vy, vw, vh), predicts next box.
3. **Two-round association (ByteTrack)**: high-score detections first, then low-score. Unmatched tracks kept alive for `max_missing` frames.
4. **Output**: `{"tracks": [{"id", "box": (x0,y0,x1,y1), "missing"}], "n_tracks"}`.

**Use for D-Avatar:** Track the face as a moving object. Initialize on first frame (or use a synthetic seed). The face box gives ROIs for eyes, mouth, brows.

**References in code:**
- Ma et al. 2023, *Adaptive confidence threshold for ByteTrack*, arXiv:2312.01650 (two-round association)
- Mbalawata & Sarkka 2013, arXiv:1308.5875 (VBAdaptiveKalman available)

---

### 2.3 FrameFeatures — HOG, Moments, Lucas–Kanade Flow Per Cell
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/features.py
```
```
code/FrameFeatures: kind=class, line=112, n_args=0, calls=['ValueError', '_as_array', '_bilinear', '_downsample', '_to_gray', 'all']
code/_hog: kind=function, line=225, n_args=2, calls=['astype', 'enumerate', 'float', 'norm', 'np.arctan2', 'np.bincount']
code/_moments: kind=function, line=244, n_args=2, calls=['float', 'img.mean', 'img.std', 'img.sum', 'max', 'np.indices']
code/_flow_cells_one_level: kind=function, line=290, n_args=5, calls=['enumerate', 'float', 'get_default_dtype', 'it.ravel', 'lstsq', 'np.column_stack']
code/_flow: kind=function, line=320, n_args=3, calls=['_downsample', 'cur_pyr.append', 'float', 'get_default_dtype', 'len', 'np.hypot']
```

**Features (docstring):**
- **HOG**: `cells=(3,4)` grid, `n_bins=9` (Huang & Huang 2017 §2) → `hog_{cy}_{cx}_{b}`
- **Moments**: `mean`, `contrast` (std), `centroid_x`, `centroid_y` (normalized [0,1])
- **Short-range optical flow**: Lucas–Kanade (1981) least squares per flow cell (`flow_cells=(2,3)`), coarse-to-fine pyramid (`flow_levels`); keys `flow_u_{fy}_{fx}`, `flow_v_{fy}_{fx}`, `flow_mag_{fy}_{fx}`

**Lucas–Kanade equations (Ziani 2025, Eqs. 4–5):**
- Solves `A^T A w = -A^T b` per cell where `A = [I_x, I_y]`, `b = I_t`
- Pyramid: 2×2 box averaging (Ziani §3.2), bilinear interpolation (Eq. 9)

**Declared budget:** `budget_s = 0.02` (20 ms), `memory_class = "O(1)"`.

**Use for D-Avatar:** Run `FrameFeatures` on each facial ROI (eye, mouth, brow boxes from `MovingObjectTracker`). The per-cell flow vectors (`flow_u`, `flow_v`) measure local deformation → eye openness, mouth opening, brow raise.

---

### 2.4 DenseFlow — Dense Optical Flow (Alternative to FrameFeatures Flow)
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/dense.py
```
```
code/DenseFlow: kind=class, line=42, n_args=0, calls=['ValueError', '_to_gray', '_unpack', 'astype', 'cv2.calcOpticalFlowFarneback', 'float']
```

**DenseFlow** uses OpenCV's Farnebäck dense optical flow (polynomial expansion). Returns full-flow field. Higher cost than `FrameFeatures` cell flow but gives per-pixel vectors.

**Use for D-Avatar (gaze):** Dense flow in the iris patch → average flow vector = gaze direction. Compare against `PatchMemory` of neutral/left/right/up/down gaze patches.

---

### 2.5 GlobalMotion — Camera Motion (Head Pose Proxy)
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/motion.py
```
```
code/GlobalMotion: kind=class, line=181, n_args=0, calls=['_to_gray', '_unpack', 'astype', 'bool', 'fft2', 'fftfreq']
code/_phase_shift: kind=function, line=255, n_args=3, calls=['fft2', 'fftfreq', 'float', 'ifft2', 'int', 'lstsq']
code/transform_one: kind=function, line=316, n_args=3, calls=['_unpack', 'float', 'np.hypot', 'self._phase_shift', 'self._to_float_gray']
```

**How it works:** Phase correlation via FFT. Computes global translation (dx, dy) between consecutive frames. Exact to 0.1 px on pure translation.

**Use for D-Avatar (head_yaw/pitch/roll):** Global translation + rotation from phase correlation. For rotation: `Registration` (phase correlation on log-polar) or decompose homography. Head pose ≈ camera motion when face is dominant.

---

### 2.6 MotionEnergy — Frame Difference Energy (Blink Detection)
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/motion.py
```
```
code/MotionEnergy: kind=class, line=83, n_args=0, calls=['ValueError', '_box_blur', '_to_gray', '_unpack', 'diff.max', 'diff.mean']
code/_box_blur: kind=function, line=69, n_args=2, calls=['np.cumsum', 'np.hstack', 'np.pad', 'np.vstack', 'np.zeros']
code/transform_one: kind=function, line=160, n_args=3, calls=['_box_blur', '_unpack', 'diff.max', 'diff.mean', 'diff.sum', 'float']
```

**Output per frame:** `max`, `mean`, `sum` of `|diff|` (box-blurred). A blink = sharp spike in eye ROI `sum` or `max`.

**Use for D-Avatar (blink):** Run `MotionEnergy` on eye ROI box from `MovingObjectTracker`. Threshold the energy spike → `blink=1`. Feed energy to `ADWIN` for baseline drift.

---

### 2.7 Registration + Stabilizer — Camera Stabilization
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/registration.py
```
```
code/Registration: kind=class, line=38, n_args=0, calls=['ValueError', '_to_gray', '_unpack', 'astype', 'fft2', 'fftfreq', 'float', 'ifft2']
code/Stabilizer: kind=class, line=112, n_args=0, calls=['Registration', 'ValueError', '_to_gray', '_unpack', 'float']
```

**Registration:** Phase correlation (same as `GlobalMotion`) → global shift (dx, dy).
**Stabilizer:** Applies inverse shift to stabilize frame sequence.

**Use for D-Avatar:** Stabilize face ROI before computing local flow (removes camera motion from facial deformation).

---

### 2.8 PatchMemory — Patch-Based Recognition (Gaze)
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/patches.py
```
```
code/PatchMemory: kind=class, line=54, n_args=0, calls=['ValueError', 'Frame', 'append', 'float', 'len', 'np.argmin', 'np.linalg.norm']
```

**PatchMemory:** Stores a dictionary of patch → label. On new patch, finds nearest neighbor (L2 distance) and returns its label. Fixed memory (FIFO).

**Use for D-Avatar (gaze):** During calibration, store iris patches at known gaze positions (neutral, left, right, up, down). At runtime, extract iris patch from stabilized eye ROI, query `PatchMemory` → interpolate gaze_x, gaze_y from nearest neighbors.

---

### 2.9 TrackToSignal / FeatureToSignal — Video → Signal for Drift
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/adapters.py
```
```
code/TrackToSignal: kind=class, line=29, n_args=0, calls=['Signal', 'ValueError', 'as_default_array', 'dict', 'float', 'get']
code/FeatureToSignal: kind=class, line=118, n_args=0, calls=['Signal', 'as_default_array', 'dict', 'float', 'get', 'get_default_dtype']
```

**TrackToSignal:** From `MovingObjectTracker` track → `Signal(cx, cy, w, h, speed)`.
**FeatureToSignal:** From `FrameFeatures` / `MotionEnergy` dict → `Signal` with one channel per feature.

**Use for D-Avatar:** Wrap eye ROI flow energy, mouth flow, blink energy as `Signal` → feed to `ADWIN` for drift detection (recalibration trigger).

---

### 2.10 ADWIN — Drift Detection (Recalibration Trigger)
```bash
python Dense-Armor/dense_armor/usable/launch.py select detect_drift --budget-ms 5 --top 5
```
```
selection:
== notice when the data changes   6 usable classes   because: 0 rows in order, one every 5 ms
   20.0%  ADWIN, CUSUMDriftDetector [core], KSWIN, PageHinkley
```

```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/online/drift/adwin.py
```
```
code/ADWIN: kind=class, line=73, n_args=0, calls=['ValueError', '_Bucket', '__init__', '_reset', 'append', 'buckets.append']
```

**Use for D-Avatar:** Two `ADWIN` instances:
- `adwin_eye_energy` on eye ROI `MotionEnergy.sum` → detects baseline shift (new person, lighting change) → reset gaze/blink calibration
- `adwin_flow_stats` on `FrameFeatures` flow magnitude stats → same

---

### 2.11 RealtimePipeline + profile — Latency Budget
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/roles/realtime.py
```
```
code/RealtimePipeline: kind=class, line=153, n_args=0, calls=['ValueError', '_budget_of', 'call_with_t', 'enumerate', 'field', 'missing.append']
code/profile: kind=function, line=89, n_args=4, calls=['ProfileResult', '_budget_of', '_memory_class_of', 'arr.max', 'float', 'fn']
```

**Use for D-Avatar:** Declare `budget_s` for each step (10× measured p99). `RealtimePipeline` refuses to build if sum > `period_s`.

---

### 2.12 Sync — Stream Alignment
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/quality.py
```
```
code/Sync: kind=class, line=130, n_args=0, calls=['ValueError', 'a.mean', 'a.std', 'aa.std', 'append', 'b.mean']
```
**Performance:** 2.9 ms error on 20 seeds vs uniform offset [-50, +50] ms.

**Use for D-Avatar:** Align camera stream to external clock (e.g., audio, VMC sender).

---

## 3. Synthetic Test Pipeline (Proposed)

All measurements on synthetic frames from `ArrayStream` (no camera, no MediaPipe).

```python
# Synthetic test: moving face with blinks, gaze shifts, head rotation
# Frame: 120x160, 30 fps, 100 frames
# Face: white ellipse on black, eyes/mouth/brows as darker ellipses
# Motion: affine transform (translation + rotation) + local deformation

from dense_armor.core.signals.vision.streams import ArrayStream
from dense_armor.core.signals.vision.tracking import MovingObjectTracker
from dense_armor.core.signals.vision.features import FrameFeatures
from dense_armor.core.signals.vision.motion import MotionEnergy, GlobalMotion
from dense_armor.core.signals.vision.registration import Stabilizer
from dense_armor.core.signals.vision.patches import PatchMemory
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.roles.realtime import RealtimePipeline, profile

# 1. CameraStream / ArrayStream → frames
# 2. Stabilizer (Registration) → stabilized frames
# 3. MovingObjectTracker → face box (tracks face as moving object)
# 4. For each facial ROI (eyes, mouth, brows from face box):
#    a. FrameFeatures(flow_cells=(2,2), flow_levels=2) → per-cell flow
#    b. MotionEnergy → blink energy
# 5. GlobalMotion → head pose (dx, dy, rotation from phase correlation)
# 6. PatchMemory on iris patch (from stabilized eye ROI) → gaze_x, gaze_y
# 7. FeatureToSignal + ADWIN → drift detection
# 8. RealtimePipeline enforces budget
```

**Expected budgets (estimated, to be measured):**
| Step | Estimated budget_s |
|---|---|
| Stabilizer | 5 ms |
| MovingObjectTracker | 10 ms |
| FrameFeatures (4 ROIs × 2×2 cells) | 8 ms |
| MotionEnergy (2 eye ROIs) | 2 ms |
| GlobalMotion | 3 ms |
| PatchMemory (iris) | 1 ms |
| ADWIN (×2) | <1 ms |
| **Total** | **~29 ms** (fits 33 ms period) |

---

## 4. Paper Citations for Key Equations

| Primitive | Paper | Section / Equation |
|---|---|---|
| Lucas–Kanade (FrameFeatures flow) | Lucas & Kanade 1981, *An Iterative Image Registration Technique with an Application to Stereo Vision*, IJCAI | Iterative LS per cell |
| Lucas–Kanade dense (Ziani) | Ziani 2025, *Dense Optical Flow*, Eqs. 4–5 | `A^T A w = -A^T b` per cell |
| Horn–Schunck (GlobalMotion basis) | Horn & Schunck 1981, *Determining Optical Flow*, AIJ | Global smoothness; phase correlation variant |
| Phase correlation (GlobalMotion/Registration) | Kuglin & Hines 1975, *Phase Correlation*, IEEE | FFT cross-power spectrum |
| ByteTrack association | Ma et al. 2023, *Adaptive confidence threshold for ByteTrack*, arXiv:2312.01650 | Two-round IoU association |
| Kalman filter (MovingObjectTracker) | Mbalawata & Sarkka 2013, arXiv:1308.5875 | VBAdaptiveKalman available |
| ADWIN drift | Bifet & Gavalda 2007, *Learning from Time-Changing Data with Adaptive Windowing* | Adaptive window, split test |
| PatchMemory | Dense-Armor `patches.py` (original) | Nearest-neighbor patch dictionary |
| Farnebäck dense flow | Farnebäck 2003, *Two-Frame Motion Estimation*, SCIA | Polynomial expansion |

---

## 5. No-ML Verification

```python
# d_avatar/tracker.py (to be implemented in Step C)
# NO imports of: mediapipe, torch, tensorflow, onnx, cv2.FaceDetector, etc.
# ONLY imports from dense_armor and numpy

from dense_armor.core.signals.vision.streams import CameraStream, ArrayStream
from dense_armor.core.signals.vision.tracking import MovingObjectTracker
from dense_armor.core.signals.vision.features import FrameFeatures
from dense_armor.core.signals.vision.motion import MotionEnergy, GlobalMotion
from dense_armor.core.signals.vision.registration import Stabilizer
from dense_armor.core.signals.vision.patches import PatchMemory
from dense_armor.core.signals.vision.adapters import FeatureToSignal
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.roles.realtime import RealtimePipeline, profile
```

**Tests:** `tests/test_no_model.py` (Step G) greps source for forbidden imports.

---

## 6. VMC Protocol Mapping (Unchanged)

| D-Avatar Param | VMC Blendshape | Range |
|---|---|---|
| `eye_l`, `eye_r` | `EyeOpen_L`, `EyeOpen_R` | [0, 1] |
| `mouth_open` | `MouthOpen` | [0, 1] |
| `brow_l`, `brow_r` | `BrowUp_L`, `BrowUp_R` | [0, 1] |
| `head_yaw`, `head_pitch`, `head_roll` | `HeadYaw`, `HeadPitch`, `HeadRoll` | degrees |
| `blink` | `Blink` | 0 or 1 |
| `gaze_x`, `gaze_y` | `GazeX`, `GazeY` | [-1, 1] |

**Protocol:** `/VMC/Ext/Blend/Val` (name, value) × N + `/VMC/Ext/Blend/Apply` per frame.

---

## 7. Summary: Model-Free D-Avatar on Dense-Armor Primitives

| D-Avatar Component | Dense-Armor Primitive(s) | Status |
|---|---|---|
| Camera frames | `CameraStream` / `ArrayStream` | ✅ Available |
| Face/ROI tracking | `MovingObjectTracker` (ByteTrack + Kalman) | 🔄 Step C: implement tracker.py |
| Facial deformation (eyes, mouth, brows) | `FrameFeatures._flow` (Lucas-Kanade per cell) on ROIs | 🔄 Step C |
| Blink detection | `MotionEnergy` on eye ROI + threshold | 🔄 Step C |
| Head pose (yaw/pitch/roll) | `GlobalMotion` (phase correlation) + `Registration` | 🔄 Step C |
| Camera stabilization | `Stabilizer` (inverse phase correlation shift) | 🔄 Step C |
| Gaze | `FrameFeatures._flow` or `DenseFlow` on iris patch + `PatchMemory` | 🔄 Step C |
| Drift detection / recalibration | `ADWIN` on `FeatureToSignal`(flow energy, blink energy) | ✅ ADWIN available |
| Latency budget | `RealtimePipeline` + `profile` | ✅ Available |
| Stream alignment | `Sync` | ✅ Available |
| 2D/3D avatar drawing | `draw_avatar()` pure function (Step D) | 🔄 Planned |
| VMC sending | `VmcSender.send_frame()` (Step E) | 🔄 Planned |

**All primitives are causal, online, fixed-memory, no pre-trained models.**

**Next: Step C — Implement `d_avatar/tracker.py` native tracker using the primitives above.**