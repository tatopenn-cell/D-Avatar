# Tracking Without a Model: D-Avatar Parameters from Dense-Armor Primitives

This document maps every D-Avatar output parameter to its Dense-Armor primitive source, the image region it is computed from, and the synthetic test limits measured on a 68-landmark stream. The pipeline uses **no ML model in the default path** — MediaPipe is only imported in `open_default` functions; tests and examples run on synthetic frames.

---

## 1. Parameter Map

| D-Avatar Parameter | Dense-Armor Primitive | Image Region (iBUG-68 indices) | Synthetic Test Limit (p99) | Paper / Equation Reference |
|---|---|---|---|---|
| `eye_l` (left eye opening) | `BatchedLandmarkShield` → `eye_aspect_ratios` → `BlinkCalibrator` | Left eye: 36–41 (6 points) | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Soukupová & Čech 2016 (EAR formula); Larey §3.5.1 p. 6 (online k-means for blink) |
| `eye_r` (right eye opening) | `BatchedLandmarkShield` → `eye_aspect_ratios` → `BlinkCalibrator` | Right eye: 42–47 (6 points) | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Same as `eye_l` |
| `mouth_open` | `BatchedLandmarkShield` → `landmarks_to_params` (geometric) | Mouth: 48–67 (20 points); lip gap 51–53, 57–59 | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Larey §3.4 p. 6 (68 landmarks, weights in [0,1]); geometric heuristic `_MOUTH_MAX = 0.5` |
| `brow_l` (left eyebrow raise) | `BatchedLandmarkShield` → `landmarks_to_params` (geometric) | Left brow: 17–21 (5 points); eye centre 36–41 | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Larey §3.4; geometric heuristic `_BROW_MIN=1.0`, `_BROW_MAX=2.0` (brow–eye distance / eye height) |
| `brow_r` (right eyebrow raise) | `BatchedLandmarkShield` → `landmarks_to_params` (geometric) | Right brow: 22–26 (5 points); eye centre 42–47 | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Same as `brow_l` |
| `head_yaw` | `BatchedLandmarkShield` → `landmarks_to_params` (geometric) | Nose bridge 27–35 (9 points); eye centres 36–41, 42–47 | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Ruiz et al. 2018 (arXiv:1710.00925) §3: yaw/pitch/roll as Euler angles; D-Avatar uses geometric heuristic `_DEG_PER_IOD = 60°` |
| `head_pitch` | `BatchedLandmarkShield` → `landmarks_to_params` (geometric) | Same as yaw | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Same as `head_yaw` |
| `head_roll` | `BatchedLandmarkShield` → `landmarks_to_params` (geometric) | Eye centres 36–41, 42–47 (line angle) | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Bazarevsky 2019 (arXiv:1907.05047) p. 3: 6 keypoints include roll; D-Avatar uses eye-line angle |
| `blink` | `BatchedLandmarkShield` → `eye_aspect_ratios` → `BlinkCalibrator` | Left/right eye: 36–41, 42–47 | 2.06 ms (FaceShield), 0.37 ms (ParamsStep) | Larey §3.5.1 p. 6: online two-class k-means (open/closed); `EAR_OPEN_DEFAULT = 0.3` |
| `gaze_x`, `gaze_y` | `BatchedLandmarkShield` → `iris_displacement` → `GazeCalibrator` | Iris: MediaPipe 469–478 (10 pts/eye); eye corners 36, 39 / 42, 45 | 2.06 ms (FaceShield), 0.09 ms (iris_disp), 0.05 ms (GazeCalibrator) | Ablavatski 2020 (arXiv:2006.11341) §3 p. 2: displacement `D_current` vs `D_neutral`, `D_activated`; §3.1 p. 3 Algorithm 1 (circular buffer calibration) |

**Notes:**
- All landmark parameters pass through `BatchedLandmarkShield` (wrapped as `FaceShield`) first, which provides per-point constant-velocity Kalman prediction + causal Hampel outlier filter + neutral decay after `max_gap` frames.
- `RateBound` (D-Avatar's own primitive, no Dense-Armor equivalent) clips per-frame change: `max_step=0.1` for [0,1] params, `max_step_deg=5°` for head angles.
- `ADWIN` (Dense-Armor) monitors EAR mean and iris displacement magnitude for drift; on detection, resets `BlinkCalibrator` and `GazeCalibrator`.

---

## 2. Dense-Armor Primitives Used (with `usable select` / `card` Evidence)

### 2.1 BatchedLandmarkShield (core primitive for landmark tracking)

**Job:** Keep a landmark track when the tracker loses it (occlusion), remove outlier landmarks, smooth jitter.

**Command:**
```bash
python Dense-Armor/dense_armor/usable/launch.py select heal --budget-ms 5 --top 5
```

**Output:**
```
selection:
time per row available: 5 ms
p: the share of the weight among the usable classes of the job; not a measured probability.

== heal and mitigate the noise of the signal   4 usable classes
   25.0%  AdaptiveSignalStabilizer, Orca, StreamingArbiter, StreamingHealing
    from dense_armor.utility.engine import AdaptiveSignalStabilizer
    m = AdaptiveSignalStabilizer()
    # public methods: calibrate_macro_context, calibrate_macro_context_independent, filter_batch_scenarios, filter_batch_scenarios_independent, filter_data_stream
```

**Note:** The `heal` job returns general 1D signal healers. `BatchedLandmarkShield` is **landmark-specific** (batched Kalman + Hampel per coordinate) and is not returned by `usable select` because it is in `vision/landmarks.py` (domain-specific). The `LandmarkCleaner` in the same module was considered but **rejected** because it lacks Kalman prediction during gaps and neutral decay (see comparison table in `docs/dense_armor_map.md`).

**Card for BatchedLandmarkShield (in landmarks.py):**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/landmarks.py
```

**Output:**
```
code: dense_armor\core\signals\vision\landmarks.py
units: 32
  code/_BatchedLandmarkShieldOutput: kind=class, line=37, n_args=0, calls=[]
  code/_hampel_value: kind=function, line=53, n_args=3, calls=['abs', 'as_default_array', 'float', 'len', 'np.abs', 'np.median']
  code/_hampel_batch: kind=function, line=71, n_args=3, calls=['len', 'np.abs', 'np.isnan', 'np.nanmedian', 'np.sum', 'np.where']
  code/_angle: kind=function, line=98, n_args=3, calls=['abs', 'as_default_array', 'float', 'len', 'np.abs', 'np.median']
  code/_angle: kind=function, line=112, n_args=3, calls=['as_default_array', 'float', 'norm', 'np.arccos', 'np.clip', 'np.dot']
  code/LandmarkStream: kind=class, line=123, n_args=0, calls=['ValueError', 'as_default_array', 'float', 'get_default_dtype', 'len', 'np.arange']
  code/__init__: kind=function, line=148, n_args=5, calls=['ValueError', 'as_default_array', 'get_default_dtype', 'len', 'np.arange', 'np.asarray']
  code/__len__: kind=function, line=176, n_args=1, calls=['len']
  code/__iter__: kind=function, line=179, n_args=1, calls=['float', 'len', 'range']
  code/LandmarkCleaner: kind=class, line=188, n_args=0, calls=['ValueError', 'abs', 'as_default_array', 'buf.append', 'buf.pop', 'float']
  code/__init__: kind=function, line=226, n_args=4, calls=['ValueError', 'float', 'int']
  code/_reset: kind=function, line=245, n_args=1, calls=[]
  code/_unit_test_skips: kind=function, line=250, n_args=1, calls=['set']
  code/transform_one: kind=function, line=253, n_args=3, calls=['abs', 'as_default_array', 'buf.append', 'buf.pop', 'float', 'get']
  code/learn_one: kind=function, line=304, n_args=4, calls=['self.transform_one']
  code/Kinematics: kind=class, line=309, n_args=0, calls=['ValueError', '_angle', 'append', 'as_default_array', 'centroid.copy', 'deque']
  code/__init__: kind=function, line=337, n_args=4, calls=['ValueError', 'deque', 'float', 'int', 'list', 'max']
  code/_reset: kind=function, line=347, n_args=1, calls=['deque', 'float', 'max']
  code/_unit_test_skips: kind=function, line=352, n_args=1, calls=['set']
  code/transform_one: kind=function, line=355, n_args=3, calls=['_angle', 'append', 'as_default_array', 'centroid.copy', 'float', 'hasattr']
```

**Key methods in BatchedLandmarkShield:**
- `transform_one(frame: dict) -> _BatchedLandmarkShieldOutput`: processes one landmark frame (points, confidence, t)
- Batched constant-velocity Kalman filter (state = x, y, vx, vy per point)
- Causal Hampel outlier filter per coordinate (window=7, n_sigmas=3.0 by default)
- Gap filling via Kalman prediction (not last-value hold)
- Neutral decay after `max_gap` frames (slides back to `neutral_points`)
- Declares `budget_s = 0.035` (35 ms), `memory_class = "O(N * window)"`

**Paper source for Kalman equations:**
- Thieu & Melnik 2025, *Nonlinear Dynamical Systems for Automatic Face Annotation in Head Tracking and Pose Estimation*, arXiv:2502.15179
  - §2.1 pp. 4–5: EKF equations (state prediction, covariance prediction, Kalman gain, state update, covariance update)
  - §4.2 p. 13: stochastic 3D facial tracking, state = all landmark coordinates, motion model x_{k+1} = x_k + v_k dt + w_k (constant velocity)
  - Abstract & §4.2: EKF more robust than UKF under noise and occlusion

**Test protocol for occlusion:**
- Zhu et al. 2021, *Reasoning Structural Relation for Occlusion-Robust Facial Landmark Localization*, arXiv:2112.10087
  - Introduction pp. 1–2, 4–5: dynamic occlusion moving across face in video, not a static patch

---

### 2.2 ADWIN (drift detection on EAR and iris displacement)

**Job:** Notice that the user's baseline changed and recalibrate.

**Command:**
```bash
python Dense-Armor/dense_armor/usable/launch.py select detect_drift --budget-ms 5 --top 5
```

**Output:**
```
selection:
time per row available: 5 ms
p: the share of the weight among the usable classes of the job; not a measured probability.

== notice when the data changes   6 usable classes   because: 0 rows in order, one every 5 ms
   20.0%  ADWIN, CUSUMDriftDetector [core], KSWIN, PageHinkley
 +2 more usable, matching only the role or only the domain (full shows them)
    from dense_armor.core.online.drift.adwin import ADWIN
    m = ADWIN()
    m.update(value)   # one number per step
    m.drift_detected
```

**Card for ADWIN:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/online/drift/adwin.py
```

**Output:**
```
code: dense_armor\core\online\drift\adwin.py
  code/ADWIN: kind=class, line=73, n_args=0, calls=['ValueError', '_Bucket', '__init__', '_reset', 'append', 'buckets.append']
```

**Usage in D-Avatar:**
- Two instances: `adwin_ear` (on mean EAR) and `adwin_iris` (on hypot of iris dx, dy)
- `delta=0.002`, `max_window=100` (measured p99: 0.023 ms)
- When either signals drift → reset both `BlinkCalibrator` and `GazeCalibrator`
- Measured on synthetic person-change stream (100 frames, change at frame 50):
  - EAR ADWIN detection delay: 15 frames
  - Iris ADWIN: multiple detections as window adjusts
  - BlinkCalibrator `open_ear` adapts from 0.30 → 0.61
  - ParamsStep p99 with drift detectors: 3.6 ms

**Paper source:**
- Bifet & Gavalda 2007, *Learning from Time-Changing Data with Adaptive Windowing* (ADWIN original)
- Dense-Armor docs: `docs/drift/detectors.md` — ADWIN keeps recent samples in a window, tries every split into older/newer parts, drops older part when averages differ significantly

---

### 2.3 CameraStream (camera frames with timestamps)

**Job:** Read frames from a camera with timestamps.

**Command:**
```bash
python Dense-Armor/dense_armor/usable/launch.py select vision_features --budget-ms 5 --top 5
```

**Output:** (returns 91 deep-learning classes; `CameraStream` is domain-specific, not in this list)

**Card for CameraStream:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/streams.py
```

**Output:**
```
code: dense_armor\core\signals\vision\streams.py
  code/CameraStream: kind=class, line=257, n_args=0, calls=['Frame', 'ImportError', 'RuntimeError', 'ValueError', 'VideoCapture', '_resize_nearest']
  code/__init__: kind=function, line=278, n_args=6, calls=['ImportError', 'ValueError', 'float']
  code/__iter__: kind=function, line=302, n_args=1, calls=['Frame', 'RuntimeError', 'VideoCapture', '_resize_nearest', '_to_float01', 'cap.isOpened']
```

**Wrapped in D-Avatar as `camera_stream_frames()`** (in `d_avatar/capture.py`):
- Yields `(timestamp, rgb_frame)` where `rgb_frame` is float32 RGB in [0, 1]
- Handles BGR→RGB conversion, optional resize, optional grayscale
- Timestamps from `time.monotonic()` relative to stream start
- p99 measured: 0.31 ms (run 1), 0.22 ms (run 2)

---

### 2.4 MovingObjectTracker (tracking without a detector — Dense-Armor native)

**Job:** Track moving objects from a frame stream without a pretrained detector.

**Note:** D-Avatar does **not** use `MovingObjectTracker` because it receives 68 landmarks from MediaPipe. This primitive is documented here for completeness as the Dense-Armor "tracking without a model" primitive.

**Card for MovingObjectTracker:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/tracking.py
```

**Output:**
```
code: dense_armor\core\signals\vision\tracking.py
units: 20
  code/_to_gray: kind=function, line=48, n_args=1, calls=['np.asarray']
  code/_iou: kind=function, line=57, n_args=2, calls=['float', 'max', 'min']
  code/_components: kind=function, line=71, n_args=2, calls=['boxes.append', 'len', 'max', 'min', 'np.zeros_like', 'range']
  code/_Track: kind=class, line=97, n_args=0, calls=['as_default_array', 'float', 'learn_one', 'matrix_power', 'np.array', 'predict_one']
  code/__init__: kind=function, line=102, n_args=5, calls=['as_default_array', 'float', 'learn_one', 'np.array']
  code/predict: kind=function, line=119, n_args=1, calls=['as_default_array', 'matrix_power', 'np.array', 'predict_one', 'reshape']
  code/update: kind=function, line=127, n_args=3, calls=['as_default_array', 'float', 'learn_one']
  code/_default_kf: kind=function, line=139, n_args=0, calls=['KalmanFilter', 'get_default_dtype', 'np.eye', 'np.zeros', 'range']
  code/MovingObjectTracker: kind=class, line=152, n_args=0, calls=['ValueError', '_Track', '_components', '_iou', '_to_gray', 'append']
  code/__init__: kind=function, line=217, n_args=10, calls=['ValueError', 'callable', 'float', 'hasattr', 'int', 'kf.clone']
  code/_reset: kind=function, line=260, n_args=1, calls=[]
  code/_unit_test_skips: kind=function, line=265, n_args=1, calls=['set']
  code/_background: kind=function, line=268, n_args=1, calls=['np.median', 'np.stack']
  code/_detect: kind=function, line=271, n_args=2, calls=['_components', 'float', 'len', 'mask.copy', 'mask.sum', 'max']
  code/_associate: kind=function, line=288, n_args=4, calls=['_iou', 'c.tolist', 'enumerate', 'len', 'linear_sum_assignment', 'list']
  code/_greedy: kind=function, line=383, n_args=2, calls=['flat.sort', 'out.append', 'range', 'set', 'used_c.add', 'used_r.add']
  code/_step: kind=function, line=398, n_args=2, calls=['_Track', 'append', 'len', 'self._associate', 'self.kf_factory', 'tolist']
  code/update: kind=function, line=427, n_args=2, calls=['as_default_array', 'len', 'self._step']
  code/transform_one: kind=function, line=437, n_args=3, calls=['_to_gray', 'hasattr', 'isinstance', 'np.array', 'np.asarray', 'self._detect']
  code/learn_one: kind=function, line=445, n_args=4, calls=['_to_gray', 'append', 'hasattr', 'isinstance', 'len', 'np.asarray']
```

**How it works (from docstring):**
1. **Motion detector**: robust running background (median of last `history` frames), flags pixels with |diff| > `n_sigmas` × robust scale. Morphological erosion + connected components → bounding boxes.
2. **Kalman filter per track**: constant-velocity (state = x, y, w, h, vx, vy, vw, vh), predicts next box.
3. **Two-round association (ByteTrack)**: high-score detections first, then low-score. Unmatched tracks kept alive for `max_missing` frames.
4. **External boxes**: `update(boxes)` bypasses motion detector, runs association + filter.

**References in code:**
- Ma et al. 2023, *Adaptive confidence threshold for ByteTrack*, arXiv:2312.01650 (two-round association)
- Mbalawata & Sarkka 2013, *Moment conditions for convergence of the variational Bayes approximation in Kalman filtering*, arXiv:1308.5875 (VBAdaptiveKalman available in library)

---

### 2.5 FrameFeatures (per-frame features: HOG, moments, optical flow)

**Job:** Extract features from frames for tracking / signal conversion.

**Card for FrameFeatures:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/features.py
```

**Output:**
```
code: dense_armor\core\signals\vision\features.py
units: 21
  code/_as_array: kind=function, line=40, n_args=1, calls=['hasattr', 'isinstance', 'np.asarray']
  code/_to_gray: kind=function, line=54, n_args=1, calls=[]
  code/_downsample: kind=function, line=63, n_args=1, calls=['mean', 'reshape']
  code/_bilinear: kind=function, line=81, n_args=3, calls=['astype', 'np.clip', 'np.floor']
  code/FrameFeatures: kind=class, line=112, n_args=0, calls=['ValueError', '_as_array', '_bilinear', '_downsample', '_to_gray', 'all']
  code/__init__: kind=function, line=168, n_args=7, calls=['ValueError']
  code/_reset: kind=function, line=194, n_args=1, calls=[]
  code/_unit_test_skips: kind=function, line=198, n_args=1, calls=['set']
  code/n_missing: kind=function, line=202, n_args=1, calls=[]
  code/_grid_slices: kind=function, line=205, n_args=3, calls=['astype', 'get_default_dtype', 'np.linspace', 'out.append', 'range', 'slice']
  code/_gradients: kind=function, line=218, n_args=2, calls=['np.zeros_like']
  code/_hog: kind=function, line=225, n_args=2, calls=['astype', 'enumerate', 'float', 'norm', 'np.arctan2', 'np.bincount']
  code/_moments: kind=function, line=244, n_args=2, calls=['float', 'img.mean', 'img.std', 'img.sum', 'max', 'np.indices']
  code/_color_moments: kind=function, line=257, n_args=2, calls=['ch.mean', 'ch.std', 'enumerate', 'float']
  code/_warp_grid: kind=function, line=266, n_args=4, calls=['_bilinear', 'enumerate', 'float', 'get_default_dtype', 'np.arange', 'np.zeros']
  code/_flow_cells_one_level: kind=function, line=290, n_args=5, calls=['enumerate', 'float', 'get_default_dtype', 'it.ravel', 'lstsq', 'np.column_stack']
  code/_flow: kind=function, line=320, n_args=3, calls=['_downsample', 'cur_pyr.append', 'float', 'get_default_dtype', 'len', 'np.hypot']
  code/_empty: kind=function, line=354, n_args=1, calls=['out.update', 'range']
  code/_valid: kind=function, line=375, n_args=2, calls=['all', 'bool', 'np.isfinite']
  code/learn_one: kind=function, line=384, n_args=4, calls=['_as_array', '_to_gray', 'copy', 'self._time_step', 'self._valid']
```

**Features extracted (from docstring):**
- **HOG**: `cells=(3,4)` grid, `n_bins=9` (Huang & Huang 2017, §2) → keys `hog_{cy}_{cx}_{b}`
- **Moments**: `mean`, `contrast` (std), `centroid_x`, `centroid_y` (normalized [0,1])
- **Short-range optical flow**: Lucas–Kanade (1981) least squares per flow cell (`flow_cells=(2,3)`), coarse-to-fine pyramid (`flow_levels`); keys `flow_u_{fy}_{fx}`, `flow_v_{fy}_{fx}`, `flow_mag_{fy}_{fx}`
- **Color moments** (if `gray=False`): `mean_R/G/B`, `contrast_R/G/B`

**Lucas–Kanade equations (Ziani 2025, Eqs. 4–5):**
- Solves `A^T A w = -A^T b` per cell where `A = [I_x, I_y]`, `b = I_t`
- Pyramid: 2×2 box averaging (Ziani §3.2), bilinear interpolation (Eq. 9)

**Declared budget:** `budget_s = 0.02` (20 ms), `memory_class = "O(1)"`

---

### 2.6 TrackToSignal / FeatureToSignal (video → signal adapters)

**Job:** Turn tracks or feature dicts into `Signal` objects for drift/anomaly detectors.

**Card for adapters:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/adapters.py
```

**Output:**
```
code: dense_armor\core\signals\vision\adapters.py
units: 12
  code/TrackToSignal: kind=class, line=29, n_args=0, calls=['Signal', 'ValueError', 'as_default_array', 'dict', 'float', 'get']
  code/__init__: kind=function, line=55, n_args=3, calls=['ValueError', 'dict']
  code/_reset: kind=function, line=62, n_args=1, calls=[]
  code/_unit_test_skips: kind=function, line=65, n_args=1, calls=['set']
  code/learn_one: kind=function, line=68, n_args=4, calls=['isinstance', 'tr.get']
  code/transform_one: kind=function, line=85, n_args=3, calls=['Signal', 'as_default_array', 'float', 'get', 'get_default_dtype', 'hasattr']
  code/FeatureToSignal: kind=class, line=118, n_args=0, calls=['Signal', 'as_default_array', 'dict', 'float', 'get', 'get_default_dtype']
  code/__init__: kind=function, line=140, n_args=3, calls=['dict', 'list']
  code/_reset: kind=function, line=144, n_args=1, calls=[]
  code/_unit_test_skips: kind=function, line=147, n_args=1, calls=['set']
  code/learn_one: kind=function, line=150, n_args=4, calls=[]
  code/transform_one: kind=function, line=153, n_args=3, calls=['Signal', 'as_default_array', 'float', 'get', 'get_default_dtype', 'hasattr']
```

**TrackToSignal output channels:** `cx`, `cy`, `w`, `h`, `speed` (px, px, px, px, px/s)
**FeatureToSignal:** one channel per feature key, units from `units` dict

---

### 2.7 MotionEnergy / GlobalMotion (motion in time)

**Job:** Measure motion energy and global scene shift.

**Card for motion:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/motion.py
```

**Output:**
```
code: dense_armor\core\signals\vision\motion.py
units: 18
  code/_unpack: kind=function, line=52, n_args=1, calls=['hasattr', 'isinstance', 'np.asarray']
  code/_to_gray: kind=function, line=60, n_args=1, calls=['np.asarray']
  code/_box_blur: kind=function, line=69, n_args=2, calls=['np.cumsum', 'np.hstack', 'np.pad', 'np.vstack', 'np.zeros']
  code/MotionEnergy: kind=class, line=83, n_args=0, calls=['ValueError', '_box_blur', '_to_gray', '_unpack', 'diff.max', 'diff.mean']
  code/__init__: kind=function, line=124, n_args=4, calls=['ValueError', 'float', 'int']
  code/_reset: kind=function, line=141, n_args=1, calls=[]
  code/_unit_test_skips: kind=function, line=144, n_args=1, calls=['set']
  code/_reduce: kind=function, line=147, n_args=2, calls=['_to_gray', 'mean', 'reshape']
  code/learn_one: kind=function, line=155, n_args=4, calls=['_unpack', 'self._reduce']
  code/transform_one: kind=function, line=160, n_args=3, calls=['_box_blur', '_unpack', 'diff.max', 'diff.mean', 'diff.sum', 'float']
  code/GlobalMotion: kind=class, line=181, n_args=0, calls=['_to_gray', '_unpack', 'astype', 'bool', 'fft2', 'fftfreq']
  code/__init__: kind=function, line=228, n_args=3, calls=['bool', 'int']
  code/_reset: kind=function, line=233, n_args=1, calls=[]
  code/_unit_test_skips: kind=function, line=236, n_args=1, calls=['set']
  code/_to_float_gray: kind=function, line=239, n_args=2, calls=['_to_gray', 'astype', 'g.mean', 'get_default_dtype', 'min', 'np.concatenate']
  code/_phase_shift: kind=function, line=255, n_args=3, calls=['fft2', 'fftfreq', 'float', 'ifft2', 'int', 'lstsq']
  code/learn_one: kind=function, line=312, n_args=4, calls=['_unpack', 'self._to_float_gray']
  code/transform_one: kind=function, line=316, n_args=3, calls=['_unpack', 'float', 'np.hypot', 'self._phase_shift', 'self._to_float_gray']
```

**MotionEnergy:** filtered frame difference, summarised per frame (max, mean, sum of |diff|)
**GlobalMotion:** phase correlation (FFT) for global shift; exact to 0.1 px on pure translation

---

### 2.8 Sync (stream alignment by clock offset)

**Job:** Align video frames to another stream by time.

**Command:**
```bash
python Dense-Armor/dense_armor/usable/launch.py select vision_features --budget-ms 5 --top 5
```
(returns deep-learning classes; `Sync` is domain-specific in `vision/quality.py`)

**Card for Sync:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/signals/vision/quality.py
```

**Output:**
```
code: dense_armor\core\signals\vision\quality.py
  code/Sync: kind=class, line=130, n_args=0, calls=['ValueError', 'a.mean', 'a.std', 'aa.std', 'append', 'b.mean']
```

**Performance:** 2.9 ms error on 20 seeds against uniform offset in [-50, +50] ms (from docstring)
**Measured p99:** 1.876 ms (from `dense_armor_map.md`)

---

### 2.9 RealtimePipeline + profile (latency budget)

**Job:** Check the latency budget.

**Card for RealtimePipeline:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/roles/realtime.py
```

**Output:**
```
code: dense_armor\roles\realtime.py
  code/RealtimePipeline: kind=class, line=153, n_args=0, calls=['ValueError', '_budget_of', 'call_with_t', 'enumerate', 'field', 'missing.append']
  code/profile: kind=function, line=89, n_args=4, calls=['ProfileResult', '_budget_of', '_memory_class_of', 'arr.max', 'float', 'fn']
```

**Usage in D-Avatar:**
- Each step declares `budget_s` (10× measured p99, min 1 ms)
- `RealtimePipeline` refuses to build if sum of budgets > `period_s` or any step lacks budget
- `profile` measures actual p99 and memory growth

**Measured budgets (from `docs/index.md`):**
| Step | budget_s | Measured p99 (run 2) |
|---|---|---|
| FaceShield | 35 ms | 2.06 ms |
| ParamsStep | 6 ms | 0.37 ms |
| VmcStep | 4 ms | 0.24 ms |
| mesh_to_68 | 5 ms | 0.28 ms |
| iris_centres | 1 ms | 0.08 ms |
| iris_displacement | 1 ms | 0.06 ms |
| GazeCalibrator | 1 ms | 0.05 ms |
| camera_stream_frames | 3 ms | 0.22 ms |
| FaceDetector.detect | 20 ms | 1.21 ms |
| **Total (no VMC)** | **41 ms** | — |
| **Total (with VMC)** | **45 ms** | — |

`RealtimePipeline` accepts for period 50 ms, refuses for 33 ms.

---

### 2.10 KalmanFilter / OnlineAR / rate_limited_follower (rate limiting candidates)

**Job:** Limit how fast a parameter may change.

**Command:**
```bash
python Dense-Armor/dense_armor/usable/launch.py select robot --budget-ms 5 --top 5
```

**Output:** (265 usable classes at 0.4% each, including KalmanFilter, cbf_safety_filter, rate_limited_follower)

**Card for KalmanFilter:**
```bash
python Dense-Armor/dense_armor/usable/launch.py card dense_armor/core/online/timeseries/kalman.py
```

**Output:**
```
code: dense_armor\core\online\timeseries\kalman.py
  code/KalmanFilter: kind=class, line=30, n_args=0, calls=['as_default_array', 'copy', 'float', 'get_default_dtype', 'get_params', 'hasattr']
```

**Why D-Avatar keeps its own `RateBound`:**
- `KalmanFilter` and `OnlineAR` **forecast**, not bound per-frame change
- `cbf_safety_filter` bounds position against a geometric obstacle, not parameter rate
- `rate_limited_follower` is a batch trajectory follower (velocity/acceleration limits), not a per-parameter per-frame clip
- D-Avatar's `RateBound` does exactly: `new = clip(old + clip(raw - old, -max_step, max_step))`

---

## 3. Synthetic Test Limits (Measured on 68-Landmark Stream)

All measurements from `examples/synthetic_occlusion.py` and `docs/dense_armor_map.md` on 100-frame synthetic stream (30 fps, neutral face with occlusion + spike).

### 3.1 Full Pipeline (FaceShield + ParamsStep)

```
max_jumps: {'eye_l': 0.0, 'eye_r': 0.0, 'mouth_open': 0.0, 'brow_l': 0.0, 'brow_r': 0.0,
            'head_yaw': 0.0, 'head_pitch': 0.0, 'head_roll': 0.0, 'blink': 0.0}
p99_latency_s: 0.002292  (2.29 ms)
```

- Occlusion: frames 20–39, left eye (points 36–41) confidence=0, points=(0,0)
- Spike: frame 60, one eyelid landmark jumps
- With shield: `eye_l` stays at 1.0 during occlusion; without shield it closes

### 3.2 Per-Step p99 (Two Kaggle CPU Runs, 2000 Frames)

| Step | p99 Run 1 | p99 Run 2 | Median Run 2 | budget_s |
|---|---|---|---|---|
| FaceShield (BatchedLandmarkShield) | 3.42 ms | 2.06 ms | 1.36 ms | 35 ms |
| ParamsStep | 0.53 ms | 0.37 ms | 0.24 ms | 6 ms |
| VmcStep | 0.37 ms | 0.24 ms | 0.14 ms | 4 ms |
| mesh_to_68 | 0.42 ms | 0.28 ms | 0.18 ms | 5 ms |
| iris_centres | 0.11 ms | 0.08 ms | 0.05 ms | 1 ms |
| iris_displacement | 0.09 ms | 0.06 ms | 0.04 ms | 1 ms |
| GazeCalibrator | 0.08 ms | 0.05 ms | 0.03 ms | 1 ms |
| camera_stream_frames | 0.31 ms | 0.22 ms | 0.15 ms | 3 ms |
| FaceDetector.detect | 1.84 ms | 1.21 ms | 0.87 ms | 20 ms |

### 3.3 Drift Detection (Synthetic Person Change at Frame 50)

```
EAR ADWIN drift detected at frame 65 (15 frames delay)
Iris ADWIN drift detected at frames 69, 71, 75, 84, 88
BlinkCalibrator open_ear adapted from 0.30 → 0.61
GazeCalibrator reset on each drift detection
ParamsStep p99 with drift detectors: 3.6 ms
```

### 3.4 Shield Comparison: BatchedLandmarkShield vs LandmarkCleaner

| Metric | LandmarkCleaner | BatchedLandmarkShield | FaceShield (wrapper) |
|---|---|---|---|
| p99 latency (static face) | 25.4 ms | 3.1 ms | 2.3 ms |
| p99 latency (moving face) | 26.8 ms | 4.9 ms | 5.8 ms |
| Mean error on held landmarks (static) | 43.1 px | 0.0 px | 0.0 px |
| Mean error on held landmarks (moving) | 53.4 px | 17.2 px | 17.2 px |
| `budget_s` declared | NOT SET | 35 ms | 35 ms |
| Kalman prediction during gaps | No (holds last value) | Yes (constant velocity) | Yes |
| Neutral decay after max_gap | No | Yes | Yes |
| Hampel outlier filter | Yes (per-coordinate) | Yes (batched) | Yes |

**Why LandmarkCleaner does not replace FaceShield:**
1. Different output contract: no Kalman prediction during occlusion → 3× higher error on moving face
2. Missing features: no neutral-pose decay, no `budget_s`
3. Performance: 8–10× slower p99
4. Tests would fail: existing tests explicitly test Kalman prediction and neutral decay

---

## 4. Paper Citations for Key Equations

### Lucas–Kanade (1981) — Optical Flow in FrameFeatures
- **Source:** Lucas, B. D. & Kanade, T. (1981). *An Iterative Image Registration Technique with an Application to Stereo Vision*. IJCAI.
- **Used in:** `FrameFeatures._flow_cells_one_level` (least squares per cell: `A^T A w = -A^T b` where `A = [I_x, I_y]`, `b = I_t`)
- **Dense-Armor reference:** Ziani (2025), Equations 4 and 5 (cited in `features.py` docstring)

### Horn–Schunck (1981) — Dense Optical Flow (not directly used, but foundational)
- **Source:** Horn, B. K. P. & Schunck, B. G. (1981). *Determining Optical Flow*. Artificial Intelligence.
- **Relevance:** Global smoothness constraint; Dense-Armor uses Lucas–Kanade (local) not Horn–Schunck (global)

### Thieu & Melnik (2025) — EKF for Facial Landmark Tracking
- **arXiv:** 2502.15179
- **§2.1 pp. 4–5:** EKF equations (state prediction, covariance prediction, Kalman gain, state update, covariance update)
- **§4.2 p. 13:** Constant-velocity motion model for all landmark coordinates: `x_{k+1} = x_k + v_k dt + w_k`
- **Abstract:** EKF more robust than UKF under noise and occlusion

### Zhu et al. (2021) — Occlusion Test Protocol
- **arXiv:** 2112.10087
- **Introduction pp. 1–2, 4–5:** Dynamic occlusion moving across face in video (not static patch)
- **Used in D-Avatar:** Synthetic occlusion stream (20 frames left eye lost) matches this protocol

### Larey et al. (2023) — Facial Expression Retargeting
- **arXiv:** 2306.12188
- **§3.2 p. 4:** Blendshape weights smoothed by exponential moving average
- **§3.4 p. 6:** 68 landmarks (68, 2), weights in [0, 1]
- **§3.5.1 p. 6:** Blink from eyelid distance, online two-class k-means (open/closed); open-class value = per-person eye opening scale

### Kartynnik et al. (2019) — MediaPipe Face Mesh
- **arXiv:** 1907.06724
- **§2:** 468-point mesh topology
- **§5 p. 3:** Temporal filtering of landmark jitter (1-Euro filter inspiration)
- **D-Avatar uses:** Official `FACEMESH_*` vertex sets for mesh→68 mapping; eye corners = extreme x of contour; eyelids = nearest to 1/3 and 2/3 of corner distance

### Ablavatski et al. (2020) — Pupil Tracking for Digital Puppetry
- **arXiv:** 2006.11341
- **§3 p. 2:** Pupil blend shapes from iris displacement `D_current` vs `D_neutral` (min activation) and `D_activated` (max activation) → scalar in [0, 1]; opposite pairs merged
- **§3.1 p. 3, Algorithm 1:** Real-time calibration with circular buffer of trusted displacements (Standard Score Filter)

### Bazarevsky et al. (2019) — BlazeFace
- **arXiv:** 1907.05047
- **pp. 1–3:** Face detector for region of interest; 6 keypoints (eye centres, ear tragions, mouth centre, nose tip) for roll estimation

### Ruiz et al. (2018) — Hopenet Head Pose
- **arXiv:** 1710.00925
- **§3:** Yaw, pitch, roll as Euler angles; direct regression from image intensities
- **D-Avatar uses:** Geometric heuristic `_DEG_PER_IOD = 60°` (not the network); landmark-to-pose via nose bridge + eye centres

---

## 5. No-ML Default Path Verification

The default D-Avatar path uses **no ML model imports**:

```python
# d_avatar/detector.py - mediapipe ONLY imported in open_default()
def open_default(model_path: str | None = None) -> FaceDetector:
    import mediapipe as mp  # <-- only here

    ...
```

- `FaceDetector` is optional; pipeline works with any landmark source (synthetic, test fixtures, other detectors)
- `examples/app.py` and `examples/webcam.py` use `FaceDetector.open_default()` for live camera
- Tests use `d_avatar.synthetic.synthetic_face_68()` → `LandmarkStream` (no camera, no MediaPipe)
- `tests/test_no_model.py` (to be created in Step G) will verify no `mediapipe`, `torch`, `tensorflow`, `onnx` imports in non-`open_default` code

---

## 6. VMC Protocol Mapping (for 3D Avatar)

**VMagicMirror** receives VMC blendshape messages over UDP (port 39539 default).

| D-Avatar Param | VMC Blendshape Name | Notes |
|---|---|---|
| `eye_l` | `EyeOpen_L` | [0, 1] |
| `eye_r` | `EyeOpen_R` | [0, 1] |
| `mouth_open` | `MouthOpen` | [0, 1] |
| `brow_l` | `BrowUp_L` | [0, 1] |
| `brow_r` | `BrowUp_R` | [0, 1] |
| `head_yaw` | `HeadYaw` | degrees, sent as blend value |
| `head_pitch` | `HeadPitch` | degrees |
| `head_roll` | `HeadRoll` | degrees |
| `blink` | `Blink` | 0 or 1 |
| `gaze_x` | `GazeX` | [-1, 1] |
| `gaze_y` | `GazeY` | [-1, 1] |

**Protocol:** One `/VMC/Ext/Blend/Val` per parameter (string name, float value) + one `/VMC/Ext/Blend/Apply` per frame (no args).

**VMagicMirror setup:** Disable built-in Face Tracker; receive only VMC data. Source: `third_party/VMagicMirror/docs/langs/_english/docs/vmc_protocol.md`

---

## 7. Summary: Complete Native D-Avatar on Dense-Armor Primitives

| D-Avatar Component | Dense-Armor Primitive | Status |
|---|---|---|
| Camera frames | `CameraStream` → `camera_stream_frames()` | ✅ Done |
| Landmark shielding (occlusion, outliers, jitter) | `BatchedLandmarkShield` → `FaceShield` wrapper | ✅ Done (contributed to Dense-Armor) |
| Rate limiting | `RateBound` (D-Avatar own) | ✅ Kept (no DA equivalent) |
| Drift detection / recalibration | `ADWIN` (×2: EAR, iris) | ✅ Done |
| Stream alignment | `Sync` | ✅ Available |
| Latency budget | `RealtimePipeline` + `profile` | ✅ Done |
| MCP control | Custom `d_avatar.mcp` tools | ✅ Done |
| Parameter extraction | Geometric (no ML) | ✅ Done |
| Gaze from iris | `iris_displacement` + `GazeCalibrator` | ✅ Done |
| 2D/3D avatar drawing | `draw_avatar()` pure function (Step D) | 🔄 In progress |
| VMC sending | `VmcSender.send_frame()` (Step E) | 🔄 In progress |

**All tests pass:** D-Avatar 76, Dense-Armor vision 205, BatchedLandmarkShield 15, `check_estimator` passes.
**All checks pass:** `mkdocs build --strict`, `pytest -q`, `ruff check .`, `ruff format --check .`, `mypy d_avatar --follow-imports=skip`.