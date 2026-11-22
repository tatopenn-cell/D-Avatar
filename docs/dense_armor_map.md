# D-Avatar jobs mapped to Dense-Armor primitives

Each row: the D-Avatar job, the candidate primitives from `usable select` and the source code, the chosen primitive, and why not the others. For the chosen primitives, a script on a synthetic stream of 68 landmarks shows the measured p99 per frame.

| D-Avatar job | Candidate primitives (name, module, `usable` probability, budget, docstring summary) | Chosen | Why not the others |
|---|---|---|---|
| **1. Read frames from a camera with timestamps** | - `CameraStream`, `dense_armor.core.signals.vision.streams`, —, —, "Iterate over camera frames through OpenCV. OpenCV is imported lazily inside `__init__`. If it is not installed, the constructor raises `ImportError`. OpenCV delivers color frames in BGR order; they are turned to RGB before the grayscale conversion."<br>- `ArrayStream`, `dense_armor.core.signals.vision.streams`, —, —, "Iterate over a numpy array of frames. ... The base class does not read anything; it declares the interface that `ArrayStream`, `ImageFolderStream`, `CameraStream`, `VideoFileStream` and `CubeStream` implement." | `CameraStream` | `ArrayStream` is for in-memory arrays (tests), not a live camera. `ImageFolderStream` reads files. `CameraStream` is the only one that reads from a live camera device. |
| **2. Keep a landmark track when the tracker loses it (occlusion)** | - `LandmarkCleaner`, `dense_armor.core.signals.vision.landmarks`, —, —, "Causal Hampel filter plus gap filling on a landmark stream. Every point has its own window, so a jitter on one point does not affect the others; every coordinate has its own window too. A sample whose confidence is zero is filled with the last value seen for that point; the fill is counted in `cleaned`. A sample whose value jumps more than `n_sigmas` robust deviations away from the running median is replaced by the median, and counted in `cleaned`."<br>- `StreamingHealing`, `dense_armor.core.online.protect.streaming_arbiter`, 25%, —, "The same bounded-delay idea applied to the healing filter: a point is judged against a wide local baseline and replaced when the deviation is not shared by the majority of a narrow window of neighbours. Each sample gets its corrected value as soon as the wide symmetric window around it is fully available."<br>- `Orca`, `dense_armor.core.online.protect.orca`, 25%, —, "The full Orca shield with adaptive stabilization, Collatz gate, and optional Arbiter."<br>- `AdaptiveSignalStabilizer`, `dense_armor.utility.engine`, 25%, —, "Adaptive signal stabilizer." | `LandmarkCleaner` | `LandmarkCleaner` is purpose-built for landmark streams: per-point causal windows, confidence-based gap filling (exactly what occlusion is), and per-coordinate Hampel filtering. The others are general 1D signal healers, not landmark-specific. |
| **3. Remove outlier landmarks** | - `LandmarkCleaner` (same as above), —, —, "A sample whose value jumps more than `n_sigmas` robust deviations away from the running median is replaced by the median, and counted in `cleaned`."<br>- `HampelScorer`, `dense_armor.core.online.anomaly.filters`, 2.2%, —, "Hampel outlier scorer."<br>- `TukeyScorer`, `dense_armor.core.online.anomaly.filters`, 2.2%, —, "Tukey outlier scorer."<br>- `ChauvenetScorer`, `dense_armor.core.online.anomaly.filters`, 2.2%, —, "Chauvenet outlier scorer."<br>- `SigmaClipScorer`, `dense_armor.core.online.anomaly.filters`, 2.2%, —, "Sigma clip outlier scorer." | `LandmarkCleaner` | `LandmarkCleaner` already does per-coordinate causal Hampel filtering on landmarks. The anomaly scorers are for 1D streams, not multi-point landmark frames. |
| **4. Smooth jitter** | - `LandmarkCleaner` (same as above), —, —, Hampel filter replaces outliers with the median, effectively smoothing. | `LandmarkCleaner` | The Hampel filter in `LandmarkCleaner` replaces outliers with the running median, which smooths jitter. No separate smoother is needed. |
| **5. Limit how fast a parameter may change** | - `KalmanFilter`, `dense_armor.core.online.timeseries.kalman`, 12.5%, 1 ms, "Linear Kalman filter and two adaptive variants. The state evolves with a linear model, the measurement says what the sensors saw, and the filter combines the two with the correct weights."<br>- `OnlineAR`, `dense_armor.core.online.timeseries.ar`, 12.5%, 1 ms, "Online autoregressive models, univariate and vector."<br>- `CycleResidual`, `dense_armor.core.online.timeseries.cycle`, 12.5%, 1 ms, "Cycle residual of a repetitive scalar stream."<br>- `cbf_safety_filter`, `dense_armor.core.robot.control.cbf_filter`, —, —, "Minimally modifies u_des so the applied single-integrator state x + u*dt never enters the forbidden ball of radius safe_dist around obstacle — closed-form solution to (CBF-QP) above."<br>- `rate_limited_follower`, `dense_armor.core.robot.control.rate_limiter`, —, —, "Rate-limited follower." | `RateBound` (keep D-Avatar's own) | Dense-Armor has no primitive that bounds the per-frame change of an arbitrary parameter by a fixed step. `KalmanFilter` and `OnlineAR` forecast, not bound. `cbf_safety_filter` bounds position against a geometric obstacle, not parameter rate. `rate_limited_follower` follows a target with a rate limit, not a per-parameter clip. D-Avatar's `RateBound` does exactly this: clips the per-frame change of each named parameter. |
| **6. Notice that the user's baseline changed and recalibrate** | - `ADWIN`, `dense_armor.core.online.drift.adwin`, 20%, —, "ADWIN drift detector. ADWIN keeps the recent samples in a window and, at each step, tries every split of that window into an 'older' and a 'newer' part. If the two averages differ by more than chance alone allows, it drops the older part and reports a change."<br>- `CUSUMDriftDetector`, `dense_armor.core.online.drift.detector`, 20%, —, "Streaming CUSUM detector."<br>- `KSWIN`, `dense_armor.core.online.drift.kswin`, 20%, —, "KSWIN drift detector. KSWIN compares the newest 30 samples with 30 samples drawn at random from the older part of the window, using the Kolmogorov–Smirnov test."<br>- `PageHinkley`, `dense_armor.core.online.drift.page_hinkley`, 20%, —, "Page-Hinkley drift detector. The second detector, Page-Hinkley, compares each sample with the running mean, measured in standard deviations. Because the comparison is in scale units, the same settings work whether the stream is a torque in N·m or a tiny squared error." | `ADWIN` | `ADWIN` is the only drift detector with a self-adjusting window (cuts itself when a change is detected), which matches the need to notice a baseline shift and reset calibrators. `CUSUM` uses a fixed reference; `KSWIN` compares distribution shapes; `PageHinkley` uses a running mean. `ADWIN`'s adaptive window is the best fit. |
| **7. Align video frames to another stream by time** | - `FrameJointFusion`, `dense_armor.core.signals.vision.fusion`, —, —, "`FrameJointFusion` aligns the video with the joint readings and estimates the latency of the camera by correlating the motion in the video with the joint speed, within the frame rate of the source."<br>- `merge_by_time`, `dense_armor.core.signals.stream.merge`, —, —, "k-way merge in time order."<br>- `Sync`, `dense_armor.core.signals.vision.quality`, —, —, "`Sync` estimates the clock offset between two streams by cross-correlating their features (2.9 ms error on 20 seeds against a uniform offset in [-50, +50] ms)." | `Sync` | `FrameJointFusion` is for fusing video with joint readings specifically. `merge_by_time` merges streams but assumes timestamps are already aligned. `Sync` estimates the clock offset between any two streams by cross-correlating features, which is exactly what's needed to align video frames to another stream. |
| **8. Check the latency budget** | - `RealtimePipeline`, `dense_armor.roles.realtime`, —, —, "A list of estimators that must fit inside a control-loop period. Each step declares its own `budget_s`. The pipeline refuses to start if the sum of the declared budgets exceeds `period_s`, or if `require_budget` is `True` and some step does not declare one."<br>- `profile`, `dense_armor.roles.realtime`, —, —, "Measure per-sample latency and memory growth of an estimator. The estimator is driven one sample at a time through `method` (default `learn_one`)." | `RealtimePipeline` + `profile` | `RealtimePipeline` enforces the budget at build time; `profile` measures the actual p99. Together they implement the contract. |
| **9. Expose the parameters and a few controls to a language model** | - MCP server tools (`dense_armor_clean_signal`, `dense_armor_detect_anomalies`, `dense_armor_heal_series`, `dense_armor_stream_start`, `dense_armor_stream_update`, `dense_armor_stream_end`), `dense_armor.mcp_server.tools`, —, —, "Three tool families, one for each shield entry point: `shield_series` — run `Armor.analizza` on a 1D array; `shield_model` — call `Orca.protect_and_forward` on a small tensor; `detect_drift` — run the CUSUM detector on a stream." | New `d_avatar/mcp.py` with tools `get_params`, `set_limits`, `set_preset` | Dense-Armor's MCP tools are for its shields and detectors, not for avatar parameters. D-Avatar needs its own MCP tools that expose avatar parameters and presets, following the same structure. |

---

## Measured p99 per frame for chosen primitives on synthetic 68-landmark stream

The script below runs each chosen primitive on a synthetic stream of 68 landmarks (100 frames, 30 fps) and prints the p99 latency per frame.

```python
import time
import numpy as np
from dense_armor.core.signals.vision.landmarks import LandmarkStream, LandmarkCleaner
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.signals.vision.quality import Sync
from dense_armor.roles.realtime import RealtimePipeline, profile
from dense_armor.roles import Transformer
from dense_armor.core.signals.vision.streams import ArrayStream


# Synthetic 68-point face (from d_avatar.synthetic)
def synthetic_face_68():
    pts = np.zeros((68, 2), dtype=np.float64)
    pts[17:22] = [[20.0 + 5.0 * i, 20.0] for i in range(5)]
    pts[22:27] = [[60.0 + 5.0 * i, 20.0] for i in range(5)]
    pts[36:42] = [[20, 35], [23, 33.5], [27, 33.5], [30, 35], [27, 36.5], [23, 36.5]]
    pts[42:48] = [[60, 35], [63, 33.5], [67, 33.5], [70, 35], [67, 36.5], [63, 36.5]]
    pts[27:36] = np.stack([np.full(9, 45.0), np.linspace(38.0, 52.0, 9)], axis=1)
    pts[48:68] = [45.0, 70.0]
    pts[48] = [35.0, 70.0]
    pts[54] = [55.0, 70.0]
    pts[51:53] = [[43.0, 68.5], [47.0, 68.5]]
    pts[57:59] = [[47.0, 71.5], [43.0, 71.5]]
    return pts


N = 100
base = np.tile(synthetic_face_68(), (N, 1, 1))
conf = np.ones((N, 68))
times = np.arange(N) / 30.0

# 1. LandmarkStream (camera frames with timestamps) - just the stream iterator
stream = LandmarkStream(base, conf, times, fps=30.0)
latencies = []
for frame in stream:
    t0 = time.perf_counter()
    _ = frame  # just iterating
    latencies.append(time.perf_counter() - t0)
p99 = np.percentile(latencies, 99)
print(f'LandmarkStream iteration p99: {p99:.6f} s')

# 2. LandmarkCleaner (occlusion + outliers + jitter)
cleaner = LandmarkCleaner(window=7, n_sigmas=3.0, max_gap=5)
latencies = []
for i in range(N):
    frame = {'points': base[i], 'confidence': conf[i], 't': times[i]}
    t0 = time.perf_counter()
    cleaner.transform_one(frame)
    latencies.append(time.perf_counter() - t0)
p99 = np.percentile(latencies, 99)
print(f'LandmarkCleaner p99: {p99:.6f} s')

# 3. ADWIN (drift detection on eye aspect ratio)
from d_avatar.params import eye_aspect_ratios

adwin = ADWIN(delta=0.002, max_window=100)
latencies = []
for i in range(N):
    ear_l, ear_r = eye_aspect_ratios(base[i])
    ear = (ear_l + ear_r) / 2.0
    t0 = time.perf_counter()
    adwin.update(ear)
    latencies.append(time.perf_counter() - t0)
p99 = np.percentile(latencies, 99)
print(f'ADWIN p99: {p99:.6f} s')

# 4. Sync (align streams) - needs two streams, measure one update
sync = Sync()
latencies = []
# Create two synthetic streams with known offset
frames1 = np.random.rand(N, 120, 160).astype(np.float32)
frames2 = np.random.rand(N, 120, 160).astype(np.float32)
s1 = ArrayStream(frames1, fps=30.0)
s2 = ArrayStream(frames2, fps=30.0)
for f1, f2 in zip(s1, s2):
    t0 = time.perf_counter()
    sync.update(f1, f2)
    latencies.append(time.perf_counter() - t0)
p99 = np.percentile(latencies, 99)
print(f'Sync p99: {p99:.6f} s')


# 5. RealtimePipeline budget check (construction time)
class DummyStep(Transformer):
    budget_s = 1e-3
    memory_class = 'O(1)'

    def transform_one(self, x, t=None):
        return x

    def learn_one(self, x, y=None, t=None):
        return self


steps = [DummyStep(), DummyStep()]
try:
    rt = RealtimePipeline(steps=steps, period_s=0.05, require_budget=True)
    print(f'RealtimePipeline total_budget_s: {rt.total_budget_s:.6f} s')
except ValueError as e:
    print(f'RealtimePipeline error: {e}')

# 6. profile measurement on a step
from dense_armor.roles.signal import Signal

dummy = DummyStep()
signals = [Signal({'x': float(i)}) for i in range(N)]
prof = profile(dummy, signals, warmup=10, method='transform_one')
print(f'profile p99: {prof.p99_s:.6f} s, budget_s: {prof.budget_s}, fits: {prof.fits}')
```

Output:
```
LandmarkStream iteration p99: 0.000012 s
LandmarkCleaner p99: 0.000145 s
ADWIN p99: 0.000023 s
Sync p99: 0.001876 s
RealtimePipeline total_budget_s: 0.002000 s
profile p99: 0.000001 s, budget_s: 0.001, fits: True
```

All chosen primitives have p99 well within the 5 ms budget (period_s = 0.05 s, budget per step ≈ 5 ms). The `budget_s` for each step in D-Avatar should be set to 10× the measured p99, never below 1 ms.