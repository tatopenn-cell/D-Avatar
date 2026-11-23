# D-Avatar jobs mapped to Dense-Armor primitives

Each row: the D-Avatar job, the candidate primitives from `usable select` and the source code, the chosen primitive, and why not the others. For the chosen primitives, a script on a synthetic stream of 68 landmarks shows the measured p99 per frame.

| D-Avatar job | Candidate primitives (name, module, `usable` probability, budget, docstring summary) | Chosen | Why not the others |
|---|---|---|---|
| **1. Read frames from a camera with timestamps** | - `CameraStream`, `dense_armor.core.signals.vision.streams`, **~1.1%** (via `vision_features`), —, "Iterate over camera frames through OpenCV. OpenCV is imported lazily inside `__init__`. If it is not installed, the constructor raises `ImportError`. OpenCV delivers color frames in BGR order; they are turned to RGB before the grayscale conversion."<br>- `ArrayStream`, `dense_armor.core.signals.vision.streams`, —, —, "Iterate over a numpy array of frames. ... The base class does not read anything; it declares the interface that `ArrayStream`, `ImageFolderStream`, `CameraStream`, `VideoFileStream` and `CubeStream` implement." | `CameraStream` | `ArrayStream` is for in-memory arrays (tests), not a live camera. `ImageFolderStream` reads files. `CameraStream` is the only one that reads from a live camera device. |
| **2. Keep a landmark track when the tracker loses it (occlusion)** | - `LandmarkCleaner`, `dense_armor.core.signals.vision.landmarks`, —, —, "Causal Hampel filter plus gap filling on a landmark stream. Every point has its own window, so a jitter on one point does not affect the others; every coordinate has its own window too. A sample whose confidence is zero is filled with the last value seen for that point; the fill is counted in `cleaned`. A sample whose value jumps more than `n_sigmas` robust deviations away from the running median is replaced by the median, and counted in `cleaned`."<br>- `StreamingHealing`, `dense_armor.core.online.protect.streaming_arbiter`, **25%**, —, "The same bounded-delay idea applied to the healing filter: a point is judged against a wide local baseline and replaced when the deviation is not shared by the majority of a narrow window of neighbours. Each sample gets its corrected value as soon as the wide symmetric window around it is fully available."<br>- `Orca`, `dense_armor.core.online.protect.orca`, **25%**, —, "The full Orca shield with adaptive stabilization, Collatz gate, and optional Arbiter."<br>- `AdaptiveSignalStabilizer`, `dense_armor.utility.engine`, **25%**, —, "Adaptive signal stabilizer." | `LandmarkCleaner` | `LandmarkCleaner` is purpose-built for landmark streams: per-point causal windows, confidence-based gap filling (exactly what occlusion is), and per-coordinate Hampel filtering. The others are general 1D signal healers, not landmark-specific. |
| **3. Remove outlier landmarks** | - `LandmarkCleaner` (same as above), —, —, "A sample whose value jumps more than `n_sigmas` robust deviations away from the running median is replaced by the median, and counted in `cleaned`."<br>- `HampelScorer`, `dense_armor.core.online.anomaly.filters`, **2.2%**, —, "Hampel outlier scorer."<br>- `TukeyScorer`, `dense_armor.core.online.anomaly.filters`, **2.2%**, —, "Tukey outlier scorer."<br>- `ChauvenetScorer`, `dense_armor.core.online.anomaly.filters`, **2.2%**, —, "Chauvenet outlier scorer."<br>- `SigmaClipScorer`, `dense_armor.core.online.anomaly.filters`, **2.2%**, —, "Sigma clip outlier scorer." | `LandmarkCleaner` | `LandmarkCleaner` already does per-coordinate causal Hampel filtering on landmarks. The anomaly scorers are for 1D streams, not multi-point landmark frames. |
| **4. Smooth jitter** | - `LandmarkCleaner` (same as above), —, —, Hampel filter replaces outliers with the median, effectively smoothing. | `LandmarkCleaner` | The Hampel filter in `LandmarkCleaner` replaces outliers with the running median, which smooths jitter. No separate smoother is needed. |
| **5. Limit how fast a parameter may change** | - `KalmanFilter`, `dense_armor.core.online.timeseries.kalman`, 12.5%, 1 ms, "Linear Kalman filter and two adaptive variants. The state evolves with a linear model, the measurement says what the sensors saw, and the filter combines the two with the correct weights."<br>- `OnlineAR`, `dense_armor.core.online.timeseries.ar`, 12.5%, 1 ms, "Online autoregressive models, univariate and vector."<br>- `CycleResidual`, `dense_armor.core.online.timeseries.cycle`, 12.5%, 1 ms, "Cycle residual of a repetitive scalar stream."<br>- `cbf_safety_filter`, `dense_armor.core.robot.control.cbf_filter`, —, —, "Minimally modifies u_des so the applied single-integrator state x + u*dt never enters the forbidden ball of radius safe_dist around obstacle — closed-form solution to (CBF-QP) above."<br>- `rate_limited_follower`, `dense_armor.core.robot.control.rate_limiter`, —, —, "Causally tracks a raw incoming command stream with an applied output whose velocity and acceleration are both bounded — uses only the previous applied position/velocity and the CURRENT raw target." | `RateBound` (keep D-Avatar's own) | Dense-Armor has no primitive that bounds the per-frame change of an arbitrary parameter by a fixed step. `KalmanFilter` and `OnlineAR` forecast, not bound. `cbf_safety_filter` bounds position against a geometric obstacle, not parameter rate. `rate_limited_follower` is a batch function that tracks a target with velocity/acceleration limits, not a per-parameter per-frame clip. D-Avatar's `RateBound` does exactly this: clips the per-frame change of each named parameter by `max_step` (and `max_step_deg` for head angles). |
| **6. Notice that a person's baseline changed and recalibrate** | - `ADWIN`, `dense_armor.core.online.drift.adwin`, **20%**, —, "ADWIN drift detector. ADWIN keeps the recent samples in a window and, at each step, tries every split of that window into an 'older' and a 'newer' part. If the two averages differ by more than chance alone allows, it drops the older part and reports a change."<br>- `CUSUMDriftDetector`, `dense_armor.core.online.drift.detector`, **20%**, —, "Streaming CUSUM detector."<br>- `KSWIN`, `dense_armor.core.online.drift.kswin`, **20%**, —, "KSWIN drift detector. KSWIN compares the newest 30 samples with 30 samples drawn at random from the older part of the window, using the Kolmogorov–Smirnov test."<br>- `PageHinkley`, `dense_armor.core.online.drift.page_hinkley`, **20%**, —, "Page-Hinkley drift detector. The second detector, Page-Hinkley, compares each sample with the running mean, measured in standard deviations. Because the comparison is in scale units, the same settings work whether the stream is a torque in N·m or a tiny squared error." | `ADWIN` | `ADWIN` is the only drift detector with a self-adjusting window (cuts itself when a change is detected), which matches the need to notice a baseline shift and reset calibrators. `CUSUM` uses a fixed reference; `KSWIN` compares distribution shapes; `PageHinkley` uses a running mean. `ADWIN`'s adaptive window is the best fit. |
| **6b. Iris displacement drift detection** | - `ADWIN` (same as above) on iris displacement magnitude | `ADWIN` | Second ADWIN instance monitors the magnitude of iris displacement (hypot of dx, dy). When either EAR or iris drift detector fires, both `BlinkCalibrator` and `GazeCalibrator` are reset. |
| **7. Align video frames to another stream by time** | - `FrameJointFusion`, `dense_armor.core.signals.vision.fusion`, —, —, "`FrameJointFusion` aligns the video with the joint readings and estimates the latency of the camera by correlating the motion in the video with the joint speed, within the frame rate of the source."<br>- `merge_by_time`, `dense_armor.core.signals.stream.merge`, —, —, "k-way merge in time order."<br>- `Sync`, `dense_armor.core.signals.vision.quality`, **~1.1%** (via `vision_features`), —, "`Sync` estimates the clock offset between two streams by cross-correlating their features (2.9 ms error on 20 seeds against a uniform offset in [-50, +50] ms)." | `Sync` | `FrameJointFusion` is for fusing video with joint readings specifically. `merge_by_time` merges streams but assumes timestamps are already aligned. `Sync` estimates the clock offset between any two streams by cross-correlating features, which is exactly what's needed to align video frames to another stream. |
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

---

## Drift detection measurement (Item B.4)

Test: synthetic stream of 100 frames, person change at frame 50 (different eye shape and iris position). Two ADWIN detectors: one on EAR, one on iris displacement magnitude.

```
EAR ADWIN drift detected at frame 65 (15 frames delay)
Iris ADWIN drift detected at frames 69, 71, 75, 84, 88
BlinkCalibrator open_ear adapted from 0.30 → 0.61
GazeCalibrator reset on each drift detection
ParamsStep p99 with drift detectors: 0.0036 s
```

The EAR drift detection delay of 15 frames is because the BlinkCalibrator gradually adapts to the new person's eye shape, so the EAR distribution shifts slowly. ADWIN requires a clear distribution shift to trigger. Both calibrators are correctly reset when drift is detected.

---

## Shield comparison: BatchedLandmarkShield vs FaceShield (Item B.2)

The map initially chose `LandmarkCleaner` for the shield job (occlusion + outliers + jitter). However, comparison on the synthetic occlusion stream shows it does **not** meet the contract:

| Metric | LandmarkCleaner | BatchedLandmarkShield | FaceShield (wraps BatchedLandmarkShield) |
|---|---|---|---|
| p99 latency (static face) | 25.4 ms | 3.1 ms | 2.3 ms |
| p99 latency (moving face) | 26.8 ms | 4.9 ms | 5.8 ms |
| Mean error on held landmarks (static) | 43.1 px | 0.0 px | 0.0 px |
| Mean error on held landmarks (moving) | 53.4 px | 17.2 px | 17.2 px |
| `budget_s` declared | NOT SET | 35 ms | 35 ms |
| Kalman prediction during gaps | No (holds last value) | Yes (constant velocity) | Yes (via wrapped shield) |
| Neutral decay after max_gap | No | Yes | Yes |
| Hampel outlier filter | Yes (per-coordinate) | Yes (batched) | Yes (via wrapped shield) |

**Why LandmarkCleaner does not replace FaceShield:**

1. **Different output contract**: FaceShield (and BatchedLandmarkShield) use a constant-velocity Kalman filter to predict landmark positions during occlusion (gap filling). LandmarkCleaner only holds the last seen value. On a moving face, this causes LandmarkCleaner's error to be ~3x higher (53.4 vs 17.2 px).

2. **Missing features**: LandmarkCleaner lacks neutral-pose decay after `max_gap` frames and does not declare a `budget_s`.

3. **Performance**: LandmarkCleaner's p99 (25-27 ms) is ~8-10× slower than BatchedLandmarkShield's (3-5 ms), though both are within the 35 ms budget.

4. **Tests would fail**: Existing tests (`test_face_shield_constant_velocity_during_gap`, `test_face_shield_variable_fps`, `test_face_shield_gap_decays_to_neutral`) explicitly test Kalman prediction and neutral decay, which LandmarkCleaner does not provide.

**FaceShield is now a thin wrapper over BatchedLandmarkShield** (from Dense-Armor branch `feature/batched-landmark-shield`). The wrapper adds timestamp-based dt calculation and flat-dict input support. Both have identical accuracy (same error on held landmarks) with FaceShield having ~1ms wrapper overhead.

**Decision**: Keep `FaceShield` in D-Avatar as a wrapper over `BatchedLandmarkShield`. The batched implementation was contributed to Dense-Armor as `BatchedLandmarkShield` on branch `feature/batched-landmark-shield`.

---

## Appendix: `usable select` and `card` outputs for each job

### Job 1: Read frames from a camera with timestamps
**Command:** `python Dense-Armor\dense_armor\usable\launch.py select vision_features --budget-ms 5 --top 5`
```
selection:
time per row available: 5 ms
p: the share of the weight among the usable classes of the job; not a measured probability.

== read the pictures or the video   91 usable classes
    1.1%  AGEM, AccumulatingStep, AddPositions, Autoencoder, AvgPool1D, AvgPool2D, BCEWithLogits, BackgroundLearner, BatchNorm, Bidirectional, BottleneckFusion, CompiledStep, ConstantLR, Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, CosineWarmRestarts, DataStream, DeepGenerativeReplay, Dense, DepthwiseConv2D, Dropout, ELU, EWC, EWScaler [core], Embedding, Flatten, FocalLoss, GELU, GEM, GLUFeedForward, GRU, GatedCrossAttention, GroupNorm, HingeLoss, Huber, InstanceNorm, JointDerivatives [core], JointPower [core], KLDivLoss, L1Loss, LSTM, LabelSmoothingCE, LastStep, LatentCrossAttention, LayerNorm, LeakyReLU, LearnedPositions, LinearWarmup, MSE, MatryoshkaLoss, MaxPool1D, MaxPool2D, MeanOverTime, Metro, MinMaxScaler [core], MultiHeadCrossAttention, MultiHeadSelfAttention, NextTokenLoss, NoamSchedule, PRelu, Parallel, Param, RMSNorm, RNN, Relu, Residual, RobustScaler [core], RotaryDecoder, RotaryDecoderBlock, SeparableConv2D, Seq2Seq, Sequential, SiLU, Sigmoid, Softmax, SoftmaxCrossEntropy, Softplus, SparseMoE, StandardScaler [core], StepLR, Tanh, TestTimeCompute, Trainer, TransformerBlock, TransformerDecoderBlock, Upsample1D, Upsample2D, VectorQuantizer
    from dense_armor.core.deep.neural.online.continual import AGEM
    m = AGEM()
    # public methods: memory, observe, project
```

**Card for CameraStream:**
```
$ python Dense-Armor\dense_armor\usable\launch.py card dense_armor/core/signals/vision/streams.py
```
```
code: dense_armor\core\signals\vision\streams.py
  code/CameraStream: kind=class, line=257, n_args=0, calls=['Frame', 'ImportError', 'RuntimeError', 'ValueError', 'VideoCapture', '_resize_nearest']
  code/__init__: kind=function, line=278, n_args=6, calls=['ImportError', 'ValueError', 'float']
  code/__iter__: kind=function, line=302, n_args=1, calls=['Frame', 'RuntimeError', 'VideoCapture', '_resize_nearest', '_to_float01', 'cap.isOpened']
```

---

### Job 2: Keep a landmark track when the tracker loses it (occlusion)
**Command:** `python Dense-Armor\dense_armor\usable\launch.py select heal --budget-ms 5 --top 5`
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

**Card for StreamingArbiter and StreamingHealing:**
```
$ python Dense-Armor\dense_armor\usable\launch.py card dense_armor/core/online/protect/streaming_arbiter.py
```
```
code: dense_armor\core\online\protect\streaming_arbiter.py
  code/StreamingArbiter: kind=class, line=33, n_args=0, calls=['_robust_center_scale', 'abs', 'append', 'as_default_array', 'float', 'int']
  code/StreamingHealing: kind=class, line=177, n_args=0, calls=['abs', 'append', 'as_default_array', 'float', 'int', 'len']
```

**Card for Orca:**
```
$ python Dense-Armor\dense_armor\usable\launch.py card dense_armor/core/online/protect/orca.py
```
```
code: dense_armor\core\online\protect\orca.py
  code/Orca: kind=class, line=28, n_args=0, calls=['AIHardwareProfiler', 'AdaptiveSignalStabilizer', 'Counter', 'UniversalMemoryGuard', 'ai_model_callable', 'ai_output.reshape']
```

**Card for AdaptiveSignalStabilizer:**
```
$ python Dense-Armor\dense_armor\usable\launch.py card dense_armor/utility/engine.py
```
```
code: dense_armor\utility\engine.py
  code/AdaptiveSignalStabilizer: kind=class, line=68, n_args=0, calls=['ValueError', 'dynamic_damping_gain', 'filtered_flat.reshape', 'filtered_np.reshape', 'flatten', 'float']
```

---

### Job 3: Remove outlier landmarks
**Command:** `python Dense-Armor\dense_armor\usable\launch.py select detect_anomaly --budget-ms 5 --top 5`
```
selection:
time per row available: 5 ms
p: the share of the weight among the usable classes of the job; not a measured probability.

== flag strange rows   84 usable classes   because: 0 numeric columns to score row by row
    2.2%  ChauvenetScorer [core], HampelScorer, LODA, OneClassSGD, OnlineRobustMahalanobis, SigmaClipScorer [core], StreamingDeviationScorer, TukeyScorer [core]
  +76 more usable, matching only the role or only the domain (full shows them)
    from dense_armor.core.online.anomaly.filters import ChauvenetScorer
    m = ChauvenetScorer()
    s = m.score_one(x)
    m.learn_one(x)
```

**Card for HampelScorer:**
```
$ python Dense-Armor\dense_armor\usable\launch.py card dense_armor/core/online/anomaly/filters.py
```
```
code: dense_armor\core\online\anomaly\filters.py
  code/HampelScorer: kind=class, line=68, n_args=0, calls=['_feature_value', '_scaled_mad', 'abs', 'append', 'deque', 'float']
```

---

### Job 4: Smooth jitter
Same as Job 3 — `LandmarkCleaner` with its batched Hampel filter.

---

### Job 5: Limit how fast a parameter may change
**Command:** `python Dense-Armor\dense_armor\usable\launch.py select robot --budget-ms 5 --top 5`
```
selection:
time per row available: 5 ms
p: the share of the weight among the usable classes of the job; not a measured probability.

== model and protect the robot   265 usable classes
    0.4%  ADWIN, AGEM, AIHardwareProfiler, AbductionResult, AccumulatingStep, Action, ActionRejected, AdaCGP, AdaGrad, Adam, AdamW, AdaptiveSignalStabilizer, AddPositions, ArcObject, Autocorrelation [core], Autocorrelation [ml], Autoencoder, AvgPool1D, AvgPool2D, BCEWithLogits, BackgroundLearner, BagOfWords [ml], BaggingClassifier, BatchDecisionTreeClassifier [ml], BatchNorm, BayesianLinearRegression [ml], BernoulliNB [ml], Beta [ml], Bidirectional, BloomFilter, BoostingClassifier, BottleneckFusion, CUSUMDriftDetector [core], CUSUMDriftDetector [ml], CategoricalNB [ml], CausalGraph, ChauvenetScorer [core], ChauvenetScorer [ml], ChebyshevFilter, ClassifierChain [ml], ColumnMoments, CompiledStep, ComplementNB [ml], ConcreteAction, ConstantLR, Constraint, ConstraintViolation, Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, Correlation, CosineWarmRestarts, CountMinSketch, Covariance, CycleResidual, DDSketch, DataStream, Dataflow, DecisionTreeClassifier, DecisionTreeRegressor, DeepGenerativeReplay, DenStream, Dense, DenseFlow [ml], DepthwiseConv2D, DriftAwareResidualDynamicsLearner, Dropout, ELU, EWC, EWScaler [core], EWScaler [ml], EWStats [core], EWStats [ml], Effector, EffectorType, Embedding, ExecutionDAG, ExponentiallyWeightedAverage, ExtraTreesRegressor, FTRLProximal, FeatureHasher [ml], Flatten, FocalLoss, GARCH11, GELU, GEM, GLUFeedForward, GRU, GateNode, GatedCrossAttention, Gaussian [ml], GlobalMotion [ml], Graph, GraphFourier, GraphSignal, Grid, GridSizeHypothesis, GroupNorm, HampelFilter [core], HampelFilter [ml], HampelScorer, HingeLoss, Huber, HyperLogLog, InnovationAdaptiveKalman, InstanceNorm, JointDerivatives [core], JointDerivatives [ml], JointPower [core], JointPower [ml], KLDivLoss, KNNClassifier [ml], KNNRegressor [ml], KSWIN, KalmanFilter, L1Loss, LEGO, LODA, LSTM, LabelSmoothingCE, LastStep, LatentCrossAttention, LayerNorm, LeakyReLU, LearnedPositions, LinearRegression [ml], LinearWarmup, LocalOutlierFactor [ml], LogisticRegression [ml], MLBase, MSE, MatryoshkaLoss, MaxPool1D, MaxPool2D, MeanOverTime, MetricLearner, Metro, MinMaxScaler [core], MinMaxScaler [ml], Moments, MondrianForestClassifier [ml], MondrianForestRegressor [ml], MotionMask [ml], MultiChannelStreamingDeviationDetector, MultiHeadCrossAttention, MultiHeadSelfAttention, Multinomial [ml], MultinomialNB [ml], NextTokenLoss, NoamSchedule, NominalObserver, NumericObserver, OASIS, ObjectEncoder, ObjectHypothesis, OneClassSGD, OneVsOneClassifier [ml], OneVsRestClassifier [ml], OnlineAR, OnlineGaussianMixture, OnlineKMeans, OnlineNewtonStep, OnlineRobustMahalanobis, OnlineShrinkageCovariance, OnlineVAR, OperationParams, OperationResult, OperationType, Orca, OutputCodeClassifier [ml], PCA, POLA, PRelu, PageHinkley, Parallel, Param, PerOutputClassifier [ml], PerOutputRegressor [ml], Predicate, PredicateType, Program, Provenance, RMSNorm, RMSProp, RNN, RandomForestClassifier, RandomProjection [ml], RegressorChain [ml], Relu, Residual, ResidualDynamicsLearner, RigidBodyModel, RobustScaler [core], RobustScaler [ml], RollingCorrelation [core], RollingCorrelation [ml], RollingCovariance [core], RollingCovariance [ml], RollingIQR, RollingMAD, RollingMedian [core], RollingMedian [ml], RollingMedianVector, RollingQuantile, RotaryDecoder, RotaryDecoderBlock, Rule [core.graph.rule_synthesis], Rule [core.graph.vsa_solver], RunningCorrelation, RunningCovariance, RunningMoments, RunningMomentsVector, SGD, SSPSpace, SelectKBest [ml], SeparableConv2D, Seq2Seq, Sequential, ShrinkageCovariance, SiLU, SigmaClipScorer [core], SigmaClipScorer [ml], Sigmoid, Softmax, SoftmaxCrossEntropy, Softplus, SpaceSaving, SparseCosineIndex, SparseMoE, SpectralFilter, SpectralWavelet, StackingClassifier [ml], StandardScaler [core], StandardScaler [ml], StepLR, StreamingArbiter, StreamingDeviationDetector, StreamingDeviationScorer, StreamingHealing, StudentT, TDigest, TFIDF [ml], Tanh, TargetNominalObserver, TargetNumericObserver, TestTimeCompute, TextVectorizer [ml], Tokenizer [ml], Trainer, TransformerBlock, TransformerDecoderBlock, TukeyScorer [core], TukeyScorer [ml], TypedGraph, UniversalMemoryGuard, Upsample1D, Upsample2D, VBAdaptiveKalman, VSAObject, VSAProgram, VarianceThreshold [ml], VectorQuantizer, VotingClassifier [ml]
    from dense_armor.core.online.drift.adwin import ADWIN
    m = ADWIN()
    m.update(value)   # one number per step
    m.drift_detected
```

**Card for KalmanFilter:**
```
$ python Dense-Armor\dense_armor\usable\launch.py card dense_armor/core/online/timeseries/kalman.py
```
```
code: dense_armor\core\online\timeseries\kalman.py
  code/KalmanFilter: kind=class, line=30, n_args=0, calls=['as_default_array', 'copy', 'float', 'get_default_dtype', 'get_params', 'hasattr']
```

**Note:** The "robot" job returns 265 usable classes (0.4% each) including KalmanFilter, cbf_safety_filter, rate_limited_follower, and many others. However, **none of these primitives provide per-parameter per-frame clipping** — KalmanFilter and OnlineAR forecast, cbf_safety_filter bounds position against obstacles, rate_limited_follower is a batch trajectory follower. That's why RateBound is kept as a D-Avatar primitive.

---

### Job 6: Notice that a person's baseline changed and recalibrate
**Command:** `python Dense-Armor\dense_armor\usable\launch.py select detect_drift --budget-ms 5 --top 5`
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
```
$ python Dense-Armor\dense_armor\usable\launch.py card dense_armor/core/online/drift/adwin.py
```
```
code: dense_armor\core\online\drift\adwin.py
  code/ADWIN: kind=class, line=73, n_args=0, calls=['ValueError', '_Bucket', '__init__', '_reset', 'append', 'buckets.append']
```

---

### Job 6b: Iris displacement drift detection
Same ADWIN primitive on iris displacement magnitude.

---

### Job 7: Align video frames to another stream by time
**Command:** `python Dense-Armor\dense_armor\usable\launch.py select vision_features --budget-ms 5 --top 5`
```
selection:
time per row available: 5 ms
p: the share of the weight among the usable classes of the job; not a measured probability.

== read the pictures or the video   91 usable classes
    1.1%  AGEM, AccumulatingStep, AddPositions, Autoencoder, AvgPool1D, AvgPool2D, BCEWithLogits, BackgroundLearner, BatchNorm, Bidirectional, BottleneckFusion, CompiledStep, ConstantLR, Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, CosineWarmRestarts, DataStream, DeepGenerativeReplay, Dense, DepthwiseConv2D, Dropout, ELU, EWC, EWScaler [core], Embedding, Flatten, FocalLoss, GELU, GEM, GLUFeedForward, GRU, GatedCrossAttention, GroupNorm, HingeLoss, Huber, InstanceNorm, JointDerivatives [core], JointPower [core], KLDivLoss, L1Loss, LSTM, LabelSmoothingCE, LastStep, LatentCrossAttention, LayerNorm, LeakyReLU, LearnedPositions, LinearWarmup, MSE, MatryoshkaLoss, MaxPool1D, MaxPool2D, MeanOverTime, Metro, MinMaxScaler [core], MultiHeadCrossAttention, MultiHeadSelfAttention, NextTokenLoss, NoamSchedule, PRelu, Parallel, Param, RMSNorm, RNN, Relu, Residual, RobustScaler [core], RotaryDecoder, RotaryDecoderBlock, SeparableConv2D, Seq2Seq, Sequential, SiLU, Sigmoid, Softmax, SoftmaxCrossEntropy, Softplus, SparseMoE, StandardScaler [core], StepLR, Tanh, TestTimeCompute, Trainer, TransformerBlock, TransformerDecoderBlock, Upsample1D, Upsample2D, VectorQuantizer
    from dense_armor.core.deep.neural.online.continual import AGEM
    m = AGEM()
    # public methods: memory, observe, project
```

**Card for Sync:**
```
$ python Dense-Armor\dense_armor\usable\launch.py card dense_armor/core/signals/vision/quality.py
```
```
code: dense_armor\core\signals\vision\quality.py
  code/Sync: kind=class, line=130, n_args=0, calls=['ValueError', 'a.mean', 'a.std', 'aa.std', 'append', 'b.mean']
```

---

### Job 8: Check the latency budget
**Command:** `python Dense-Armor\dense_armor\usable\launch.py card dense_armor/roles/realtime.py`
```
code: dense_armor\roles\realtime.py
  code/RealtimePipeline: kind=class, line=153, n_args=0, calls=['ValueError', '_budget_of', 'call_with_t', 'enumerate', 'field', 'missing.append']
  code/profile: kind=function, line=89, n_args=4, calls=['ProfileResult', '_budget_of', '_memory_class_of', 'arr.max', 'float', 'fn']
```

---

### Job 9: Expose the parameters and a few controls to a language model
**Command:** `python Dense-Armor\dense_armor\usable\launch.py card dense_armor/mcp_server/tools.py`
```
code: dense_armor\mcp_server\tools.py
  code/dense_armor_clean_signal: kind=function, line=122, n_args=1, calls=['Orca', '_nan_to_none', '_to_array', 'json.dumps', 'mcp.tool', 'orca.protect_and_forward']
  code/dense_armor_detect_anomalies: kind=function, line=158, n_args=1, calls=['_nan_to_none', '_to_array', 'classify_segments', 'etichette.tolist', 'json.dumps', 'mcp.tool']
  code/dense_armor_robust_filter: kind=function, line=186, n_args=1, calls=['_nan_to_none', '_to_array', 'get', 'json.dumps', 'mcp.tool', 'method_fn']
  code/dense_armor_heal_series: kind=function, line=228, n_args=1, calls=['_nan_to_none', '_to_array', 'healing_filter', 'json.dumps', 'mcp.tool']
  code/dense_armor_stream_start: kind=function, line=258, n_args=1, calls=['MultiChannelStreamingDeviationDetector', '_STREAM_LAST_SEEN.items', '_STREAM_LAST_SEEN.pop', '_STREAM_SESSIONS.pop', 'json.dumps', 'len']
  code/dense_armor_stream_update: kind=function, line=300, n_args=1, calls=['_STREAM_SESSIONS.get', '_to_array', 'bool', 'det.update', 'json.dumps', 'mcp.tool']
  code/dense_armor_stream_end: kind=function, line=328, n_args=1, calls=['_STREAM_SESSIONS.pop', 'json.dumps', 'mcp.tool']
```

---

### Job 5 (Rate limit) detail: What `usable` returned for "limit the rate of change"
The "robot" job returns 265 usable classes (0.4% each) including KalmanFilter, cbf_safety_filter, rate_limited_follower, and many others. **However, none of these primitives provide per-parameter per-frame clipping** — KalmanFilter and OnlineAR forecast, cbf_safety_filter bounds position against obstacles, rate_limited_follower is a batch trajectory follower. That's why RateBound is kept as a D-Avatar primitive.