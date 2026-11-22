# D-Avatar

D-Avatar turns a stream of face landmarks into a few stable numbers that move an avatar, and sends them over the VMC protocol. It is built on [dense_armor](https://github.com/tatopenn-cell/Dense-Armor).

A face tracker gives you 68 landmarks per video frame. D-Avatar cleans them (a lost or jumping landmark does not make the avatar jump), turns them into parameters such as eye opening, mouth opening, eyebrows, head angles and gaze, limits how fast each parameter may change, and sends one VMC message per parameter and frame over UDP.

The parameters are `eye_l`, `eye_r`, `mouth_open`, `brow_l` and `brow_r` in the range 0 to 1, `gaze_x` and `gaze_y` in the range -1 to 1, `head_yaw`, `head_pitch` and `head_roll` in degrees, and `blink` (0 or 1). Gaze is computed only when the frame carries the iris centres.

**Task DA3 (Dense-Armor integration)**: D-Avatar is now built from Dense-Armor primitives:
- Camera input via `dense_armor.core.signals.vision.CameraStream`
- Drift detection via `dense_armor.core.online.drift.ADWIN` on EAR and iris displacement
- Latency budget via `dense_armor.roles.RealtimePipeline`
- MCP control via new `d_avatar.mcp` tools (`get_params`, `set_limits`, `set_preset`)
- The batched landmark shield (`FaceShield`) was contributed to Dense-Armor as `BatchedLandmarkShield` (branch `feature/batched-landmark-shield`)

## Install

Python 3.10 or newer. The core needs `dense_armor` and `numpy`; the extras add what each part uses.

```bash
pip install "d-avatar[all] @ git+https://github.com/tatopenn-cell/D-Avatar.git"
```

The extras are `osc` (`python-osc`, to send VMC), `cam` (`opencv-python`, to read a camera) and `mp` (`mediapipe`, the face detector). Install only the ones you need, for example `d-avatar[osc]`.

## Quick start

One frame in, parameters out. The face below is a synthetic neutral face that ships with the library.

```python
import numpy as np
from d_avatar import AvatarPipeline
from d_avatar.synthetic import synthetic_face_68

ap = AvatarPipeline()
frame = {'points': synthetic_face_68(), 'confidence': np.ones(68), 't': 0.0}
print(ap.step(frame))
```

A point with `confidence` 0 means the tracker lost it; the shield holds it on its predicted track and brings it back to a neutral position after a few frames. To send the parameters, give the pipeline a destination: `AvatarPipeline(vmc_host='127.0.0.1', vmc_port=39539)`.

## Live from a camera

This needs the `cam`, `mp` and `osc` extras and the MediaPipe model file, which is not part of the repository.

```bash
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python examples/webcam.py --index 0 --host 127.0.0.1 --port 39539
```

Run the second command from the folder that holds `face_landmarker.task`. `--index` is the camera number; with OBS Studio running, the physical camera may have another index. Stop the example with Ctrl+C.

## What it does not do

- The parameter scalings for eyebrows, mouth and head angles are fixed constants chosen on a synthetic face. They are not calibrated on real faces; on a real face the eyebrow values can stay at 0 and the head pitch can read high.
- The layout of the VMC messages is not checked against the protocol specification, and the parameter names are D-Avatar names, not the VRM blendshape presets. An avatar program may need a name mapping.
- The declared latency budgets add up to 41 ms without VMC and 45 ms with it, so the default frame period is 50 ms. The measured time of a whole frame is about 2 to 3 ms on a Kaggle CPU.
- There is no audio, no emotion learning and no control from a language model.

## Development

```bash
pip install -e ".[dev,osc,docs]"
pytest tests -q
ruff check . && ruff format --check .
mypy d_avatar --follow-imports=skip
mkdocs build --strict
```

The tests run without a camera and without `mediapipe`; the canonical face mesh they use is in `papers/canonical_face_model.obj`.

## Documentation

The guide, with the shield, the plots, the measured cost and the references, is in [docs/index.md](docs/index.md).
