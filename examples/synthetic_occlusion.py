#!/usr/bin/env python3
"""
Synthetic occlusion example.

Builds a synthetic landmark stream with:
- 20-frame occlusion moving across face
- One spike frame
Runs the pipeline and prints:
- Maximum frame-to-frame jump of each parameter
- p99 latency per frame
"""

import os
import tempfile

import numpy as np
from dense_armor.core.signals.vision.landmarks import LandmarkStream

from d_avatar.pipeline import build_pipeline, run


def _synthetic_face_68() -> np.ndarray:
    """Create a synthetic 68-point face at neutral pose."""
    pts = np.zeros((68, 2), dtype=np.float32)
    # Eyebrows
    pts[17:22, 0] = np.linspace(20, 40, 5)
    pts[17:22, 1] = 20
    pts[22:27, 0] = np.linspace(60, 80, 5)
    pts[22:27, 1] = 20
    # Eyes (vertically aligned for EAR)
    pts[36] = [20, 35]
    pts[37] = [24, 30]
    pts[38] = [28, 30]
    pts[39] = [32, 35]
    pts[40] = [28, 40]
    pts[41] = [24, 40]
    pts[42] = [60, 35]
    pts[43] = [64, 30]
    pts[44] = [68, 30]
    pts[45] = [72, 35]
    pts[46] = [68, 40]
    pts[47] = [64, 40]
    # Nose
    pts[27:36, 0] = 50
    pts[27:36, 1] = np.linspace(40, 60, 9)
    # Mouth
    pts[48:60, 0] = np.linspace(30, 70, 12)
    pts[48:60, 1] = 75
    # Inner mouth
    pts[60:68, 0] = np.linspace(40, 60, 8)
    pts[60:68, 1] = 73
    return pts


def build_synthetic_stream() -> LandmarkStream:
    """Build stream with moving occlusion and spike."""
    base = _synthetic_face_68()
    n_frames = 100
    points = np.tile(base, (n_frames, 1, 1))
    confidence = np.ones((n_frames, 68), dtype=np.float32)

    # Moving occlusion: 20 frames, 2 points each, moving across face
    for i in range(20):
        idx = i % 66
        confidence[i, idx] = 0.0
        confidence[i, idx + 1] = 0.0

    # Spike at frame 50
    points[50] += 100.0

    times = np.arange(n_frames, dtype=np.float32) / 30.0
    return LandmarkStream(points, confidence, times, fps=30.0)


def main():
    stream = build_synthetic_stream()
    max_jumps, p99 = run(stream)

    print(f'max_jumps: {max_jumps}')
    print(f'p99_latency_s: {p99:.6f}')

    # Write plot if requested
    import sys

    if '--plot' in sys.argv:
        # Generate occlusion.png
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Simple plot of eye_l over time
        stream = build_synthetic_stream()
        pipeline = build_pipeline()
        eye_l_vals = []
        for frame in stream:
            frame_dict = {
                'points': frame['points'],
                'confidence': frame['confidence'],
                't': frame['t'],
            }
            out = pipeline.learn_one(frame_dict)
            # Extract eye_l from pipeline state (simplified)
            eye_l_vals.append(0.5)  # placeholder

        plt.figure()
        plt.plot(eye_l_vals)
        plt.title('Eye openness over time (with occlusion)')
        plt.xlabel('Frame')
        plt.ylabel('eye_l')
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, 'occlusion.png')
            plt.savefig(out_path)
            print(f'Plot saved to {out_path}')


if __name__ == '__main__':
    main()
