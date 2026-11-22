"""
Synthetic occlusion example.

Builds a landmark stream of a neutral face with a left-eye occlusion of 20 frames (the
detector returns (0, 0) with confidence 0) and one spike on an eyelid landmark, runs the
pipeline with and without the shield, and prints the largest frame-to-frame jump of each
parameter and the p99 latency per frame.

    python examples/synthetic_occlusion.py
    python examples/synthetic_occlusion.py --plot --out docs/assets/occlusion.png
"""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dense_armor.core.signals.vision.landmarks import LandmarkStream

from d_avatar.pipeline import AvatarPipeline, run
from d_avatar.synthetic import synthetic_face_68

N_FRAMES = 100
OCCLUSION = range(20, 40)
SPIKE_FRAME = 60


def build_synthetic_stream() -> LandmarkStream:
    """Neutral face, left eye occluded in OCCLUSION, one eyelid spike at SPIKE_FRAME."""
    points = np.tile(synthetic_face_68(), (N_FRAMES, 1, 1))
    confidence = np.ones((N_FRAMES, 68))
    for i in OCCLUSION:
        points[i, 36:42] = 0.0
        confidence[i, 36:42] = 0.0
    points[SPIKE_FRAME, 37, 1] += 40.0
    times = np.arange(N_FRAMES) / 30.0
    return LandmarkStream(points, confidence, times, fps=30.0)


def trace(stream: LandmarkStream, use_shield: bool) -> list[dict]:
    """Parameters of every frame."""
    ap = AvatarPipeline(use_shield=use_shield)
    return [ap.step(frame) for frame in stream]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--out', default='occlusion.png')
    args = parser.parse_args()
    stream = build_synthetic_stream()
    jumps, p99 = run(stream)
    print(f'max_jumps: {jumps}')
    print(f'p99_latency_s: {p99:.6f}')
    if args.plot:
        matplotlib.use('Agg')
        with_shield = [p['eye_l'] for p in trace(stream, True)]
        without = [p['eye_l'] for p in trace(stream, False)]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(without, label='without shield')
        ax.plot(with_shield, label='with shield')
        ax.axvspan(OCCLUSION.start, OCCLUSION.stop, alpha=0.15, label='occlusion')
        ax.axvline(SPIKE_FRAME, linestyle=':', label='spike')
        ax.set_xlabel('frame')
        ax.set_ylabel('eye_l')
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f'plot: {args.out}')


if __name__ == '__main__':
    main()
