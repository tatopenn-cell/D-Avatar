"""Compare BatchedLandmarkShield vs FaceShield on synthetic occlusion stream."""

import time

import numpy as np
from dense_armor.core.signals.vision.landmarks import BatchedLandmarkShield, LandmarkStream

from d_avatar.shield import FaceShield
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


def run_shield(shield, stream):
    """Run shield on stream and return p99 latency and mean error on held landmarks."""
    latencies = []
    errors = []
    for i, frame in enumerate(stream):
        t0 = time.perf_counter()
        out = shield.transform_one(frame)
        latencies.append(time.perf_counter() - t0)
        if i in OCCLUSION:
            for p in range(36, 42):
                if frame['confidence'][p] == 0:
                    base = np.tile(synthetic_face_68(), (N_FRAMES, 1, 1))
                    err = np.linalg.norm(out['points'][p] - base[i, p])
                    errors.append(err)
    p99 = np.percentile(latencies, 99)
    mean_err = np.mean(errors) if errors else 0.0
    return p99, mean_err


def main():
    stream = build_synthetic_stream()

    # Test BatchedLandmarkShield (from Dense-Armor)
    bls = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=5)
    p99_bls, err_bls = run_shield(bls, stream)
    print(f'BatchedLandmarkShield: p99={p99_bls:.6f}s, mean_err_on_held={err_bls:.4f}')

    # Test FaceShield (D-Avatar wrapper)
    fs = FaceShield(window=7, n_sigmas=3.0, max_gap=5)
    p99_fs, err_fs = run_shield(fs, stream)
    print(f'FaceShield: p99={p99_fs:.6f}s, mean_err_on_held={err_fs:.4f}')

    # Moving face test
    print('\n--- Moving face test ---')
    base = synthetic_face_68()
    moving_points = np.zeros((N_FRAMES, 68, 2))
    for i in range(N_FRAMES):
        moving_points[i] = base + np.array([i * 0.5, i * 0.3])
    moving_conf = np.ones((N_FRAMES, 68))
    moving_conf[:, 36:42] = 0.0  # occlude left eye
    moving_times = np.arange(N_FRAMES) / 30.0
    moving_stream = LandmarkStream(moving_points, moving_conf, moving_times, fps=30.0)

    bls2 = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=5)
    fs2 = FaceShield(window=7, n_sigmas=3.0, max_gap=5)

    p99_bls_m, err_bls_m = run_shield(bls2, moving_stream)
    p99_fs_m, err_fs_m = run_shield(fs2, moving_stream)
    print(f'BatchedLandmarkShield (moving): p99={p99_bls_m:.6f}s, mean_err_on_held={err_bls_m:.4f}')
    print(f'FaceShield (moving): p99={p99_fs_m:.6f}s, mean_err_on_held={err_fs_m:.4f}')


if __name__ == '__main__':
    main()
