"""
Live webcam example.

Reads a camera, finds the face with MediaPipe, runs the D-Avatar pipeline and sends the
parameters over VMC. A window shows the camera image with the detected landmarks and the
main parameters; press q or Esc in the window to quit. Needs the cam, mp and osc extras
and the MediaPipe model file face_landmarker.task.

    python examples/webcam.py --index 0 --host 127.0.0.1 --port 39539
    python examples/webcam.py --no-preview --frames 300
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from d_avatar import AvatarPipeline, CameraSource, FaceDetector

SHOWN = ('eye_l', 'eye_r', 'mouth_open', 'brow_l', 'head_yaw', 'head_pitch', 'head_roll')
STATUS_EVERY = 30
KEY_ESC = 27


def draw(image: np.ndarray, frame: dict, params: dict, face: bool) -> np.ndarray:
    """Copy of the image with the landmarks and the main parameters written on it."""
    view = image.copy()
    pts = frame['points']
    for x, y in pts[np.all(np.isfinite(pts), axis=1)].astype(int):
        cv2.circle(view, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.putText(view, 'face' if face else 'no face', (10, 20), 0, 0.6, (0, 255, 255), 2)
    for i, key in enumerate(SHOWN):
        cv2.putText(
            view, f'{key}: {params[key]:.2f}', (10, 45 + 18 * i), 0, 0.5, (255, 255, 255), 1
        )
    return view


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', help='VMC destination host')
    parser.add_argument('--port', type=int, default=39539, help='VMC destination port')
    parser.add_argument('--index', type=int, default=0, help='camera index')
    parser.add_argument('--dshow', action='store_true', help='open the camera with DirectShow')
    parser.add_argument('--model', default='face_landmarker.task', help='MediaPipe model file')
    parser.add_argument('--no-preview', action='store_true', help='do not open a window')
    parser.add_argument('--frames', type=int, default=0, help='stop after this many frames')
    args = parser.parse_args()

    cap = (
        cv2.VideoCapture(args.index, cv2.CAP_DSHOW) if args.dshow else cv2.VideoCapture(args.index)
    )
    if not cap.isOpened():
        print(f'Cannot open camera {args.index}', file=sys.stderr)
        return 1
    cam = CameraSource(cap)
    try:
        det = FaceDetector.open_default(args.model)
    except Exception as e:
        print(f'Cannot open the detector: {e}', file=sys.stderr)
        cam.close()
        return 1
    ap = AvatarPipeline(vmc_host=args.host, vmc_port=args.port)

    print(f'Sending VMC to {args.host}:{args.port}. Press q in the window, or Ctrl+C, to quit.')
    n = 0
    try:
        for t, bgr in cam.frames():
            frame = det.detect(np.ascontiguousarray(bgr[:, :, ::-1]), t)
            params = ap.step(frame)
            face = bool(frame['confidence'].any())
            n += 1
            if n % STATUS_EVERY == 0:
                print(
                    f'frame {n}: face={"yes" if face else "no"}',
                    {k: round(params[k], 2) for k in SHOWN},
                )
            if not args.no_preview:
                cv2.imshow('D-Avatar', draw(bgr, frame, params, face))
                if cv2.waitKey(1) & 0xFF in (ord('q'), KEY_ESC):
                    break
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
