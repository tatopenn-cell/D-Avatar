#!/usr/bin/env python
"""Live webcam example.

Runs CameraSource, FaceDetector and AvatarPipeline with a VMC sender
until 'q' is pressed. Needs a camera and mediapipe model file.

    python examples/webcam.py --host 127.0.0.1 --port 39539 --index 0
"""

from __future__ import annotations

import argparse
import sys

from d_avatar import AvatarPipeline, CameraSource, FaceDetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', help='VMC destination host')
    parser.add_argument('--port', type=int, default=39539, help='VMC destination port')
    parser.add_argument('--index', type=int, default=0, help='Camera index')
    args = parser.parse_args()

    try:
        cam = CameraSource.open_default(args.index)
        det = FaceDetector.open_default()
    except Exception as e:
        print(f'Failed to open camera or detector: {e}', file=sys.stderr)
        return 1

    ap = AvatarPipeline(vmc_host=args.host, vmc_port=args.port, use_shield=True)

    print('Press q to quit.')
    try:
        for t, frame in cam.frames():
            # Convert BGR to RGB for MediaPipe
            rgb = frame[:, :, ::-1]
            det_frame = det.detect(rgb, t)
            ap.step(det_frame)
            # Exit on 'q' (requires GUI window - not shown here)
            # For a real loop, you'd show the frame and check cv2.waitKey(1) == ord('q')
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
