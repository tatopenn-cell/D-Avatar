#!/usr/bin/env python
"""Live webcam example with screenshot button.

Runs CameraSource, FaceDetector and AvatarPipeline with a VMC sender
until 'q' is pressed or the window is closed. Needs a camera and mediapipe model file.

    python examples/webcam.py --host 127.0.0.1 --port 39539 --index 0
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
import tkinter as tk
from pathlib import Path

import cv2

from d_avatar import AvatarPipeline, CameraSource, FaceDetector


class WebcamApp:
    def __init__(
        self,
        root: tk.Tk,
        pipeline: AvatarPipeline,
        detector: FaceDetector,
        camera: CameraSource,
    ) -> None:
        self.root = root
        self.pipeline = pipeline
        self.detector = detector
        self.camera = camera
        self.running = True

        root.title('D-Avatar Live Input')
        root.geometry('640x520')

        self.label = tk.Label(root)
        self.label.pack()

        self.status = tk.Label(root, text='Ready', anchor='w')
        self.status.pack(fill='x')

        self.btn = tk.Button(
            root,
            text='Screenshot schermo intero',
            command=self._take_screenshot,
        )
        self.btn.pack(side='bottom', pady=5)

        root.protocol('WM_DELETE_WINDOW', self.close)

    def close(self) -> None:
        self.running = False
        self.root.destroy()

    def _take_screenshot(self) -> None:
        """Capture full screen and save as PNG."""
        ok, frame = self.camera.read()
        if not ok:
            self.status.config(text='Errore: nessun frame dalla camera')
            self.root.after(2000, lambda: self.status.config(text='Ready'))
            return

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path('docs/assets/screenshots')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'screenshot_{ts}.png'

        ok_write = cv2.imwrite(str(out_path), frame)
        if ok_write:
            self.status.config(text=f'Screenshot salvato: {out_path}')
            self.root.after(3000, lambda: self.status.config(text='Ready'))
        else:
            self.status.config(text='Errore salvataggio screenshot')
            self.root.after(2000, lambda: self.status.config(text='Ready'))

    def loop(self) -> None:
        if not self.running:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.root.after(30, self.loop)
            return

        rgb = frame[:, :, ::-1]
        det_frame = self.detector.detect(rgb, time.time())
        self.pipeline.step(det_frame)

        h, w = frame.shape[:2]
        max_w = 640
        max_h = 480
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        disp = cv2.resize(frame, (new_w, new_h))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        img = tk.PhotoImage(data=cv2.imencode('.ppm', disp)[1].tobytes())
        self.label.img = img  # keep reference
        self.label.config(image=img)

        self.root.after(30, self.loop)


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

    root = tk.Tk()
    app = WebcamApp(root, ap, det, cam)

    root.after(30, app.loop)
    root.mainloop()

    cam.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
