"""
D-Avatar app: your face in a window, a Start and a Stop button.

Start opens the camera, finds the face, runs the D-Avatar pipeline and sends the parameters
over VMC to the host and port written in the window. The MediaPipe model file is downloaded
on the first Start. Needs the cam, mp and osc extras.

    python examples/app.py
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import ttk
from typing import Any

import cv2
import numpy as np

from d_avatar import AvatarPipeline, FaceDetector, camera_stream_frames

MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/'
    'float16/1/face_landmarker.task'
)
MODEL = Path(__file__).resolve().parents[1] / 'face_landmarker.task'
SHOWN = ('eye_l', 'eye_r', 'mouth_open', 'brow_l', 'brow_r', 'head_yaw', 'head_pitch', 'head_roll')
PAD = 8


class App:
    """The window and its two buttons."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title('D-Avatar')
        self.cap: Any = None
        self.det: FaceDetector | None = None
        self.ap: AvatarPipeline | None = None
        self.running = False
        self.t0 = 0.0
        self.frames = 0
        self.face_frames = 0
        self.photo: tk.PhotoImage | None = None

        self.video = tk.Label(root, width=80, height=30, bg='black')
        self.video.grid(row=0, column=0, rowspan=8, padx=PAD, pady=PAD)

        ttk.Label(root, text='Camera').grid(row=0, column=1, sticky='w', padx=PAD)
        self.index = ttk.Combobox(root, values=[0, 1, 2, 3], width=8, state='readonly')
        self.index.set(0)
        self.index.grid(row=1, column=1, padx=PAD, sticky='w')
        ttk.Label(root, text='Invia a (host)').grid(row=2, column=1, sticky='w', padx=PAD)
        self.host = ttk.Entry(root, width=16)
        self.host.insert(0, '127.0.0.1')
        self.host.grid(row=3, column=1, padx=PAD, sticky='w')
        ttk.Label(root, text='Porta VMC').grid(row=4, column=1, sticky='w', padx=PAD)
        self.port = ttk.Entry(root, width=16)
        self.port.insert(0, '39539')
        self.port.grid(row=5, column=1, padx=PAD, sticky='w')

        self.start_button = ttk.Button(root, text='Avvia', command=self.start)
        self.start_button.grid(row=6, column=1, padx=PAD, pady=PAD, sticky='ew')
        self.stop_button = ttk.Button(root, text='Ferma', command=self.stop, state='disabled')
        self.stop_button.grid(row=7, column=1, padx=PAD, sticky='ew')

        self.status = ttk.Label(root, text='Fermo', font=('Segoe UI', 11, 'bold'))
        self.status.grid(row=8, column=0, padx=PAD, sticky='w')
        self.values = ttk.Label(root, text='', font=('Consolas', 9), justify='left')
        self.values.grid(row=8, column=1, padx=PAD, sticky='w')
        root.protocol('WM_DELETE_WINDOW', self.close)

    def say(self, text: str) -> None:
        self.status.configure(text=text)
        self.root.update_idletasks()

    def start(self) -> None:
        if self.running:
            return
        self.start_button.configure(state='disabled')
        try:
            if not MODEL.exists():
                self.say('Scarico il modello (3,7 MB), una volta sola...')
                urllib.request.urlretrieve(MODEL_URL, MODEL)
            index = int(self.index.get())
            # Use Dense-Armor's CameraStream via camera_stream_frames
            self._cam_gen = camera_stream_frames(device=index, fps=30.0, gray=False)
            self.det = FaceDetector.open_default(str(MODEL))
            self.ap = AvatarPipeline(vmc_host=self.host.get(), vmc_port=int(self.port.get()))
        except Exception as e:
            self.say(f'Errore: {e}')
            self.start_button.configure(state='normal')
            return
        self.running = True
        self.t0 = time.perf_counter()
        self.frames = self.face_frames = 0
        self.stop_button.configure(state='normal')
        self.say('In esecuzione')
        self.tick()

    def tick(self) -> None:
        if not self.running or self.det is None or self.ap is None:
            return
        try:
            t, rgb = next(self._cam_gen)
        except StopIteration:
            self.say('Camera stream ended')
            self.stop()
            return
        except Exception as e:
            self.say(f'Camera error: {e}')
            self.root.after(50, self.tick)
            return

        # camera_stream_frames yields RGB float32 in [0, 1]; convert to BGR uint8 for display
        bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        frame = self.det.detect(np.ascontiguousarray(rgb), t)
        params = self.ap.step(frame)
        face = bool(frame['confidence'].any())
        self.frames += 1
        self.face_frames += int(face)
        view = bgr.copy()
        pts = frame['points']
        for x, y in pts[np.all(np.isfinite(pts), axis=1)].astype(int):
            cv2.circle(view, (int(x), int(y)), 2, (0, 255, 0), -1)
        self.photo = tk.PhotoImage(data=cv2.imencode('.ppm', cv2.flip(view, 1))[1].tobytes())
        self.video.configure(image=self.photo, width=0, height=0)
        self.say('Volto trovato: sto inviando' if face else 'Nessun volto')
        self.values.configure(text='\n'.join(f'{k:11s}{params[k]:7.2f}' for k in SHOWN))
        self.root.after(1, self.tick)

    def stop(self) -> None:
        self.running = False
        # camera_stream_frames handles camera release internally
        if self.ap is not None and self.ap.vmc is not None:
            self.ap.vmc.close()
        self.ap = None
        self.video.configure(image='', width=80, height=30)
        self.photo = None
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.say('Fermo')

    def close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', type=float, default=0.0, help='start, run N seconds, quit')
    args = parser.parse_args()
    root = tk.Tk()
    app = App(root)
    if args.selftest > 0:
        root.after(500, app.start)

        def finish() -> None:
            print(f'selftest: frames={app.frames} face_frames={app.face_frames}')
            print(f'status: {app.status.cget("text")}')
            app.close()

        root.after(int(args.selftest * 1000), finish)
    root.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
