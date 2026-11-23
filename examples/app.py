"""D-Avatar example app: live camera -> MediaPipe Face Landmarker -> avatar 2D + VMC.

Window layout:
- left: live camera preview (always on when the app is open)
- right: 2D avatar (draw_avatar), driven by the native pipeline
- bottom: status bar with live Dense-Armor class names + 7 parameter bars
- buttons: Start / Stop tracking (+ VMC), Quit

Run: python examples/app.py [--vmc-port 39539]

Default path uses MediaPipe Face Landmarker (face_landmarker.task already present)
+ Dense-Armor pipeline (FaceShield + ParamsStep + RateBound). No neural network
is built or trained by this code; the pre-trained task file is loaded only when
the detector is opened.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time

import numpy as np

try:
    import tkinter as tk

    from PIL import Image, ImageTk

    _TK_AVAILABLE = True
except ImportError:  # pragma: no cover - tkinter optional
    _TK_AVAILABLE = False

from d_avatar.avatar import draw_avatar
from d_avatar.capture import camera_stream_frames
from d_avatar.detector import FaceDetector
from d_avatar.pipeline import AvatarPipeline
from d_avatar.vmc import VmcSender

# Dense-Armor classes used in the pipeline.
_DENSE_ARMOR_CLASSES = (
    'FaceShield',
    'ParamsStep',
    'RateBound',
    'BlinkCalibrator',
    'ADWIN',
    'VmcSender',
)

_CAM_SIZE = (640, 480)
_AVATAR_SIZE = 400


class DAvatarApp:
    """Tkinter application: camera -> MediaPipe landmarker -> avatar + VMC.

    The camera preview runs as soon as the app opens. Clicking "Avvia
    tracking" starts the MediaPipe landmarker, the Dense-Armor pipeline and
    (optionally) VMC sending to VMagicMirror.
    """

    def __init__(self, root: tk.Tk, vmc_port: int = 39539) -> None:
        if not _TK_AVAILABLE:
            raise ImportError('tkinter and Pillow are required for the app')
        self.root = root
        self.root.title('D-Avatar — live avatar')
        self.vmc_port = vmc_port

        # Shared state between camera thread and UI
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_params: dict[str, float] = {}

        # Tracking state (started by "Avvia tracking")
        self._tracking = False
        self._track_thread: threading.Thread | None = None
        self._track_stop = threading.Event()
        self._detector: FaceDetector | None = None
        self._pipeline: AvatarPipeline | None = None
        self._vmc: VmcSender | None = None

        # Camera thread (always running while the app is open)
        self._cam_thread: threading.Thread | None = None
        self._cam_stop = threading.Event()

        self._build_ui()
        self._start_camera()
        self.root.after(100, self._refresh_ui)

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        # Video + avatar area
        area = tk.Frame(self.root)
        area.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.video_label = tk.Label(area, bg='black', text='camera (starting…)')
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.avatar_label = tk.Label(area, bg='#e0e0e0', text='avatar')
        self.avatar_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Status + parameter bars
        bar_frame = tk.Frame(self.root)
        bar_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Label(
            bar_frame,
            text='Dense-Armor classes: ' + ', '.join(_DENSE_ARMOR_CLASSES),
            anchor='w',
        ).pack(fill=tk.X)

        self._param_vars: dict[str, tk.DoubleVar] = {}
        self._param_labels: dict[str, tk.Label] = {}
        for name in (
            'head_yaw',
            'head_pitch',
            'head_roll',
            'mouth_open',
            'eye_l',
            'eye_r',
            'blink',
        ):
            row = tk.Frame(bar_frame)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=name, width=12, anchor='w').pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            scale = tk.Scale(
                row,
                from_=-90,
                to=90,
                orient=tk.HORIZONTAL,
                variable=var,
                showvalue=False,
                state='normal',
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            lbl = tk.Label(row, text='0.00', width=6, anchor='e')
            lbl.pack(side=tk.RIGHT)
            self._param_vars[name] = var
            self._param_labels[name] = lbl

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.btn_start = tk.Button(
            btn_frame, text='Avvia tracking', command=self._start_tracking, width=16
        )
        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_stop = tk.Button(
            btn_frame,
            text='Ferma tracking',
            command=self._stop_tracking,
            width=16,
            state='disabled',
        )
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.btn_quit = tk.Button(btn_frame, text='Esci', command=self.destroy, width=10)
        self.btn_quit.pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value='Camera attiva — premi Avvia tracking')
        tk.Label(btn_frame, textvariable=self.status_var, anchor='e').pack(side=tk.RIGHT)

    # -- camera (always on) ----------------------------------------------

    def _start_camera(self) -> None:
        """Start the camera preview thread immediately on app launch."""
        self._cam_stop.clear()
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()

    def _cam_loop(self) -> None:
        """Yield frames from the camera for the preview (no tracking)."""
        try:
            for _t, rgb in camera_stream_frames(device=0, fps=30.0, size=_CAM_SIZE, gray=False):
                if self._cam_stop.is_set():
                    break
                # rgb is float32 [0,1] RGB -> convert to uint8 BGR for OpenCV/PIL
                frame_bgr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                frame_bgr = frame_bgr[..., ::-1]  # RGB -> BGR
                with self._lock:
                    self._latest_frame = frame_bgr
                time.sleep(1.0 / 30.0)
        except Exception as exc:  # pragma: no cover - runtime errors shown in UI
            self.status_var.set(f'Errore camera: {exc}')
        finally:
            with self._lock:
                self._latest_frame = None

    # -- tracking + VMC (started by "Avvia tracking") --------------------

    def _start_tracking(self) -> None:
        """Start the MediaPipe landmarker, the Dense-Armor pipeline and VMC."""
        if self._tracking:
            return
        if self._latest_frame is None:
            self.status_var.set('Errore: camera non disponibile')
            return

        self._tracking = True
        self._track_stop.clear()
        self._latest_params = {}

        self.btn_start.configure(state='disabled')
        self.btn_stop.configure(state='normal')
        self.status_var.set('Caricamento MediaPipe…')

        try:
            self._detector = FaceDetector.open_default('face_landmarker.task')
        except Exception as exc:  # pragma: no cover - model file missing
            self.status_var.set(f'MediaPipe non disponibile: {exc}')
            self._tracking = False
            self.btn_start.configure(state='normal')
            self.btn_stop.configure(state='disabled')
            return

        try:
            self._pipeline = AvatarPipeline(
                max_step=0.1, max_step_deg=5.0, vmc_host='127.0.0.1', vmc_port=self.vmc_port
            )
        except Exception as exc:  # pragma: no cover - python-osc missing
            self.status_var.set(f'Pipeline/VMC non disponibile: {exc}')
            self._tracking = False
            self.btn_start.configure(state='normal')
            self.btn_stop.configure(state='disabled')
            return

        self.status_var.set('Tracking attivo — muovi testa e bocca')
        self._track_thread = threading.Thread(target=self._track_loop, daemon=True)
        self._track_thread.start()

    def _stop_tracking(self) -> None:
        """Stop tracking and VMC, keep camera running."""
        if not self._tracking:
            return
        self._tracking = False
        self._track_stop.set()
        if self._track_thread is not None:
            self._track_thread.join(timeout=2.0)
            self._track_thread = None
        self._detector = None
        self._pipeline = None
        self._close_vmc()
        self.btn_start.configure(state='normal')
        self.btn_stop.configure(state='disabled')
        self.status_var.set('Tracking fermato — camera attiva')
        with self._lock:
            self._latest_params = {}

    def _track_loop(self) -> None:
        """Run MediaPipe + Dense-Armor pipeline on the latest camera frame."""
        try:
            while not self._track_stop.is_set():
                with self._lock:
                    frame = self._latest_frame.copy() if self._latest_frame is not None else None
                if frame is None:
                    time.sleep(0.01)
                    continue

                # frame is uint8 BGR (from camera loop); detector expects RGB
                rgb = frame[..., ::-1]
                assert self._detector is not None
                frame_dict = self._detector.detect(rgb, time.time())

                if self._pipeline is not None:
                    params = self._pipeline.step(frame_dict)

                    with self._lock:
                        self._latest_params = params

                time.sleep(1.0 / 30.0)

        except Exception as exc:  # pragma: no cover - runtime errors shown in UI
            self.status_var.set(f'Errore tracking: {exc}')
        finally:
            self._tracking = False
            self.root.after(0, self._stop_tracking)

    # -- VMC -------------------------------------------------------------

    def _close_vmc(self) -> None:
        if self._pipeline is not None and self._pipeline.vmc is not None:
            with contextlib.suppress(Exception):
                self._pipeline.vmc.close()
            self._pipeline.vmc = None

    # -- UI refresh ------------------------------------------------------

    def _refresh_ui(self) -> None:
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            params = dict(self._latest_params)

        if frame is not None:
            self._show_video(frame)
            if self._tracking:
                self._show_avatar(params)
                self._update_bars(params)

        self.root.after(33, self._refresh_ui)

    def _show_video(self, frame_bgr: np.ndarray) -> None:
        if not _TK_AVAILABLE:
            return
        h, w = frame_bgr.shape[:2]
        max_w, max_h = 320, 240
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        small = cv2_resize(frame_bgr, (new_w, new_h))
        rgb = small[..., ::-1]  # BGR -> RGB
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo, text='')
        self.video_label.image = photo  # type: ignore[attr-defined]

    def _show_avatar(self, params: dict[str, float]) -> None:
        avatar = draw_avatar(params, size=_AVATAR_SIZE)
        rgb = cv2_cvtColor_rgb(avatar)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.avatar_label.configure(image=photo, text='')
        self.avatar_label.image = photo  # type: ignore[attr-defined]

    def _update_bars(self, params: dict[str, float]) -> None:
        for name in (
            'head_yaw',
            'head_pitch',
            'head_roll',
            'mouth_open',
            'eye_l',
            'eye_r',
            'blink',
        ):
            val = float(params.get(name, 0.0))
            self._param_vars[name].set(val)
            self._param_labels[name].configure(text=f'{val:.2f}')

    def destroy(self) -> None:
        """Clean shutdown: stop camera and tracking threads."""
        self._cam_stop.set()
        self._track_stop.set()
        if self._cam_thread is not None:
            self._cam_thread.join(timeout=2.0)
        if self._track_thread is not None:
            self._track_thread.join(timeout=2.0)
        self._close_vmc()
        self.root.destroy()


def cv2_resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize without importing cv2 at module level (optional)."""
    try:
        import cv2  # noqa: PLC0415

        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # type: ignore[no-any-return]
    except ImportError:
        from PIL import Image  # noqa: PLC0415

        pil = Image.fromarray(img)
        pil = pil.resize(size, Image.Resampling.BILINEAR)
        return np.array(pil)  # type: ignore[no-any-return]


def cv2_cvtColor_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR -> RGB without importing cv2 at module level."""
    return bgr[..., ::-1].copy()  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(description='D-Avatar live app')
    parser.add_argument('--vmc-port', type=int, default=39539, help='VMC/UDP port (default 39539)')
    args = parser.parse_args()

    if not _TK_AVAILABLE:
        print('tkinter/Pillow not available: cannot run the app')
        sys.exit(1)

    root = tk.Tk()
    app = DAvatarApp(root, vmc_port=args.vmc_port)
    root.protocol('WM_DELETE_WINDOW', app.destroy)
    root.mainloop()


if __name__ == '__main__':
    main()
