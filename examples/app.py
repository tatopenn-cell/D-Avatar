"""D-Avatar example app: live camera → native tracker → 2D avatar + optional VMC.

Window layout:
- left: stabilised face video (Dense-Armor primitives, no dots)
- right: 2D avatar (draw_avatar)
- bottom: status bar with live Dense-Armor class names + 7 parameter bars
- buttons: Start / Stop

Run: python examples/app.py [--detector mediapipe] [--vmc-port 39539]

MediaPipe path is present in the repo but OFF by default: the default path
uses only Dense-Armor primitives (no pre-trained model).
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
from d_avatar.tracker import NativeTracker
from d_avatar.vmc import VmcSender

# Dense-Armor classes used in the default (model-free) path.
_DENSE_ARMOR_CLASSES = (
    'MovingObjectTracker',
    'FrameFeatures',
    'MotionEnergy',
    'GlobalMotion',
    'Registration',
    'Stabilizer',
    'RateBound',
)

_CAM_SIZE = (640, 480)
_AVATAR_SIZE = 400
_CALIBRATION_SECONDS = 3


class DAvatarApp:
    """Tkinter application: camera → tracker → avatar + VMC."""

    def __init__(self, root: tk.Tk, vmc_port: int = 39539, use_mediapipe: bool = False) -> None:
        if not _TK_AVAILABLE:
            raise ImportError('tkinter and Pillow are required for the app')
        self.root = root
        self.root.title('D-Avatar — live native tracker')
        self.vmc_port = vmc_port
        self.use_mediapipe = use_mediapipe

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest_params: dict[str, float] = {}
        self._latest_frame: np.ndarray | None = None
        self._latest_stabilised: np.ndarray | None = None
        self._lock = threading.Lock()
        self._vmc: VmcSender | None = None
        self._tracker: NativeTracker | None = None

        self._build_ui()

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        # Video + avatar area
        area = tk.Frame(self.root)
        area.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.video_label = tk.Label(area, bg='black', text='video (stopped)')
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.avatar_label = tk.Label(area, bg='#e0e0e0', text='avatar')
        self.avatar_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Status + parameter bars
        bar_frame = tk.Frame(self.root)
        bar_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Label(
            bar_frame, text='Dense-Armor classes: ' + ', '.join(_DENSE_ARMOR_CLASSES), anchor='w'
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
                state='disabled',
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            lbl = tk.Label(row, text='0.00', width=6, anchor='e')
            lbl.pack(side=tk.RIGHT)
            self._param_vars[name] = var
            self._param_labels[name] = lbl

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.btn_start = tk.Button(btn_frame, text='Avvia', command=self._start, width=12)
        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_stop = tk.Button(
            btn_frame, text='Ferma', command=self._stop_app, width=12, state='disabled'
        )
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value='Pronto')
        tk.Label(btn_frame, textvariable=self.status_var, anchor='e').pack(side=tk.RIGHT)

    # -- start / stop -----------------------------------------------------

    def _start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop.clear()
        self._latest_params = {}
        self._latest_frame = None
        self._latest_stabilised = None
        self.btn_start.configure(state='disabled')
        self.btn_stop.configure(state='normal')
        self.status_var.set('Avvio camera e calibrazione…')
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.root.after(100, self._refresh_ui)

    def _stop_app(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_vmc()
        self._tracker = None
        self.btn_start.configure(state='normal')
        self.btn_stop.configure(state='disabled')
        self.status_var.set('Fermato')
        self.video_label.configure(image='', text='video (stopped)')
        self.avatar_label.configure(image='', text='avatar')

    # -- main loop --------------------------------------------------------

    def _run_loop(self) -> None:
        """Camera capture, calibration, tracking, VMC send (runs in thread)."""
        try:
            self._open_vmc()
            self._tracker = NativeTracker(max_step=0.1, max_step_deg=5.0, calibrate_frames=90)

            # Calibration: first ~3 seconds at 30 fps = ~90 frames.
            # During calibration the tracker builds its background model;
            # we set the face box manually once a stable track exists.
            calibrated = False

            for calib_frames, (_t, rgb) in enumerate(
                camera_stream_frames(device=0, fps=30.0, size=_CAM_SIZE, gray=False)
            ):
                if self._stop.is_set():
                    break

                # rgb is float32 [0,1] RGB → convert to uint8 BGR for OpenCV/PIL
                frame_bgr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                frame_bgr = frame_bgr[..., ::-1]  # RGB → BGR

                gray = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                gray = gray.mean(axis=2).astype(np.uint8)

                # Tracker needs color/gray frame; it converts internally.
                params = self._tracker(frame_bgr)

                with self._lock:
                    self._latest_params = params
                    self._latest_frame = frame_bgr
                    self._latest_stabilised = (
                        frame_bgr  # tracker provides stabilised signal via box
                    )

                if self._vmc is not None:
                    self._vmc.send_frame(params)

                if not calibrated and calib_frames >= self._tracker.calibrate_frames:
                    calibrated = True
                    self.status_var.set('Calibrato — avatar attivo')

                time.sleep(1.0 / 30.0)

        except Exception as exc:  # pragma: no cover - runtime errors shown in UI
            self.status_var.set(f'Errore: {exc}')
        finally:
            self._running = False
            self.root.after(0, self._stop_app)

    # -- UI refresh -------------------------------------------------------

    def _refresh_ui(self) -> None:
        if not self._running:
            return
        with self._lock:
            params = dict(self._latest_params)
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is not None:
            self._show_video(frame)
            self._show_avatar(params)
            self._update_bars(params)

        self.root.after(33, self._refresh_ui)

    def _show_video(self, frame_bgr: np.ndarray) -> None:
        if not _TK_AVAILABLE:
            return
        # Resize to fit the label area while keeping aspect
        h, w = frame_bgr.shape[:2]
        max_w, max_h = 320, 240
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        small = cv2_resize(frame_bgr, (new_w, new_h))
        rgb = small[..., ::-1]  # BGR → RGB
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo, text='')
        self.video_label.image = photo  # type: ignore[attr-defined]  # keep reference

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

    # -- VMC --------------------------------------------------------------

    def _open_vmc(self) -> None:
        try:
            self._vmc = VmcSender(host='127.0.0.1', port=self.vmc_port)
        except ImportError:
            self._vmc = None
            self.status_var.set('python-osc non installato: VMC disattivato')

    def _close_vmc(self) -> None:
        if self._vmc is not None:
            with contextlib.suppress(Exception):
                self._vmc.close()
            self._vmc = None


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
    """BGR → RGB without importing cv2 at module level."""
    return bgr[..., ::-1].copy()  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(description='D-Avatar live app')
    parser.add_argument(
        '--detector',
        choices=['mediapipe'],
        default=None,
        help='detector to use (mediapipe path is OFF by default)',
    )
    parser.add_argument('--vmc-port', type=int, default=39539, help='VMC/UDP port (default 39539)')
    args = parser.parse_args()

    if not _TK_AVAILABLE:
        print('tkinter/Pillow not available: cannot run the app')
        sys.exit(1)

    root = tk.Tk()
    DAvatarApp(root, vmc_port=args.vmc_port, use_mediapipe=(args.detector == 'mediapipe'))
    root.mainloop()


if __name__ == '__main__':
    main()
