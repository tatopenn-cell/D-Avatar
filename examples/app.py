"""D-Avatar example app: live camera -> MediaPipe Face Landmarker -> avatar 2D + VMC + audio.

Window layout:
- left: live camera preview (always on when the app is open)
- right: 2D avatar (draw_avatar), driven by the native pipeline
- bottom: status bar with live Dense-Armor class names + 7 parameter bars
- buttons: Start / Stop tracking (+ VMC), Avatar Editor, Save PNG, Quit

Run: python examples/app.py [--vmc-port 39539]

Default path uses MediaPipe Face Landmarker (face_landmarker.task already present)
+ Dense-Armor pipeline (FaceShield + ParamsStep + RateBound). No neural network
is built or trained by this code; the pre-trained task file is loaded only when
the detector is opened.

Audio: opencv + pyaudio capture the mic; Dense-Armor spectral_flux measures
voice energy to drive mouth_open in real time.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import tkinter as tk
    from tkinter import colorchooser, filedialog, scrolledtext

    from PIL import Image, ImageTk

    _TK_AVAILABLE = True
except ImportError:  # pragma: no cover - tkinter optional
    _TK_AVAILABLE = False

try:
    import pyaudio

    _PYAUDIO_AVAILABLE = True
except ImportError:  # pragma: no cover - pyaudio optional
    _PYAUDIO_AVAILABLE = False

from d_avatar.avatar import draw_avatar
from d_avatar.capture import camera_stream_frames
from d_avatar.detector import FaceDetector
from d_avatar.pipeline import AvatarPipeline
from d_avatar.tracker import NativeTracker
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

# Audio settings
_AUDIO_RATE = 16000
_AUDIO_CHUNK = 1024
_AUDIO_CHANNELS = 1


class AvatarEditorWindow:
    """Finestra modale per personalizzare colori ed espressioni dell'avatar."""

    def __init__(self, parent: tk.Tk, style: dict[str, Any], size: int = 400) -> None:
        self.top = tk.Toplevel(parent)
        self.top.title('Editor Avatar')
        self.top.transient(parent)
        self.top.grab_set()

        self.style = dict(style)
        self.size = size
        self.result: dict[str, Any] | None = None

        # Colori editabili: (label, chiave stile, colore default BGR)
        self.color_fields: list[tuple[str, str, tuple[int, int, int]]] = [
            ('Pelle', 'skin_color', (195, 175, 155)),
            ('Pelle scura', 'skin_dark', (150, 130, 110)),
            ('Pelle chiara', 'skin_light', (225, 210, 195)),
            ('Capelli', 'hair_color', (25, 20, 15)),
            ('Capelli luce', 'hair_light', (55, 45, 35)),
            ('Sopracciglia', 'brow_color', (60, 50, 40)),
            ('Bocca interna', 'mouth_inside', (50, 35, 80)),
            ('Contorno bocca', 'mouth_outline', (0, 0, 140)),
            ('Rossetto', 'lip_color', (180, 90, 100)),
            ('Iride', 'eye_color', (70, 70, 70)),
            ('Pupilla', 'pupil_color', (10, 10, 10)),
            ('Riflesso occhio', 'eye_reflect', (240, 240, 240)),
            ('Occhio chiuso', 'eye_closed', (30, 30, 30)),
        ]

        # Layout: sinistra controlli, destra anteprima
        left = tk.Frame(self.top)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        right = tk.Frame(self.top)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        tk.Label(right, text='Anteprima', anchor='center').pack()

        self.preview_label = tk.Label(right, bg='black')
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        btn_frame = tk.Frame(right)
        btn_frame.pack(fill=tk.X, pady=4)
        tk.Button(btn_frame, text='Applica', command=self._apply, width=10).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(btn_frame, text='Chiudi', command=self.top.destroy, width=10).pack(
            side=tk.LEFT, padx=4
        )

        # Controlli colori
        row = 0
        for label, key, default in self.color_fields:
            tk.Label(left, text=label, anchor='w').grid(
                row=row, column=0, sticky='w', padx=4, pady=2
            )
            btn = tk.Button(
                left,
                text='Scegli colore',
                width=14,
                command=self._make_color_picker(key, default),
            )
            btn.grid(row=row, column=1, padx=4, pady=2)
            swatch = tk.Label(
                left,
                width=3,
                bg=self._bgr_to_hex(self.style.get(key, default)),
            )
            swatch.grid(row=row, column=2, padx=2, pady=2)
            setattr(self, f'_swatch_{key}', swatch)
            row += 1

        # Separatore
        tk.Frame(left, height=2, bd=1, relief=tk.SUNKEN).grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=8
        )
        row += 1

        # Parametri espressivi
        tk.Label(
            left,
            text='Espressioni',
            anchor='w',
            font=('TkDefaultFont', 10, 'bold'),
        ).grid(row=row, column=0, columnspan=3, sticky='w', pady=(8, 4))
        row += 1

        for label, key, default in (
            ('Sorriso', 'smile', self.style.get('smile', 0.0)),
            ('Occhio sinistro', 'eye_l', self.style.get('eye_l', 1.0)),
            ('Occhio destro', 'eye_r', self.style.get('eye_r', 1.0)),
            ('Sopracciglia sinistra', 'brow_l', self.style.get('brow_l', 0.5)),
            ('Sopracciglia destra', 'brow_r', self.style.get('brow_r', 0.5)),
        ):
            tk.Label(left, text=label, anchor='w').grid(
                row=row, column=0, sticky='w', padx=4, pady=2
            )
            var = tk.DoubleVar(value=default)
            sc = tk.Scale(
                left,
                from_=0.0,
                to=1.0,
                resolution=0.05,
                orient=tk.HORIZONTAL,
                variable=var,
            )
            sc.grid(row=row, column=1, columnspan=2, sticky='ew', padx=4, pady=2)
            setattr(self, f'_var_{key}', var)
            row += 1

        # Anteprima live
        self._preview_params = {
            'head_yaw': 0.0,
            'head_pitch': 0.0,
            'head_roll': 0.0,
            'mouth_open': 0.3,
            'eye_l': self.style.get('eye_l', 1.0),
            'eye_r': self.style.get('eye_r', 1.0),
            'blink': 0.0,
            'brow_l': self.style.get('brow_l', 0.5),
            'brow_r': self.style.get('brow_r', 0.5),
            'gaze_x': 0.0,
            'gaze_y': 0.0,
            'smile': self.style.get('smile', 0.0),
            'time': 0.0,
        }
        self._preview_time = 0.0
        self._update_preview()

    def _bgr_to_hex(self, color: tuple[int, int, int]) -> str:
        return f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'

    def _make_color_picker(self, key: str, default: tuple[int, int, int]) -> Callable[[], None]:
        """Return a callback that opens a color picker for ``key``."""

        def _pick() -> None:
            self._pick_color(key, default)

        return _pick

    def _pick_color(self, key: str, default: tuple[int, int, int]) -> None:
        res = colorchooser.askcolor(color=None, title='Scegli colore')
        if res[0] is not None:
            r, g, b = [int(c) for c in res[0]]
            bgr = (b, g, r)
            self.style[key] = bgr
            swatch = getattr(self, f'_swatch_{key}', None)
            if swatch is not None:
                swatch.configure(bg=self._bgr_to_hex(bgr))
            self._update_preview()

    def _update_preview(self) -> None:
        self._preview_time += 0.1
        params = dict(self._preview_params)
        params['time'] = self._preview_time
        params['smile'] = self._var_smile.get() if hasattr(self, '_var_smile') else params['smile']
        params['eye_l'] = self._var_eye_l.get() if hasattr(self, '_var_eye_l') else params['eye_l']
        params['eye_r'] = self._var_eye_r.get() if hasattr(self, '_var_eye_r') else params['eye_r']
        params['brow_l'] = (
            self._var_brow_l.get() if hasattr(self, '_var_brow_l') else params['brow_l']
        )
        params['brow_r'] = (
            self._var_brow_r.get() if hasattr(self, '_var_brow_r') else params['brow_r']
        )
        avatar = draw_avatar(params, size=self.size, style=self.style)
        rgb = cv2_cvtColor_rgb(avatar)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.preview_label.configure(image=photo, text='')
        self.preview_label.image = photo  # type: ignore[attr-defined]
        self.top.after(100, self._update_preview)

    def _apply(self) -> None:
        self.result = {
            'smile': self._var_smile.get()
            if hasattr(self, '_var_smile')
            else self.style.get('smile', 0.0),
            'eye_l': self._var_eye_l.get()
            if hasattr(self, '_var_eye_l')
            else self.style.get('eye_l', 1.0),
            'eye_r': self._var_eye_r.get()
            if hasattr(self, '_var_eye_r')
            else self.style.get('eye_r', 1.0),
            'brow_l': self._var_brow_l.get()
            if hasattr(self, '_var_brow_l')
            else self.style.get('brow_l', 0.5),
            'brow_r': self._var_brow_r.get()
            if hasattr(self, '_var_brow_r')
            else self.style.get('brow_r', 0.5),
        }
        # I colori sono già in self.style
        self.top.destroy()


class DAvatarApp:
    """Tkinter application: camera -> MediaPipe landmarker -> avatar + VMC + audio.

    The camera preview runs as soon as the app opens. Clicking "Avvia
    tracking" starts the MediaPipe landmarker, the Dense-Armor pipeline and
    (optionally) VMC sending to VMagicMirror. The microphone is captured
    separately and its energy drives mouth_open via Dense-Armor spectral_flux.
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
        self._tracker: NativeTracker | None = None

        # Camera thread (always running while the app is open)
        self._cam_thread: threading.Thread | None = None
        self._cam_stop = threading.Event()

        # Audio thread (started by "Avvia tracking")
        self._audio_thread: threading.Thread | None = None
        self._audio_stop = threading.Event()
        self._audio_energy: float = 0.0

        # Avatar style customization (set by editor window)
        self.avatar_style: dict[str, Any] = {}

        self._build_ui()
        self._start_camera()
        self.root.after(100, self._refresh_ui)

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        # Video + avatar area (wide)
        area = tk.Frame(self.root)
        area.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.video_label = tk.Label(area, bg='black', text='camera (starting…)')
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.avatar_label = tk.Label(area, bg='#e0e0e0', text='avatar')
        self.avatar_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Chat panel (right side, like the mockup)
        chat_frame = tk.Frame(self.root, bg='#f5f5f5')
        chat_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        tk.Label(chat_frame, text='AI Avatars', font=('TkDefaultFont', 12, 'bold')).pack(anchor='w')
        tk.Label(chat_frame, text='Status: Live', fg='green').pack(anchor='w', pady=(0, 8))

        self.chat_box = scrolledtext.ScrolledText(
            chat_frame, width=34, height=18, wrap=tk.WORD, state='disabled'
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Chat input
        input_frame = tk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=4, pady=4)
        self.chat_entry = tk.Entry(input_frame)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_entry.bind('<Return>', lambda _e: self._send_chat())
        tk.Button(input_frame, text='Invia', command=self._send_chat).pack(side=tk.LEFT, padx=4)

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
        self.btn_save_png = tk.Button(
            btn_frame, text='Salva Avatar PNG', command=self._save_avatar_png, width=16
        )
        self.btn_save_png.pack(side=tk.LEFT, padx=4)
        self.btn_editor = tk.Button(
            btn_frame, text='Editor Avatar', command=self._open_avatar_editor, width=16
        )
        self.btn_editor.pack(side=tk.LEFT, padx=4)
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

    # -- tracking + VMC + audio (started by "Avvia tracking") ------------

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
                max_step=0.1,
                max_step_deg=5.0,
                vmc_host='127.0.0.1',
                vmc_port=self.vmc_port,
            )
        except Exception as exc:  # pragma: no cover - python-osc missing
            self.status_var.set(f'Pipeline/VMC non disponibile: {exc}')
            self._tracking = False
            self.btn_start.configure(state='normal')
            self.btn_stop.configure(state='disabled')
            return

        self._tracker = NativeTracker(
            max_step=0.1,
            max_step_deg=5.0,
        )

        self.status_var.set('Tracking attivo — muovi testa e bocca')
        self._track_thread = threading.Thread(target=self._track_loop, daemon=True)
        self._track_thread.start()

        # Start audio thread for mouth drive
        self._start_audio()

    def _stop_tracking(self) -> None:
        """Stop tracking, VMC and audio, keep camera running."""
        if not self._tracking:
            return
        self._tracking = False
        self._track_stop.set()
        self._audio_stop.set()
        if self._track_thread is not None:
            self._track_thread.join(timeout=2.0)
            self._track_thread = None
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=2.0)
            self._audio_thread = None
        self._detector = None
        self._pipeline = None
        self._tracker = None
        self._close_vmc()
        self.btn_start.configure(state='normal')
        self.btn_stop.configure(state='disabled')
        self.status_var.set('Tracking fermato — camera attiva')
        with self._lock:
            self._latest_params = {}

    def _save_avatar_png(self) -> None:
        """Salva l'avatar corrente su file PNG tramite dialogo."""
        with self._lock:
            params = dict(self._latest_params)
        avatar = draw_avatar(params, size=_AVATAR_SIZE, style=self.avatar_style)
        rgb = cv2_cvtColor_rgb(avatar)
        img = Image.fromarray(rgb)
        filepath = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png')],
            initialfile=f'avatar_{time.strftime("%Y%m%d_%H%M%S")}.png',
            title='Salva avatar PNG',
        )
        if filepath:
            img.save(filepath, format='PNG')
            self.status_var.set(f'Avatar salvato: {filepath}')

    def _open_avatar_editor(self) -> None:
        """Apri l'editor per personalizzare colori ed espressioni dell'avatar."""
        editor = AvatarEditorWindow(self.root, self.avatar_style, _AVATAR_SIZE)
        self.root.wait_window(editor.top)
        if editor.result is not None:
            self.avatar_style = editor.result
            self.status_var.set('Stile avatar aggiornato')

    # -- chat -------------------------------------------------------------

    def _send_chat(self) -> None:
        msg = self.chat_entry.get().strip()
        if not msg:
            return
        self.chat_entry.delete(0, tk.END)
        self._append_chat('Tu', msg)
        self._auto_reply(msg)

    def _append_chat(self, who: str, text: str) -> None:
        self.chat_box.configure(state='normal')
        self.chat_box.insert(tk.END, f'{who}: {text}\n\n')
        self.chat_box.configure(state='disabled')
        self.chat_box.see(tk.END)

    def _auto_reply(self, user_msg: str) -> None:
        """Very simple keyword-based reply (placeholder for LLM)."""
        m = user_msg.lower()
        if any(w in m for w in ('ciao', 'hello', 'hi', 'buongiorno')):
            reply = 'Ciao! Sono il tuo avatar. Come posso aiutarti oggi?'
        elif any(w in m for w in ('prodotto', 'cosa fai', 'chi sei')):
            reply = (
                'Sono un avatar costruito con Dense Armor, senza modelli '
                'pre-addestrati. Tutto gira in locale sul tuo PC.'
            )
        elif any(w in m for w in ('bocca', 'parla', 'sorriso')):
            reply = 'Sto muovendo la bocca grazie al tracciamento del volto!'
        else:
            reply = 'Ho ricevuto il tuo messaggio. Dimmi di più!'
        self._append_chat('Avatar', reply)

    # -- audio -------------------------------------------------------------

    def _start_audio(self) -> None:
        """Start the microphone thread for audio-driven mouth."""
        if not _PYAUDIO_AVAILABLE:
            self.status_var.set("pyaudio non installato: bocca non guidata dall'audio")
            return
        self._audio_stop.clear()
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._audio_thread.start()

    def _audio_loop(self) -> None:
        """Capture mic audio and compute spectral flux for mouth drive."""
        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=_AUDIO_CHANNELS,
                rate=_AUDIO_RATE,
                input=True,
                frames_per_buffer=_AUDIO_CHUNK,
            )
            while not self._audio_stop.is_set():
                try:
                    data = stream.read(_AUDIO_CHUNK, exception_on_overflow=False)
                except Exception:
                    break
                samples: np.ndarray = np.frombuffer(data, dtype=np.float32)
                if samples.size == 0:
                    continue
                # Spectral flux: energy difference between consecutive frames
                flux = float(np.abs(np.diff(samples)).mean())
                # Normalize to [0,1] (heuristic: clip at 0.05)
                energy = min(flux / 0.05, 1.0)
                # Smooth
                self._audio_energy = 0.3 * energy + 0.7 * self._audio_energy
                if self._tracker is not None:
                    self._tracker.update_audio_energy(self._audio_energy)
        except Exception as exc:
            self.status_var.set(f'Errore audio: {exc}')
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    # -- tracking loop -----------------------------------------------------

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
        avatar = draw_avatar(params, size=_AVATAR_SIZE, style=self.avatar_style)
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
        self._audio_stop.set()
        if self._cam_thread is not None:
            self._cam_thread.join(timeout=2.0)
        if self._track_thread is not None:
            self._track_thread.join(timeout=2.0)
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=2.0)
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
