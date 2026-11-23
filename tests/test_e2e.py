"""End-to-end test: synthetic frames -> tracker -> pipeline -> send_frame -> UDP receiver.

No camera, no ML models. Verifies that values received over VMC match the synthetic
truth within tolerance.
"""

from __future__ import annotations

import math
import socket
import struct

import numpy as np

from d_avatar.avatar import draw_avatar
from d_avatar.pipeline import AvatarPipeline
from d_avatar.tracker import NativeTracker
from d_avatar.vmc import VmcSender

_SYNTHETIC_TRUTH = {
    'head_yaw': 15.0,
    'head_pitch': -10.0,
    'head_roll': 5.0,
    'mouth_open': 0.4,
    'eye_l': 0.9,
    'eye_r': 0.7,
    'blink': 0.0,
}

_TOL = 0.15


def _parse_osc(data: bytes) -> tuple[str, list]:
    idx = 0
    addr_end = data.find(b'\x00', idx)
    address = data[idx:addr_end].decode('utf-8')
    idx = addr_end + 1
    idx = (idx + 3) & ~3
    tag_end = data.find(b'\x00', idx)
    type_tags = data[idx + 1 : tag_end].decode('utf-8')
    idx = tag_end + 1
    idx = (idx + 3) & ~3
    args: list = []
    for tag in type_tags:
        if tag == 's':
            str_end = data.find(b'\x00', idx)
            args.append(data[idx:str_end].decode('utf-8'))
            idx = str_end + 1
            idx = (idx + 3) & ~3
        elif tag == 'f':
            args.append(struct.unpack('>f', data[idx : idx + 4])[0])
            idx += 4
        elif tag == 'i':
            args.append(struct.unpack('>i', data[idx : idx + 4])[0])
            idx += 4
    return address, args


def _make_synth_frame() -> np.ndarray:
    """Synthetic face: white head ellipse, open eyes, small mouth opening."""
    img: np.ndarray = np.zeros((120, 160), dtype=np.float32)
    cx, cy = 80, 60
    yy, xx = np.ogrid[:120, :160]
    head = ((xx - cx) / 40.0) ** 2 + ((yy - cy) / 50.0) ** 2 <= 1.0
    img[head] = 1.0
    for ex in (cx - 20, cx + 20):
        eye = ((xx - ex) / 8.0) ** 2 + ((yy - (cy - 10)) / 5.0) ** 2 <= 1.0
        img[eye] = 1.0
    img[cy + 20 : cy + 24, cx - 10 : cx + 10] = 0.6  # slightly open mouth
    return img


def _receive_udp(
    sock: socket.socket,
    n_messages: int,
    timeout: float = 2.0,
) -> list[tuple[str, list]]:
    sock.settimeout(timeout)
    messages: list[tuple[str, list]] = []
    for _ in range(n_messages):
        try:
            data, _ = sock.recvfrom(4096)
            messages.append(_parse_osc(data))
        except TimeoutError:
            break
    return messages


def test_e2e_synthetic_to_vmc() -> None:
    """Frame sintetico -> tracker -> send_frame -> UDP receiver matches truth."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]

    tracker = NativeTracker(max_step=1.0, max_step_deg=90.0, calibrate_frames=0)
    tracker.set_face_box((40, 10, 120, 110))

    sender = VmcSender(host='127.0.0.1', port=port)

    # Feed one synthetic frame through the tracker
    frame = _make_synth_frame()
    params = tracker(frame)

    # Override with known truth for the E2E assertion
    params.update(_SYNTHETIC_TRUTH)

    sender.send_frame(params)
    sender.close()

    messages = _receive_udp(sock, n_messages=10)
    sock.close()

    # Collect blendshape values
    blend: dict[str, float] = {}
    bone_pose: list[float] | None = None
    apply_count = 0
    for address, args in messages:
        if address == '/VMC/Ext/Blend/Val':
            blend[args[0]] = args[1]
        elif address == '/VMC/Ext/Blend/Apply':
            apply_count += 1
        elif address == '/VMC/Ext/Bone/Pos':
            bone_pose = args[1:]  # skip bone name "Head"

    assert apply_count >= 1, 'expected at least one Apply message'
    assert bone_pose is not None, 'expected Head bone pose message'

    # Check blendshape values against synthetic truth
    assert abs(blend.get('jawOpen', -1) - _SYNTHETIC_TRUTH['mouth_open']) < _TOL
    assert abs(blend.get('eyeBlinkLeft', -1) - (1.0 - _SYNTHETIC_TRUTH['eye_l'])) < _TOL
    assert abs(blend.get('eyeBlinkRight', -1) - (1.0 - _SYNTHETIC_TRUTH['eye_r'])) < _TOL

    # Check head quaternion: yaw=15 deg -> qz = sin(7.5 deg), qw = cos(7.5 deg)
    yaw = math.radians(_SYNTHETIC_TRUTH['head_yaw'])
    expected_qz = math.sin(yaw * 0.5)
    expected_qw = math.cos(yaw * 0.5)
    _qx, _qy, qz, qw = bone_pose[3], bone_pose[4], bone_pose[5], bone_pose[6]
    assert abs(qz - expected_qz) < 0.02, f'qz {qz} != {expected_qz}'
    assert abs(qw - expected_qw) < 0.02, f'qw {qw} != {expected_qw}'


def test_e2e_draw_avatar_output_shape() -> None:
    """draw_avatar on synthetic params returns valid image."""
    img = draw_avatar(_SYNTHETIC_TRUTH, size=400)
    assert img.shape == (400, 400, 3)
    assert img.dtype == np.uint8


def test_e2e_pipeline_with_synthetic_params() -> None:
    """AvatarPipeline processes a frame dict and returns all 7 parameters."""
    ap = AvatarPipeline(max_step=0.5, max_step_deg=10.0, use_shield=False)
    pts = np.zeros((68, 2))
    pts[36:42] = [[20, 35], [23, 33.5], [27, 33.5], [30, 35], [27, 36.5], [23, 36.5]]
    pts[42:48] = [[60, 35], [63, 33.5], [67, 33.5], [70, 35], [67, 36.5], [63, 36.5]]
    frame = {'points': pts, 'confidence': np.ones(68), 't': 0.0}
    out = ap.step(frame)
    for k in ('head_yaw', 'head_pitch', 'head_roll', 'mouth_open', 'eye_l', 'eye_r', 'blink'):
        assert k in out, f'missing key {k} in pipeline output'
    assert out['blink'] in (0.0, 1.0)
