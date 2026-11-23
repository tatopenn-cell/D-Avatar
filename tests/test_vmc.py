"""
Tests for VmcSender.

Covers:
- Bind UDP socket on loopback port 0, send one frame, parse packets.
- Assert addresses and values.
- 6 parameters -> 7 messages (6 Val + 1 Apply).
- send_frame maps to VRM blendshape names and head bone pose.
"""

import math
import socket
import struct

from d_avatar.vmc import VmcSender

_EXPECTED_VAL_COUNT = 6
_EXPECTED_APPLY_COUNT = 1
_EXPECTED_TOTAL = _EXPECTED_VAL_COUNT + _EXPECTED_APPLY_COUNT
_DEFAULT_PORT = 39539
_CUSTOM_PORT = 12345
_OTHER_PORT = 9999


def _parse_osc_message(data: bytes) -> tuple[str, list]:
    """Simple OSC message parser for testing.
    Returns (address, args).
    """
    idx = 0
    addr_end = data.find(b'\x00', idx)
    if addr_end == -1:
        return '', []
    address = data[idx:addr_end].decode('utf-8')
    idx = addr_end + 1
    idx = (idx + 3) & ~3

    if idx >= len(data) or data[idx] != ord(','):
        return address, []
    tag_end = data.find(b'\x00', idx)
    if tag_end == -1:
        return address, []
    type_tags = data[idx + 1 : tag_end].decode('utf-8')
    idx = tag_end + 1
    idx = (idx + 3) & ~3

    args = []
    for tag in type_tags:
        if tag == 's':
            str_end = data.find(b'\x00', idx)
            if str_end == -1:
                break
            args.append(data[idx:str_end].decode('utf-8'))
            idx = str_end + 1
            idx = (idx + 3) & ~3
        elif tag == 'f':
            if idx + 4 <= len(data):
                val = struct.unpack('>f', data[idx : idx + 4])[0]
                args.append(val)
                idx += 4
        elif tag == 'i':
            if idx + 4 <= len(data):
                val = struct.unpack('>i', data[idx : idx + 4])[0]
                args.append(val)
                idx += 4
        else:
            pass
    return address, args


def test_vmc_sender_import() -> None:
    """VmcSender can be instantiated when python-osc is available."""
    sender = VmcSender()
    assert sender.host == '127.0.0.1'
    assert sender.port == _DEFAULT_PORT
    sender.close()


def test_vmc_sender_send_one_frame() -> None:
    """Send one frame with 6 parameters -> 7 messages (6 Val + 1 Apply)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.settimeout(1.0)

    sender = VmcSender(host='127.0.0.1', port=port)

    params = {
        'eye_l': 1.0,
        'eye_r': 0.8,
        'mouth_open': 0.3,
        'brow_l': 0.5,
        'brow_r': 0.5,
        'gaze_x': 0.2,
    }

    sender.send(params)

    messages = []
    for _ in range(_EXPECTED_TOTAL):
        try:
            data, _ = sock.recvfrom(4096)
            messages.append(data)
        except TimeoutError:
            break

    sender.close()
    sock.close()

    assert len(messages) == _EXPECTED_TOTAL, (
        f'Expected {_EXPECTED_TOTAL} messages, got {len(messages)}'
    )

    val_count = 0
    apply_count = 0
    for data in messages:
        address, args = _parse_osc_message(data)
        if address == '/VMC/Ext/Blend/Val':
            val_count += 1
            assert len(args) == 2
            assert isinstance(args[0], str)
            assert isinstance(args[1], float)
        elif address == '/VMC/Ext/Blend/Apply':
            apply_count += 1
            assert len(args) == 0

    assert val_count == _EXPECTED_VAL_COUNT, (
        f'Expected {_EXPECTED_VAL_COUNT} Val messages, got {val_count}'
    )
    assert apply_count == _EXPECTED_APPLY_COUNT, (
        f'Expected {_EXPECTED_APPLY_COUNT} Apply message, got {apply_count}'
    )


def test_vmc_sender_custom_host_port() -> None:
    """VmcSender accepts custom host and port."""
    sender = VmcSender(host='192.168.1.100', port=_CUSTOM_PORT)
    assert sender.host == '192.168.1.100'
    assert sender.port == _CUSTOM_PORT
    sender.close()


def test_vmc_sender_context_manager() -> None:
    """VmcSender works as context manager."""
    with VmcSender() as sender:
        sender.send({'test': 1.0})


def test_vmc_sender_get_params() -> None:
    """get_params returns config."""
    sender = VmcSender(host='10.0.0.1', port=_OTHER_PORT)
    params = sender.get_params()
    assert params['host'] == '10.0.0.1'
    assert params['port'] == _OTHER_PORT
    sender.close()


def test_vmc_sender_close_closes_socket() -> None:
    sender = VmcSender()
    sender.close()
    assert sender._client._sock.fileno() == -1


def test_vmc_send_frame_maps_to_vrm_blendshapes_and_head_bone() -> None:
    """send_frame maps D-Avatar params to VRM blendshapes + Head bone quaternion."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.settimeout(1.0)

    sender = VmcSender(host='127.0.0.1', port=port)

    # yaw = 90 deg -> quaternion (x,y,z,w) = (0, 0, sqrt(2)/2, sqrt(2)/2)
    sqrt2_over_2 = math.sqrt(2) / 2.0
    expected_quat = (0.0, 0.0, sqrt2_over_2, sqrt2_over_2)

    params = {
        'head_yaw': 90.0,
        'head_pitch': 0.0,
        'head_roll': 0.0,
        'mouth_open': 0.5,
        'eye_l': 1.0,
        'eye_r': 0.0,
        'blink': 0.0,
    }

    sender.send_frame(params)

    messages = []
    for _ in range(10):
        try:
            data, _ = sock.recvfrom(4096)
            messages.append(data)
        except TimeoutError:
            break

    sender.close()
    sock.close()

    assert len(messages) == 5, f'Expected 5 messages, got {len(messages)}'

    val_addresses = []
    bone_addresses = []
    apply_count = 0
    for data in messages:
        address, args = _parse_osc_message(data)
        if address == '/VMC/Ext/Blend/Val':
            val_addresses.append((args[0], args[1]))
            assert len(args) == 2
            assert isinstance(args[0], str)
            assert isinstance(args[1], float)
        elif address == '/VMC/Ext/Blend/Apply':
            apply_count += 1
            assert len(args) == 0
        elif address == '/VMC/Ext/Bone/Pos':
            bone_addresses.append(args)
            assert len(args) == 8
            assert args[0] == 'Head'
            assert all(isinstance(v, float) for v in args[1:])

    assert apply_count == 1
    assert len(bone_addresses) == 1
    bone_args = bone_addresses[0]
    # Position is (0,0,0), quaternion is (x,y,z,w) = args[4:8]
    quat = tuple(bone_args[4:8])
    for got, exp in zip(quat, expected_quat, strict=False):
        assert abs(got - exp) < 1e-5, f'quat {quat} != expected {expected_quat}'

    # Blendshape values
    val_dict = dict(val_addresses)
    assert val_dict['eyeBlinkLeft'] == 0.0  # 1 - eye_l = 0
    assert val_dict['eyeBlinkRight'] == 1.0  # 1 - eye_r = 1
    assert val_dict['jawOpen'] == 0.5
