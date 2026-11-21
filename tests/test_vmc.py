"""
Tests for VmcSender.

Covers:
- Bind UDP socket on loopback port 0, send one frame, parse packets.
- Assert addresses and values.
- 6 parameters -> 7 messages (6 Val + 1 Apply).
"""

import socket
import struct

from d_avatar.vmc import VmcSender

# Constants
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
    # OSC message format:
    # address pattern (null-terminated string)
    # type tag string (null-terminated, starts with ',')
    # arguments (4-byte aligned)
    idx = 0
    # Parse address
    addr_end = data.find(b'\x00', idx)
    if addr_end == -1:
        return '', []
    address = data[idx:addr_end].decode('utf-8')
    idx = addr_end + 1
    # Align to 4 bytes
    idx = (idx + 3) & ~3

    # Parse type tag
    if idx >= len(data) or data[idx] != ord(','):
        return address, []
    tag_end = data.find(b'\x00', idx)
    if tag_end == -1:
        return address, []
    type_tags = data[idx + 1 : tag_end].decode('utf-8')
    idx = tag_end + 1
    idx = (idx + 3) & ~3

    # Parse arguments
    args = []
    for tag in type_tags:
        if tag == 's':  # string
            str_end = data.find(b'\x00', idx)
            if str_end == -1:
                break
            args.append(data[idx:str_end].decode('utf-8'))
            idx = str_end + 1
            idx = (idx + 3) & ~3
        elif tag == 'f':  # float32
            if idx + 4 <= len(data):
                val = struct.unpack('>f', data[idx : idx + 4])[0]
                args.append(val)
                idx += 4
        elif tag == 'i':  # int32
            if idx + 4 <= len(data):
                val = struct.unpack('>i', data[idx : idx + 4])[0]
                args.append(val)
                idx += 4
        else:
            # Skip unknown
            pass
    return address, args


def test_vmc_sender_import():
    """VmcSender can be instantiated when python-osc is available."""
    sender = VmcSender()
    assert sender.host == '127.0.0.1'
    assert sender.port == _DEFAULT_PORT
    sender.close()


def test_vmc_sender_send_one_frame():
    """Send one frame with 6 parameters -> 7 messages (6 Val + 1 Apply)."""
    # Bind UDP socket on loopback port 0
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

    # Receive all messages (7 expected)
    messages = []
    for _ in range(_EXPECTED_TOTAL):
        try:
            data, _ = sock.recvfrom(4096)
            messages.append(data)
        except TimeoutError:
            break

    sender.close()
    sock.close()

    # Should have received 7 messages
    assert len(messages) == _EXPECTED_TOTAL, (
        f'Expected {_EXPECTED_TOTAL} messages, got {len(messages)}'
    )

    # Parse and verify
    val_count = 0
    apply_count = 0
    for data in messages:
        address, args = _parse_osc_message(data)
        if address == '/VMC/Ext/Blend/Val':
            val_count += 1
            # Verify arguments: name (string), value (float)
            assert len(args) == 2
            assert isinstance(args[0], str)
            assert isinstance(args[1], float)
        elif address == '/VMC/Ext/Blend/Apply':
            apply_count += 1
            # Apply has no arguments
            assert len(args) == 0

    assert val_count == _EXPECTED_VAL_COUNT, (
        f'Expected {_EXPECTED_VAL_COUNT} Val messages, got {val_count}'
    )
    assert apply_count == _EXPECTED_APPLY_COUNT, (
        f'Expected {_EXPECTED_APPLY_COUNT} Apply message, got {apply_count}'
    )


def test_vmc_sender_custom_host_port():
    """VmcSender accepts custom host and port."""
    sender = VmcSender(host='192.168.1.100', port=_CUSTOM_PORT)
    assert sender.host == '192.168.1.100'
    assert sender.port == _CUSTOM_PORT
    sender.close()


def test_vmc_sender_context_manager():
    """VmcSender works as context manager."""
    with VmcSender() as sender:
        sender.send({'test': 1.0})
    # Socket should be closed


def test_vmc_sender_get_params():
    """get_params returns config."""
    sender = VmcSender(host='10.0.0.1', port=_OTHER_PORT)
    params = sender.get_params()
    assert params['host'] == '10.0.0.1'
    assert params['port'] == _OTHER_PORT
    sender.close()
