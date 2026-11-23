"""
VMC (Virtual Motion Capture) blendshape sender over UDP.

Sends one frame of named parameters as:
- one /VMC/Ext/Blend/Val message per parameter (string name, float value)
- one /VMC/Ext/Blend/Apply message per frame (no arguments)

``send_frame`` maps D-Avatar parameters to VRM blendshape names and head bone
pose, per the VMC protocol verified in task DA4 step A:
- Blink_L/Blink_R = 1 - eye openness (or 1.0 when blink == 1)
- A (mouth) = mouth_open
- Head bone = /VMC/Ext/Bone/Pos with quaternion (x, y, z, w) from Euler angles

Protocol: https://github.com/VirtualMotionCapture/VMCProtocol.
"""

from __future__ import annotations

import math
from typing import Any

try:
    from pythonosc import udp_client

    _PYTHONOSC_AVAILABLE = True
except ImportError:
    _PYTHONOSC_AVAILABLE = False
    udp_client = None

VMC_DEFAULT_PORT = 39539
VMC_DEFAULT_HOST = '127.0.0.1'

# D-Avatar param -> VRM blendshape key (VMC Perfect Sync keys, camelCase)
_BLENDSHAPE_MAP = {
    'eye_l': 'eyeBlinkLeft',
    'eye_r': 'eyeBlinkRight',
    'mouth_open': 'jawOpen',
}

_HEAD_BONE = 'Head'


class VmcSender:
    """Send avatar parameters as VMC blendshape messages over UDP.

    Args:
        host: destination IP address (default 127.0.0.1)
        port: destination UDP port (default 39539)

    Raises:
        ImportError: if python-osc is not installed (install with the [osc] extra)

    Example:
        >>> sender = VmcSender()
        >>> sender.send({"eye_l": 1.0, "eye_r": 1.0, "mouth_open": 0.0})
        >>> sender.close()
    """

    def __init__(self, host: str = VMC_DEFAULT_HOST, port: int = VMC_DEFAULT_PORT) -> None:
        if not _PYTHONOSC_AVAILABLE:
            raise ImportError(
                'python-osc is required for VmcSender. Install with: pip install "d-avatar[osc]"'
            )
        self.host = host
        self.port = int(port)
        self._client = udp_client.SimpleUDPClient(host, self.port)

    def send(self, params: dict[str, float]) -> None:
        """Send one frame: a Val message per parameter, then one Apply message."""
        for name, value in params.items():
            self._client.send_message('/VMC/Ext/Blend/Val', [name, float(value)])
        self._client.send_message('/VMC/Ext/Blend/Apply', [])

    def send_frame(self, params: dict[str, float]) -> None:
        """Send one frame mapped to VRM blendshape names and head bone pose.

        Maps D-Avatar parameters to VRM blendshape names and sends:
        - one /VMC/Ext/Blend/Val per blendshape (string name, float value)
        - one /VMC/Ext/Blend/Apply (no arguments)
        - one /VMC/Ext/Bone/Pos for the Head bone (string "Head", float posX/Y/Z,
          float rotX/Y/Z/W quaternion from yaw/pitch/roll degrees)

        Args:
            params: dict with keys head_yaw, head_pitch, head_roll (deg),
                mouth_open, eye_l, eye_r (0, 1), blink (0 or 1).
        """
        blink = float(params.get('blink', 0.0))
        eye_l = float(params.get('eye_l', 1.0))
        eye_r = float(params.get('eye_r', 1.0))
        mouth_open = float(params.get('mouth_open', 0.0))

        blendshapes: dict[str, float] = {}
        if blink == 1.0:
            blendshapes['eyeBlinkLeft'] = 1.0
            blendshapes['eyeBlinkRight'] = 1.0
        else:
            blendshapes['eyeBlinkLeft'] = 1.0 - eye_l
            blendshapes['eyeBlinkRight'] = 1.0 - eye_r
        blendshapes['jawOpen'] = mouth_open

        for name, value in blendshapes.items():
            self._client.send_message('/VMC/Ext/Blend/Val', [name, float(value)])
        self._client.send_message('/VMC/Ext/Blend/Apply', [])

        # Head bone pose from Euler angles (degrees) -> quaternion (x, y, z, w)
        yaw = math.radians(float(params.get('head_yaw', 0.0)))
        pitch = math.radians(float(params.get('head_pitch', 0.0)))
        roll = math.radians(float(params.get('head_roll', 0.0)))

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        self._client.send_message(
            '/VMC/Ext/Bone/Pos',
            [_HEAD_BONE, 0.0, 0.0, 0.0, float(qx), float(qy), float(qz), float(qw)],
        )

    def close(self) -> None:
        """Close the UDP socket."""
        self._client._sock.close()

    def __enter__(self) -> VmcSender:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.close()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {'host': self.host, 'port': self.port}
