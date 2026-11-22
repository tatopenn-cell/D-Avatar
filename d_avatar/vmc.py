"""
VMC (Virtual Motion Capture) blendshape sender over UDP.

Sends one frame of named parameters as:
- one /VMC/Ext/Blend/Val message per parameter (string name, float value)
- one /VMC/Ext/Blend/Apply message per frame (no arguments)

Protocol: https://github.com/VirtualMotionCapture/VMCProtocol. The message layout above is
not checked against the specification in this repository. The parameter names are the
D-Avatar names, not the VRM blendshape presets, and head angles are not sent as bones.
"""

from __future__ import annotations

from typing import Any

try:
    from pythonosc import udp_client

    _PYTHONOSC_AVAILABLE = True
except ImportError:
    _PYTHONOSC_AVAILABLE = False
    udp_client = None

VMC_DEFAULT_PORT = 39539
VMC_DEFAULT_HOST = '127.0.0.1'


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
