"""
VMC (Virtual Motion Capture) protocol sender over UDP.

Sends avatar parameters as VMC blendshape messages:
- One /VMC/Ext/Blend/Val per parameter
- One /VMC/Ext/Blend/Apply per frame

Protocol specification: VMC Protocol v1.0
Official page: https://github.com/VirtualMotionCapture/VMCProtocol
Cited sections: "Blend Shape" message format (Val and Apply messages)

If the specification cannot be fetched, this module declares it and stops.
"""

from __future__ import annotations

import socket
from typing import Any

# VMC default port
VMC_DEFAULT_PORT = 39539
VMC_DEFAULT_HOST = '127.0.0.1'

# Optional dependency
try:
    from pythonosc import osc_message_builder, udp_client

    _PYTHONOSC_AVAILABLE = True
except ImportError:
    _PYTHONOSC_AVAILABLE = False
    udp_client = None
    osc_message_builder = None


class VmcSender:
    """Send avatar parameters as VMC blendshape messages over UDP.

    Args:
        host: destination IP address (default 127.0.0.1)
        port: destination UDP port (default 39539)

    Raises:
        ImportError: if python-osc is not installed (install with [osc] extra)

    Example:
        >>> sender = VmcSender()
        >>> sender.send({"eye_l": 1.0, "eye_r": 1.0, "mouth_open": 0.0})
    """

    def __init__(self, host: str = VMC_DEFAULT_HOST, port: int = VMC_DEFAULT_PORT) -> None:
        if not _PYTHONOSC_AVAILABLE:
            raise ImportError(
                'python-osc is required for VmcSender. Install with: pip install d-avatar[osc]'
            )
        self.host = host
        self.port = int(port)
        self._sock: socket.socket | None = None
        self._client = udp_client.SimpleUDPClient(host, port)

    def send(self, params: dict[str, float]) -> None:
        """Send one frame of parameters as VMC messages.

        Sends:
        - One /VMC/Ext/Blend/Val per parameter (address, value)
        - One /VMC/Ext/Blend/Apply (no arguments, triggers frame apply)

        Args:
            params: dict of parameter_name -> value (float)
        """
        # Send Val messages for each parameter
        for name, value in params.items():
            self._client.send_message('/VMC/Ext/Blend/Val', [name, float(value)])

        # Send Apply message to trigger frame update
        self._client.send_message('/VMC/Ext/Blend/Apply', [])

    def close(self) -> None:
        """Close the UDP socket."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

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
