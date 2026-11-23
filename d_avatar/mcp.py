"""MCP tools for D-Avatar: expose avatar parameters and presets to a language model.

This module follows the structure of Dense-Armor's MCP server (dense_armor.mcp_server.tools)
but exposes D-Avatar specific functionality. The tools are callable directly without
the MCP library for testing.

Tools:
- get_params(): returns current avatar parameters
- set_limits(max_step, max_step_deg): sets rate limits on parameters
- set_preset(name): applies a named preset (calm, lively)
"""

from __future__ import annotations

from typing import Any

from d_avatar.pipeline import ParamsStep

# Presets: named sets of limits
PRESETS = {
    'calm': {'max_step': 0.05, 'max_step_deg': 2.0},
    'lively': {'max_step': 0.2, 'max_step_deg': 10.0},
    'default': {'max_step': 0.1, 'max_step_deg': 5.0},
}


class DAvatarMCPTools:
    """MCP-style tools for controlling a D-Avatar pipeline.

    This class holds a reference to a running ParamsStep (or AvatarPipeline)
    and exposes tools that an LLM can call via MCP. For testing, the methods
    can be called directly without the MCP library.

    Example:
        >>> params_step = ParamsStep()
        >>> tools = DAvatarMCPTools(params_step)
        >>> tools.get_params()
        >>> tools.set_preset("lively")
    """

    def __init__(self, params_step: ParamsStep) -> None:
        self.params_step = params_step
        self._preset_limits = dict(PRESETS)

    def get_params(self) -> dict[str, Any]:
        """Return the current avatar parameters and limits.

        Returns:
            dict with keys:
            - "params": current parameter values (from last frame)
            - "limits": current rate limits {max_step, max_step_deg, per_name, head_limits}
            - "calibrators": state of BlinkCalibrator and GazeCalibrator
        """
        bound = self.params_step.bound
        calibrator = self.params_step.calibrator
        gaze_cal = self.params_step.gaze_calibrator

        # Extract head angle limits from per_name
        head_limits = {k: bound.per_name.get(k) for k in ('head_yaw', 'head_pitch', 'head_roll')}
        # max_step_deg is the max of head angle limits (for backward compatibility)
        head_values = [v for v in head_limits.values() if v is not None]
        max_step_deg = max(head_values) if head_values else 0.0

        return {
            'params': dict(self.params_step._ema) | dict(self.params_step._gaze_ema),
            'limits': {
                'max_step': bound.max_step,
                'max_step_deg': max_step_deg,
                'per_name': dict(bound.per_name),
                'head_limits': head_limits,
            },
            'calibrators': {
                'blink': {
                    'open_ear': calibrator.open_ear(),
                    'centroids': calibrator._c.tolist() if calibrator._c is not None else None,
                },
                'gaze': {
                    'neutral': gaze_cal.neutral(),
                    'buffer_size': gaze_cal._n,
                },
            },
        }

    def set_limits(
        self, max_step: float | None = None, max_step_deg: float | None = None
    ) -> dict[str, Any]:
        """Update the rate limits for parameter changes.

        Args:
            max_step: maximum change per frame for [0, 1] parameters.
            max_step_deg: maximum change per frame for head angles (degrees).

        Returns:
            dict with updated limits.
        """
        bound = self.params_step.bound
        if max_step is not None:
            old_max_step = bound.max_step
            bound.max_step = float(max_step)
            # Update per_name entries that use the old default max_step
            for k in bound.per_name:
                if bound.per_name[k] == old_max_step:
                    bound.per_name[k] = float(max_step)
        if max_step_deg is not None:
            # Update head angle limits
            for k in ('head_yaw', 'head_pitch', 'head_roll'):
                if k in bound.per_name:
                    bound.per_name[k] = float(max_step_deg)

        head_limits = {k: bound.per_name.get(k) for k in ('head_yaw', 'head_pitch', 'head_roll')}
        head_values = [v for v in head_limits.values() if v is not None]
        max_step_deg_val = max(head_values) if head_values else 0.0

        return {
            'max_step': bound.max_step,
            'max_step_deg': max_step_deg_val,
            'per_name': dict(bound.per_name),
            'head_limits': head_limits,
        }

    def set_preset(self, name: str) -> dict[str, Any]:
        """Apply a named preset of rate limits.

        Args:
            name: preset name ("calm", "lively", "default").

        Returns:
            dict with applied limits.

        Raises:
            ValueError: if preset name is unknown.
        """
        if name not in self._preset_limits:
            available = list(self._preset_limits.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")

        preset = self._preset_limits[name]
        return self.set_limits(preset['max_step'], preset['max_step_deg'])

    def list_presets(self) -> dict[str, dict[str, float]]:
        """List all available presets with their limits."""
        return {k: dict(v) for k, v in self._preset_limits.items()}


# Standalone tool functions for MCP registration (can be called directly for testing)
def get_params(params_step: ParamsStep) -> dict[str, Any]:
    """MCP tool: get current avatar parameters and limits."""
    tools = DAvatarMCPTools(params_step)
    return tools.get_params()


def set_limits(
    params_step: ParamsStep,
    max_step: float | None = None,
    max_step_deg: float | None = None,
) -> dict[str, Any]:
    """MCP tool: set rate limits on parameters."""
    tools = DAvatarMCPTools(params_step)
    return tools.set_limits(max_step, max_step_deg)


def set_preset(params_step: ParamsStep, name: str) -> dict[str, Any]:
    """MCP tool: apply a named preset (calm, lively, default)."""
    tools = DAvatarMCPTools(params_step)
    return tools.set_preset(name)


def list_presets(params_step: ParamsStep) -> dict[str, dict[str, float]]:
    """MCP tool: list available presets."""
    tools = DAvatarMCPTools(params_step)
    return tools.list_presets()
