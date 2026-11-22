"""Tests for D-Avatar MCP tools.

These tests call the tool functions directly with a fake pipeline,
without the MCP library.
"""

import numpy as np

from d_avatar.gaze import iris_displacement
from d_avatar.mcp import DAvatarMCPTools, get_params, list_presets, set_limits, set_preset
from d_avatar.params import eye_aspect_ratios
from d_avatar.pipeline import ParamsStep
from d_avatar.synthetic import synthetic_face_68


def test_get_params():
    """Test get_params returns current parameters and limits."""
    params_step = ParamsStep()
    tools = DAvatarMCPTools(params_step)

    result = tools.get_params()

    assert 'params' in result
    assert 'limits' in result
    assert 'calibrators' in result
    assert result['limits']['max_step'] == 0.1
    assert result['limits']['max_step_deg'] == 5.0
    assert 'blink' in result['calibrators']
    assert 'gaze' in result['calibrators']


def test_set_limits():
    """Test set_limits updates the rate bounds."""
    params_step = ParamsStep()
    tools = DAvatarMCPTools(params_step)

    # Change max_step
    result = tools.set_limits(max_step=0.05)
    assert result['max_step'] == 0.05
    assert params_step.bound.max_step == 0.05

    # Change max_step_deg (head angle limits)
    result = tools.set_limits(max_step_deg=10.0)
    assert result['max_step_deg'] == 10.0
    # Check head angle limits in per_name
    assert params_step.bound.per_name['head_yaw'] == 10.0
    assert params_step.bound.per_name['head_pitch'] == 10.0
    assert params_step.bound.per_name['head_roll'] == 10.0

    # Change both
    result = tools.set_limits(max_step=0.2, max_step_deg=3.0)
    assert result['max_step'] == 0.2
    assert result['max_step_deg'] == 3.0
    assert params_step.bound.per_name['head_yaw'] == 3.0


def test_set_preset():
    """Test set_preset applies named limit sets."""
    params_step = ParamsStep()
    tools = DAvatarMCPTools(params_step)

    # Test calm preset
    result = tools.set_preset('calm')
    assert result['max_step'] == 0.05
    assert result['max_step_deg'] == 2.0
    assert params_step.bound.max_step == 0.05
    assert params_step.bound.per_name['head_yaw'] == 2.0

    # Test lively preset
    result = tools.set_preset('lively')
    assert result['max_step'] == 0.2
    assert result['max_step_deg'] == 10.0

    # Test default preset
    result = tools.set_preset('default')
    assert result['max_step'] == 0.1
    assert result['max_step_deg'] == 5.0

    # Test unknown preset
    try:
        tools.set_preset('unknown')
        raise AssertionError('Should have raised ValueError')
    except ValueError as e:
        assert 'Unknown preset' in str(e)


def test_list_presets():
    """Test list_presets returns all available presets."""
    params_step = ParamsStep()
    tools = DAvatarMCPTools(params_step)

    presets = tools.list_presets()
    assert 'calm' in presets
    assert 'lively' in presets
    assert 'default' in presets
    assert presets['calm'] == {'max_step': 0.05, 'max_step_deg': 2.0}
    assert presets['lively'] == {'max_step': 0.2, 'max_step_deg': 10.0}
    assert presets['default'] == {'max_step': 0.1, 'max_step_deg': 5.0}


def test_standalone_functions():
    """Test the standalone tool functions (for MCP registration)."""
    params_step = ParamsStep()

    # get_params
    result = get_params(params_step)
    assert 'params' in result
    assert 'limits' in result

    # set_limits
    result = set_limits(params_step, max_step=0.15)
    assert result['max_step'] == 0.15

    # set_preset
    result = set_preset(params_step, 'calm')
    assert result['max_step'] == 0.05

    # list_presets
    result = list_presets(params_step)
    assert 'calm' in result


def test_preset_changes_head_limits():
    """Test that presets update head angle limits."""
    params_step = ParamsStep()
    tools = DAvatarMCPTools(params_step)

    # Lively preset should update head angles
    tools.set_preset('lively')
    bound = params_step.bound
    assert bound.per_name['head_yaw'] == 10.0
    assert bound.per_name['head_pitch'] == 10.0
    assert bound.per_name['head_roll'] == 10.0

    # Calm preset
    tools.set_preset('calm')
    assert bound.per_name['head_yaw'] == 2.0
    assert bound.per_name['head_pitch'] == 2.0
    assert bound.per_name['head_roll'] == 2.0


def test_calibrator_state_in_get_params():
    """Test that calibrator state is included in get_params."""
    params_step = ParamsStep()
    tools = DAvatarMCPTools(params_step)

    # Feed some data to calibrators
    base = synthetic_face_68()
    ear_l, ear_r = eye_aspect_ratios(base)
    ear = (ear_l + ear_r) / 2.0
    params_step.calibrator.update(ear)

    # Add iris data
    iris = np.zeros((2, 2))
    iris[0] = base[39]
    iris[1] = base[42]
    dx, dy = iris_displacement(iris[0], base[39], base[36], base[37], base[40])
    params_step.gaze_calibrator.update(dx, dy)

    result = tools.get_params()
    assert 'blink' in result['calibrators']
    assert 'gaze' in result['calibrators']
    assert result['calibrators']['blink']['open_ear'] > 0
    assert result['calibrators']['gaze']['neutral'] is not None


def test_mcp_tools_without_mcp_library():
    """Verify tools work without MCP library installed."""
    # This test passes if the above tests pass, since they don't import MCP
    params_step = ParamsStep()
    tools = DAvatarMCPTools(params_step)

    # All operations should work without MCP
    result = tools.get_params()
    assert 'params' in result

    result = tools.set_preset('lively')
    assert result['max_step'] == 0.2

    # The class should be importable and usable without mcp package
    assert DAvatarMCPTools is not None
