"""
D-Avatar: landmark stream to stable avatar parameters to OSC/VMC.

An application on top of `dense_armor`.
"""

from d_avatar.capture import camera_stream_frames
from d_avatar.detector import FaceDetector
from d_avatar.gaze import GazeCalibrator, iris_displacement
from d_avatar.mcp import DAvatarMCPTools, get_params, list_presets, set_limits, set_preset
from d_avatar.mesh import iris_centres, mesh_to_68
from d_avatar.params import BlinkCalibrator, RateBound, ema, landmarks_to_params
from d_avatar.pipeline import AvatarPipeline, build_pipeline, run
from d_avatar.shield import FaceShield
from d_avatar.vmc import VmcSender

__version__ = '0.1.0'

__all__ = [
    'AvatarPipeline',
    'BlinkCalibrator',
    'DAvatarMCPTools',
    'FaceDetector',
    'FaceShield',
    'GazeCalibrator',
    'RateBound',
    'VmcSender',
    'build_pipeline',
    'camera_stream_frames',
    'ema',
    'get_params',
    'iris_centres',
    'iris_displacement',
    'landmarks_to_params',
    'list_presets',
    'mesh_to_68',
    'run',
    'set_limits',
    'set_preset',
]
