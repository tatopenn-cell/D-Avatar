"""
D-Avatar: landmark stream to stable avatar parameters to OSC/VMC.

An application on top of `dense_armor`.
"""

from d_avatar.capture import CameraSource
from d_avatar.detector import FaceDetector
from d_avatar.gaze import GazeCalibrator, iris_displacement
from d_avatar.mesh import iris_centres, mesh_to_68
from d_avatar.params import BlinkCalibrator, RateBound, ema, landmarks_to_params
from d_avatar.pipeline import AvatarPipeline, build_pipeline, run
from d_avatar.shield import FaceShield
from d_avatar.vmc import VmcSender

__version__ = '0.1.0'

__all__ = [
    'AvatarPipeline',
    'BlinkCalibrator',
    'CameraSource',
    'FaceDetector',
    'FaceShield',
    'GazeCalibrator',
    'RateBound',
    'VmcSender',
    'build_pipeline',
    'ema',
    'iris_centres',
    'iris_displacement',
    'landmarks_to_params',
    'mesh_to_68',
    'run',
]
