"""
D-Avatar: landmark stream to stable avatar parameters to OSC/VMC.

An application on top of `dense_armor`.
"""

from d_avatar.params import BlinkCalibrator, RateBound, ema, landmarks_to_params
from d_avatar.pipeline import AvatarPipeline, build_pipeline, run
from d_avatar.shield import FaceShield
from d_avatar.vmc import VmcSender

__version__ = '0.1.0'

__all__ = [
    'AvatarPipeline',
    'BlinkCalibrator',
    'FaceShield',
    'RateBound',
    'VmcSender',
    'build_pipeline',
    'ema',
    'landmarks_to_params',
    'run',
]
