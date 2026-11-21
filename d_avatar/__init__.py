"""
D-Avatar: landmark stream to stable avatar parameters to OSC/VMC.

An application on top of `dense_armor`.
"""

__version__ = '0.1.0'

# Public API (items B–E will populate these)
__all__ = [
    # B. The shield
    'FaceShield',
    # C. The parameters
    'landmarks_to_params',
    'BlinkCalibrator',
    'ema',
    'RateBound',
    # D. The VMC sender
    'VmcSender',
    # E. The pipeline
    'build_pipeline',
    'run',
]
