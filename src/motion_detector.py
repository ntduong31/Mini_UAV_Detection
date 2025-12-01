"""
motion_detector.py
Re-exports motion detection components from glyomo_modules.py for backward compatibility.
"""

from glyomo_modules import (
    KalmanFilter8State,
    MotionDetector,
    GLYOMODetector,
    GLYOMOParams,
    compute_iou,
    estimate_homography,
    pyramidal_lk_flow,
    normalized_cross_correlation_multiscale,
    compute_displacement_similarity,
)

# Backward compatibility aliases
SimpleKalman = KalmanFilter8State

def detect_motion(frames, last_bbox, params):
    """
    Wrapper function for backward compatibility.
    Uses MotionDetector.detect() internally.
    """
    detector_params = GLYOMOParams(
        delta_p=params.get('delta_p', 50),
        grid_size=params.get('grid_size', 7),
        thresh=params.get('thresh', 15),
        k2=params.get('k2', 0.6),
        k3=params.get('k3', 0.4),
        scales=params.get('scales', [0.8, 1.0, 1.2]),
        motion_score_thresh=params.get('motion_score_thresh', 0.5)
    )
    
    detector = MotionDetector(detector_params)
    return detector.detect(frames, last_bbox)

__all__ = [
    'SimpleKalman', 'KalmanFilter8State', 'MotionDetector',
    'GLYOMODetector', 'GLYOMOParams',
    'detect_motion', 'compute_iou',
    'estimate_homography', 'pyramidal_lk_flow',
    'normalized_cross_correlation_multiscale', 'compute_displacement_similarity'
]
