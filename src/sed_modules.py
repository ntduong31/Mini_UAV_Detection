"""
sed_modules.py
Re-exports all GL-YOMO modules from glyomo_modules.py for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import all modules from the main implementation
from glyomo_modules import (
    # YOLO Enhancement modules
    GhostConv,
    GhostBottleneck, 
    C3Ghost,
    CPAM,
    TFE,
    SSFF,
    SAC,
    C3_SAC,
    EMA,
    AdaptiveConcat,
    DyHead,
    
    # Motion detection components
    KalmanFilter8State,
    MotionDetector,
    GLYOMODetector,
    GLYOMOParams,
    
    # Utility functions
    compute_iou,
    estimate_homography,
    pyramidal_lk_flow,
    normalized_cross_correlation_multiscale,
    compute_displacement_similarity,
)

# Backward compatibility alias
SimpleKalman = KalmanFilter8State

# Register into global scope for YOLO parser
globals()['SAC'] = SAC
globals()['C3_SAC'] = C3_SAC
globals()['EMA'] = EMA
globals()['AdaptiveConcat'] = AdaptiveConcat
globals()['DyHead'] = DyHead
globals()['GhostConv'] = GhostConv
globals()['C3Ghost'] = C3Ghost
globals()['GhostBottleneck'] = GhostBottleneck
globals()['CPAM'] = CPAM
globals()['TFE'] = TFE
globals()['SSFF'] = SSFF

__all__ = [
    'GhostConv', 'GhostBottleneck', 'C3Ghost',
    'CPAM', 'TFE', 'SSFF',
    'SAC', 'C3_SAC', 'EMA', 'AdaptiveConcat', 'DyHead',
    'KalmanFilter8State', 'SimpleKalman', 'MotionDetector', 
    'GLYOMODetector', 'GLYOMOParams',
    'compute_iou', 'estimate_homography', 'pyramidal_lk_flow',
    'normalized_cross_correlation_multiscale', 'compute_displacement_similarity'
]
