"""
glyomo_modules.py
Complete implementation of GL-YOMO paper algorithms:
- Enhanced YOLO modules: Ghost, CPAM, TFE, SSFF
- Motion Detection: Optical flow, Template matching (NCC), Kalman filter
- Global-Local collaborative detection strategy
- ROI adaptive update mechanism

Reference: "Real-Time Detection for Small UAVs: Combining YOLO and Multi-frame Motion Analysis"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field


# ==============================================================================
# YOLO Enhancement Modules (from GL-YOMO paper)
# ==============================================================================

class GhostConv(nn.Module):
    """
    Ghost Convolution: generates more features from cheap operations.
    From GL-YOMO paper - reduces computational complexity significantly.
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c1, c2 = max(1, int(c1)), max(1, int(c2))
        c_ = max(1, c2 // 2)  # hidden channels (primary conv output)
        self.cv1 = nn.Conv2d(c1, c_, k, s, k // 2, groups=g, bias=False)
        self.cv2 = nn.Conv2d(c_, c_, 5, 1, 2, groups=c_, bias=False)  # cheap operations
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        return torch.cat([y, self.act(self.bn2(self.cv2(y)))], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck with residual connection"""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c1, c2 = max(1, int(c1)), max(1, int(c2))
        c_ = max(1, c2 // 2)
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            nn.Conv2d(c_, c_, k, s, k // 2, groups=c_, bias=False) if s == 2 else nn.Identity(),
            nn.BatchNorm2d(c_) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(c1, c1, k, s, k // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Conv2d(c1, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2)
        ) if s == 2 or c1 != c2 else nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C3Ghost(nn.Module):
    """C3 module with GhostBottleneck - lightweight and efficient"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c1, c2 = max(1, int(c1)), max(1, int(c2))
        n = max(1, int(n))
        c_ = max(1, int(c2 * e))  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, bias=False)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))))


class CPAM(nn.Module):
    """
    Channel and Position Attention Module (from GL-YOMO/ASF-YOLO).
    - Channel attention without dimension reduction
    - Position attention along horizontal and vertical axes
    Memory-optimized: uses downsampled position attention for large feature maps.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        channels = max(1, int(channels))
        reduction = max(1, int(reduction))
        
        # Channel attention (no dimension reduction as per paper)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_ch = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_ch, channels, 1, bias=False)
        )
        self.sigmoid_c = nn.Sigmoid()
        
        # Position attention (horizontal + vertical axes)
        query_ch = max(1, channels // 8)
        self.conv_query = nn.Conv2d(channels, query_ch, 1)
        self.conv_key = nn.Conv2d(channels, query_ch, 1)
        self.conv_value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        # Downsampling for large feature maps (memory optimization)
        self.max_spatial = 32  # Max spatial size for attention (32x32 = 1024 positions)
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_c(avg_out + max_out)
        x_c = x * channel_att
        
        # Position attention with memory optimization
        if h * w > self.max_spatial * self.max_spatial:
            # Downsample for attention computation, then upsample
            scale = self.max_spatial / max(h, w)
            new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
            x_down = F.interpolate(x_c, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            query = self.conv_query(x_down).view(b, -1, new_h * new_w).permute(0, 2, 1)
            key = self.conv_key(x_down).view(b, -1, new_h * new_w)
            energy = torch.bmm(query, key)
            attention = self.softmax(energy)
            value = self.conv_value(x_down).view(b, -1, new_h * new_w)
            out = torch.bmm(value, attention.permute(0, 2, 1))
            out = out.view(b, c, new_h, new_w)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        else:
            # Original full attention for small feature maps
            query = self.conv_query(x_c).view(b, -1, h * w).permute(0, 2, 1)
            key = self.conv_key(x_c).view(b, -1, h * w)
            energy = torch.bmm(query, key)
            attention = self.softmax(energy)
            value = self.conv_value(x_c).view(b, -1, h * w)
            out = torch.bmm(value, attention.permute(0, 2, 1))
            out = out.view(b, c, h, w)
        
        return self.gamma * out + x


class TFE(nn.Module):
    """
    Triple Feature Encoder (from ASF-YOLO, used in GL-YOMO).
    Fuses feature maps at large/medium/small scales, preserving fine-grained info.
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        if c2 is None:
            c2 = c1
        c1, c2 = max(1, int(c1)), max(1, int(c2))
        c_ = max(1, c2 // 3)
        
        # Three parallel branches with different kernel sizes
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)  # 1x1
        self.cv2 = nn.Conv2d(c1, c_, 3, 1, 1, bias=False)  # 3x3
        self.cv3 = nn.Conv2d(c1, c_, 5, 1, 2, bias=False)  # 5x5
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c_)
        self.bn3 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        
        # Final projection
        self.cv_out = nn.Conv2d(c_ * 3, c2, 1, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(c2)
    
    def forward(self, x):
        x1 = self.act(self.bn1(self.cv1(x)))
        x2 = self.act(self.bn2(self.cv2(x)))
        x3 = self.act(self.bn3(self.cv3(x)))
        out = torch.cat([x1, x2, x3], dim=1)
        return self.act(self.bn_out(self.cv_out(out)))


class SSFF(nn.Module):
    """
    Scale Sequence Feature Fusion (from ASF-YOLO, used in GL-YOMO).
    Uses Gaussian smoothing with progressively increasing sigma,
    then 3D convolution for scale-space processing.
    """
    def __init__(self, c1, c2=None, num_scales=3):
        super().__init__()
        if c2 is None:
            c2 = c1
        c1, c2 = max(1, int(c1)), max(1, int(c2))
        
        self.num_scales = num_scales
        self.sigmas = [0.5, 1.0, 2.0]  # progressively increasing sigma
        
        # 3D convolution for scale-space processing
        self.conv3d = nn.Conv3d(c1, c2, kernel_size=(num_scales, 3, 3), 
                               stride=1, padding=(0, 1, 1), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def _gaussian_blur(self, x, sigma):
        """Apply Gaussian blur with given sigma"""
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Separable convolution
        kernel = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
        kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)
        
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
    
    def forward(self, x):
        # Apply Gaussian at multiple scales
        scales = []
        for sigma in self.sigmas:
            blurred = self._gaussian_blur(x, sigma)
            scales.append(blurred)
        
        # Stack along new dimension: (B, C, num_scales, H, W)
        stacked = torch.stack(scales, dim=2)
        
        # 3D conv: (B, C, num_scales, H, W) -> (B, C_out, 1, H, W)
        out = self.conv3d(stacked)
        out = out.squeeze(2)  # (B, C_out, H, W)
        
        return self.act(self.bn(out))


class SAC(nn.Module):
    """Spatial Attention Channel Module"""
    def __init__(self, c1, c2):
        super().__init__()
        c1, c2 = max(1, int(c1)), max(1, int(c2))
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU()
        # Spatial Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.fc(out))
        return x * out


class C3_SAC(nn.Module):
    """C3 module with SAC attention blocks"""
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        c1, c2 = max(1, int(c1)), max(1, int(c2))
        n = max(1, int(n))
        self.shortcut = shortcut and c1 == c2
        
        c_ = max(1, c2 // 2)
        self.cv1 = nn.Conv2d(c1, c_, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, bias=False)
        self.sac = nn.Sequential(*[SAC(c_, c_) for _ in range(n)])
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
        if self.shortcut and c1 != c2:
            self.shortcut_conv = nn.Conv2d(c1, c2, 1, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(c2)
        else:
            self.shortcut_conv = None

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.sac(self.cv2(x))
        out = self.act(self.bn(self.cv3(torch.cat((x1, x2), dim=1))))
        
        if self.shortcut:
            if self.shortcut_conv is not None:
                out = out + self.shortcut_bn(self.shortcut_conv(x))
            else:
                out = out + x
        return out


class EMA(nn.Module):
    """Exponential Moving Average Module"""
    def __init__(self, channels, beta=0.99):
        super().__init__()
        channels = max(1, int(channels))
        self.channels = channels
        self.beta = beta
        self.register_buffer('ema', torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean([0, 2, 3], keepdim=True).detach()
            self.ema = self.beta * self.ema + (1 - self.beta) * batch_mean
        return x + self.ema


class AdaptiveConcat(nn.Module):
    """Adaptive Concatenation with channel projection"""
    def __init__(self, c2):
        super().__init__()
        c2 = max(1, int(c2))
        self.c2 = c2
        self.conv = None

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            concat_dim = sum([xi.shape[1] for xi in x])
            if self.conv is None or self.conv.in_channels != concat_dim:
                self.conv = nn.Conv2d(concat_dim, self.c2, 1, bias=False).to(x[0].device).to(x[0].dtype)
            x = torch.cat(x, dim=1)
        else:
            if self.conv is None or self.conv.in_channels != x.shape[1]:
                self.conv = nn.Conv2d(x.shape[1], self.c2, 1, bias=False).to(x.device).to(x.dtype)
        return self.conv(x)


class DyHead(nn.Module):
    """Dynamic Head for multi-scale detection"""
    def __init__(self, nc):
        super().__init__()
        self.nc = nc
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8, 16, 32, 64], dtype=torch.float32)
        self.conv3 = nn.Conv2d(64, self.no, 1)
        self.conv4 = nn.Conv2d(128, self.no, 1)
        self.conv5 = nn.Conv2d(256, self.no, 1)

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3
        x3, x4, x5 = x
        return [self.conv3(x3), self.conv4(x4), self.conv5(x5)]


# ==============================================================================
# Motion Detection Components (from GL-YOMO paper)
# ==============================================================================

class KalmanFilter8State:
    """
    8-state Kalman Filter for UAV tracking.
    State: [x, y, w, h, vx, vy, vw, vh]
    Measurement: [x, y, w, h]
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0
        
        # Measurement matrix
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        
        # Noise covariances
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        self.initialized = False
    
    def initialize(self, bbox: np.ndarray):
        """Initialize with [x, y, w, h]"""
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost[:4, 0] = bbox
        self.initialized = True
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        pred = self.kf.predict()
        return pred[:4, 0]
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update with measurement"""
        meas = measurement.reshape(4, 1).astype(np.float32)
        self.kf.correct(meas)
        return self.kf.statePost[:4, 0]
    
    @property
    def state_bbox(self) -> np.ndarray:
        return self.kf.statePost[:4, 0]


def estimate_homography(src_gray: np.ndarray, dst_gray: np.ndarray,
                       max_features: int = 500) -> Optional[np.ndarray]:
    """
    Estimate 2D perspective transformation using ORB + RANSAC.
    Used to compensate for camera motion.
    """
    if src_gray.shape != dst_gray.shape:
        return None
    
    orb = cv2.ORB_create(max_features)
    kp1, desc1 = orb.detectAndCompute(src_gray, None)
    kp2, desc2 = orb.detectAndCompute(dst_gray, None)
    
    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    if len(matches) < 8:
        return None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def pyramidal_lk_flow(prev_gray: np.ndarray, curr_gray: np.ndarray,
                     pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optical flow using pyramidal Lucas-Kanade"""
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **lk_params)
    return next_pts, status


def normalized_cross_correlation_multiscale(template: np.ndarray, image: np.ndarray,
                                           scales: List[float] = [0.7, 1.0, 1.3]
                                           ) -> Tuple[float, float, Tuple[int, int, int, int]]:
    """
    Multi-scale NCC as described in paper.
    Returns: (best_score, best_scale, best_bbox)
    """
    if template.size == 0 or image.size == 0:
        return 0.0, 1.0, (0, 0, 0, 0)
    
    best_score = 0.0
    best_scale = 1.0
    best_bbox = (0, 0, template.shape[1], template.shape[0])
    
    for scale in scales:
        new_w = max(1, int(template.shape[1] * scale))
        new_h = max(1, int(template.shape[0] * scale))
        
        if new_h > image.shape[0] or new_w > image.shape[1]:
            continue
        
        scaled_templ = cv2.resize(template, (new_w, new_h))
        
        try:
            res = cv2.matchTemplate(image, scaled_templ, cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # Normalize to 0-1 range
            Nc = (max_val + 1) / 2
            
            if Nc > best_score:
                best_score = Nc
                best_scale = scale
                best_bbox = (max_loc[0], max_loc[1], new_w, new_h)
        except:
            continue
    
    return best_score, best_scale, best_bbox


def compute_displacement_similarity(center_t2: np.ndarray, center_t1: np.ndarray,
                                   center_t: np.ndarray, img_size: Tuple[int, int]) -> float:
    """
    Compute displacement similarity Cd as in paper.
    Cd = 1 - (Δd_norm + Δθ_norm) / 2
    """
    # Displacement vectors
    disp_prev = center_t1 - center_t2  # d_{F_{t-1}}
    disp_curr = center_t - center_t1   # d_{F_t}
    
    # Magnitude difference (normalized by image diagonal)
    img_diag = np.sqrt(img_size[0]**2 + img_size[1]**2)
    d_prev = np.linalg.norm(disp_prev)
    d_curr = np.linalg.norm(disp_curr)
    delta_d_norm = abs(d_prev - d_curr) / (img_diag + 1e-6)
    
    # Direction difference (normalized by π)
    if d_prev > 1e-3 and d_curr > 1e-3:
        theta_prev = np.arctan2(disp_prev[1], disp_prev[0])
        theta_curr = np.arctan2(disp_curr[1], disp_curr[0])
        delta_theta = abs(theta_prev - theta_curr)
        # Handle wraparound
        delta_theta = min(delta_theta, 2 * np.pi - delta_theta)
        delta_theta_norm = delta_theta / np.pi
    else:
        delta_theta_norm = 0.0
    
    # Combined similarity
    Cd = 1.0 - (delta_d_norm + delta_theta_norm) / 2.0
    return max(0.0, min(1.0, Cd))


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two [x, y, w, h] boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


# ==============================================================================
# GL-YOMO Parameters Dataclass
# ==============================================================================

@dataclass
class GLYOMOParams:
    """Parameters for GL-YOMO detector (from paper)"""
    # Global-Local switching
    Ng: int = 30           # consecutive detections to switch to local
    Nl: int = 60           # consecutive misses to switch to global
    
    # ROI
    roi_base_size: int = 300
    Rs_ratio: float = 0.8   # Rs = 4/5 of ROI radius
    k1: float = 1.0         # ROI expansion rate per lost frame
    
    # Confidence thresholds
    tau_g: float = 0.3      # global detector threshold
    tau_l: float = 0.1      # local detector threshold
    
    # Motion detection
    delta_p: int = 50       # ROI padding for motion extraction
    grid_size: int = 7      # grid keypoints for optical flow
    thresh: int = 15        # frame difference threshold
    
    # Template matching weights
    k2: float = 0.6         # weight for NCC (Cc)
    k3: float = 0.4         # weight for displacement (Cd)
    scales: List[float] = field(default_factory=lambda: [0.7, 1.0, 1.3])
    
    # Motion acceptance
    motion_score_thresh: float = 0.5


# ==============================================================================
# GL-YOMO Motion Detector
# ==============================================================================

class MotionDetector:
    """
    Complete motion detector from GL-YOMO paper:
    1. Motion information extraction (optical flow + frame differencing)
    2. Template matching with weighted NCC
    3. Kalman filter verification
    """
    
    def __init__(self, params: GLYOMOParams = None):
        self.params = params or GLYOMOParams()
        self.kalman = KalmanFilter8State()
    
    def extract_motion_info(self, frames: List[np.ndarray], 
                           last_bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract motion information from ROI around last detection.
        Uses frame t-2 and frame t to capture more distinct motion.
        """
        frame_t2, frame_t1, frame_t = frames
        x, y, w, h = map(int, last_bbox)
        
        # Crop ROI around target
        cx, cy = x + w // 2, y + h // 2
        r = self.params.delta_p + max(w, h)
        
        x0 = max(0, cx - r)
        y0 = max(0, cy - r)
        x1 = min(frame_t.shape[1], cx + r)
        y1 = min(frame_t.shape[0], cy + r)
        
        if x1 <= x0 or y1 <= y0:
            return None, None, None
        
        roi_t2 = frame_t2[y0:y1, x0:x1]
        roi_t1 = frame_t1[y0:y1, x0:x1]
        roi_t = frame_t[y0:y1, x0:x1]
        
        # Convert to grayscale
        gray_t2 = cv2.cvtColor(roi_t2, cv2.COLOR_BGR2GRAY) if len(roi_t2.shape) == 3 else roi_t2
        gray_t1 = cv2.cvtColor(roi_t1, cv2.COLOR_BGR2GRAY) if len(roi_t1.shape) == 3 else roi_t1
        gray_t = cv2.cvtColor(roi_t, cv2.COLOR_BGR2GRAY) if len(roi_t.shape) == 3 else roi_t
        
        return (gray_t2, gray_t1, gray_t), (x0, y0, x1 - x0, y1 - y0), roi_t1
    
    def detect(self, frames: List[np.ndarray], last_bbox: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Main motion detection pipeline.
        Returns: (detected_bbox, confidence_score)
        """
        if len(frames) != 3:
            return None, 0.0
        
        # Step 1: Extract motion info
        result = self.extract_motion_info(frames, last_bbox)
        if result[0] is None:
            return None, 0.0
        
        (gray_t2, gray_t1, gray_t), roi_coords, roi_t1_color = result
        x0, y0, rw, rh = roi_coords
        
        # Step 2: Geometric alignment using homography
        H = estimate_homography(gray_t2, gray_t)
        if H is not None:
            try:
                gray_t2_aligned = cv2.warpPerspective(gray_t2, H, (gray_t.shape[1], gray_t.shape[0]))
            except:
                gray_t2_aligned = gray_t2
        else:
            gray_t2_aligned = gray_t2
        
        # Step 3: Optical flow on grid points
        gx = np.linspace(0, gray_t.shape[1] - 1, num=self.params.grid_size, dtype=np.int32)
        gy = np.linspace(0, gray_t.shape[0] - 1, num=self.params.grid_size, dtype=np.int32)
        pts = np.array([[xx, yy] for yy in gy for xx in gx], dtype=np.float32)
        next_pts, status = pyramidal_lk_flow(gray_t2_aligned, gray_t, pts)
        
        # Step 4: Frame difference mask
        diff = cv2.absdiff(gray_t, gray_t2_aligned)
        _, mask = cv2.threshold(diff, self.params.thresh, 255, cv2.THRESH_BINARY)
        
        # Morphological post-processing
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Step 5: Template matching
        x, y, w, h = map(int, last_bbox)
        templ_x = max(0, x - x0)
        templ_y = max(0, y - y0)
        templ_w = min(w, gray_t1.shape[1] - templ_x)
        templ_h = min(h, gray_t1.shape[0] - templ_y)
        
        if templ_w <= 0 or templ_h <= 0:
            return None, 0.0
        
        template = gray_t1[templ_y:templ_y + templ_h, templ_x:templ_x + templ_w]
        if template.size == 0:
            return None, 0.0
        
        # Multi-scale NCC
        Cc, best_scale, match_bbox = normalized_cross_correlation_multiscale(
            template, gray_t, self.params.scales
        )
        
        if Cc < 0.3:  # Skip low correlations
            return None, 0.0
        
        bx, by, bw, bh = match_bbox
        
        # Step 6: Displacement similarity
        center_t2 = np.array([templ_x + templ_w / 2, templ_y + templ_h / 2])
        center_t1 = center_t2.copy()
        center_t = np.array([bx + bw / 2, by + bh / 2])
        img_size = (frames[0].shape[1], frames[0].shape[0])
        
        Cd = compute_displacement_similarity(center_t2, center_t1, center_t, img_size)
        
        # Step 7: Weighted matching score
        Cw = self.params.k2 * Cc + self.params.k3 * Cd
        
        # Step 8: Convert back to full-frame coordinates
        detected_bbox = np.array([x0 + bx, y0 + by, bw, bh], dtype=np.float32)
        
        # Step 9: Kalman verification
        if self.kalman.initialized:
            pred_bbox = self.kalman.predict()
            iou = compute_iou(detected_bbox, pred_bbox)
            
            if iou > 0:
                # Match accepted, update Kalman
                self.kalman.update(detected_bbox)
                return detected_bbox, Cw
            else:
                # Use Kalman prediction instead
                return pred_bbox, Cw * 0.5  # Lower confidence for prediction
        else:
            # Initialize Kalman with first detection
            self.kalman.initialize(detected_bbox)
            return detected_bbox, Cw
        
        return detected_bbox, Cw


# ==============================================================================
# GL-YOMO Detector (combines YOLO + Motion + Global-Local strategy)
# ==============================================================================

class GLYOMODetector:
    """
    Complete GL-YOMO detector implementation.
    Combines YOLO detection with motion detection using global-local strategy.
    """
    
    def __init__(self, yolo_model, params: GLYOMOParams = None):
        """
        Args:
            yolo_model: Ultralytics YOLO model
            params: GLYOMOParams configuration
        """
        self.yolo_model = yolo_model
        self.params = params or GLYOMOParams()
        self.motion_detector = MotionDetector(self.params)
        
        # State variables
        self.reset()
    
    def reset(self):
        """Reset detector state"""
        self.mode = 'global'  # 'global' or 'local'
        self.Ng_counter = 0   # consecutive global detections
        self.Nl_counter = 0   # consecutive local misses
        self.Flost = 0        # frames lost counter
        self.roi = None       # [x, y, w, h]
        self.last_bbox = None
        self.frame_buffer = []
        self.motion_detector.kalman = KalmanFilter8State()  # Reset Kalman
    
    def _update_roi(self, bbox: np.ndarray, frame_shape: Tuple[int, int]):
        """Update ROI based on detection"""
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        
        size = self.params.roi_base_size
        rx = max(0, int(cx - size / 2))
        ry = max(0, int(cy - size / 2))
        rw = min(size, frame_shape[1] - rx)
        rh = min(size, frame_shape[0] - ry)
        
        self.roi = np.array([rx, ry, rw, rh], dtype=np.float32)
    
    def _expand_roi(self, frame_shape: Tuple[int, int]):
        """Expand ROI based on lost frames: Rsize = 300 + k1 * Flost"""
        if self.roi is None:
            return
        
        expansion = self.params.k1 * self.Flost
        cx = self.roi[0] + self.roi[2] / 2
        cy = self.roi[1] + self.roi[3] / 2
        
        new_size = self.params.roi_base_size + expansion
        rx = max(0, int(cx - new_size / 2))
        ry = max(0, int(cy - new_size / 2))
        rw = min(int(new_size), frame_shape[1] - rx)
        rh = min(int(new_size), frame_shape[0] - ry)
        
        self.roi = np.array([rx, ry, rw, rh], dtype=np.float32)
    
    def _should_update_roi(self, bbox: np.ndarray) -> bool:
        """Check if target moved outside Rs boundary"""
        if self.roi is None:
            return True
        
        # Center of detection
        det_cx = bbox[0] + bbox[2] / 2
        det_cy = bbox[1] + bbox[3] / 2
        
        # Center and radius of ROI
        roi_cx = self.roi[0] + self.roi[2] / 2
        roi_cy = self.roi[1] + self.roi[3] / 2
        roi_radius = min(self.roi[2], self.roi[3]) / 2
        Rs = roi_radius * self.params.Rs_ratio
        
        # Distance from detection to ROI center
        dist = np.sqrt((det_cx - roi_cx)**2 + (det_cy - roi_cy)**2)
        
        return dist > Rs
    
    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> Dict:
        """
        Process single frame with GL-YOMO strategy.
        
        Returns:
            dict with keys: 'bbox', 'conf', 'mode', 'source' ('yolo' or 'motion')
        """
        h, w = frame.shape[:2]
        
        # Update frame buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
        
        # Determine threshold
        if conf_threshold is None:
            threshold = self.params.tau_l if self.mode == 'local' else self.params.tau_g
        else:
            threshold = conf_threshold
        
        # Prepare input frame
        if self.mode == 'local' and self.roi is not None:
            rx, ry, rw, rh = map(int, self.roi)
            rx = max(0, min(rx, w - 1))
            ry = max(0, min(ry, h - 1))
            rx2 = min(rx + rw, w)
            ry2 = min(ry + rh, h)
            
            if rx2 > rx and ry2 > ry:
                crop = frame[ry:ry2, rx:rx2]
                inference_frame = cv2.resize(crop, (640, 640))
                scale_x = (rx2 - rx) / 640
                scale_y = (ry2 - ry) / 640
                offset = (rx, ry)
            else:
                inference_frame = frame
                scale_x = scale_y = 1.0
                offset = (0, 0)
        else:
            inference_frame = frame
            scale_x = scale_y = 1.0
            offset = (0, 0)
        
        # YOLO inference
        detection = None
        source = 'none'
        
        try:
            results = self.yolo_model.predict(source=inference_frame, conf=threshold, verbose=False)
            if results and len(results) > 0:
                boxes = getattr(results[0], 'boxes', None)
                if boxes is not None and len(boxes) > 0:
                    # Get best detection
                    best_conf = 0.0
                    best_box = None
                    
                    for box in boxes:
                        try:
                            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            if conf > best_conf:
                                best_conf = conf
                                xy = box.xyxy[0] if hasattr(box.xyxy, '__getitem__') else box.xyxy
                                best_box = xy
                        except:
                            continue
                    
                    if best_box is not None and best_conf >= threshold:
                        bx1, by1, bx2, by2 = map(float, best_box)
                        
                        # Scale back if local mode
                        if self.mode == 'local' and self.roi is not None:
                            bx1 = bx1 * scale_x + offset[0]
                            by1 = by1 * scale_y + offset[1]
                            bx2 = bx2 * scale_x + offset[0]
                            by2 = by2 * scale_y + offset[1]
                        
                        detection = {
                            'bbox': np.array([bx1, by1, bx2 - bx1, by2 - by1], dtype=np.float32),
                            'conf': best_conf
                        }
                        source = 'yolo'
        except Exception as e:
            pass
        
        # If no YOLO detection, try motion detector
        if detection is None and len(self.frame_buffer) >= 3 and self.last_bbox is not None:
            motion_bbox, motion_score = self.motion_detector.detect(self.frame_buffer, self.last_bbox)
            if motion_bbox is not None and motion_score >= self.params.motion_score_thresh:
                detection = {
                    'bbox': motion_bbox,
                    'conf': motion_score
                }
                source = 'motion'
        
        # Update state based on detection result
        if detection is not None:
            current_bbox = detection['bbox']
            self.last_bbox = current_bbox
            self.Flost = 0
            
            # Update ROI
            if self._should_update_roi(current_bbox):
                self._update_roi(current_bbox, (h, w))
            
            # Update mode counters (Global-Local switching)
            if self.mode == 'global':
                self.Ng_counter += 1
                self.Nl_counter = 0
                if self.Ng_counter >= self.params.Ng:
                    self.mode = 'local'
            else:
                self.Nl_counter = 0
        else:
            self.Flost += 1
            
            if self.mode == 'local':
                self.Nl_counter += 1
                self._expand_roi((h, w))
                
                if self.Nl_counter >= self.params.Nl:
                    self.mode = 'global'
                    self.Ng_counter = 0
                    self.roi = None
        
        return {
            'bbox': detection['bbox'] if detection else None,
            'conf': detection['conf'] if detection else 0.0,
            'mode': self.mode,
            'source': source,
            'roi': self.roi.copy() if self.roi is not None else None,
            'Ng': self.Ng_counter,
            'Nl': self.Nl_counter,
            'Flost': self.Flost
        }


# Register modules for YOLO
__all__ = [
    'GhostConv', 'GhostBottleneck', 'C3Ghost',
    'CPAM', 'TFE', 'SSFF',
    'SAC', 'C3_SAC', 'EMA', 'AdaptiveConcat', 'DyHead',
    'KalmanFilter8State', 'MotionDetector', 'GLYOMODetector', 'GLYOMOParams',
    'compute_iou', 'estimate_homography', 'pyramidal_lk_flow',
    'normalized_cross_correlation_multiscale', 'compute_displacement_similarity'
]
