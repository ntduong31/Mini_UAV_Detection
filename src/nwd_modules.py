"""
nwd_modules.py
Implementation of Normalized Wasserstein Distance (NWD) for tiny object detection.

Reference: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
Paper: https://arxiv.org/abs/2110.13389

NWD replaces IoU metric in:
1. Label assignment (pos/neg sample selection)
2. Loss function (bbox regression)
3. NMS (Non-Maximum Suppression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def wasserstein_distance_2d_gaussian(bbox1, bbox2, eps=1e-7):
    """
    Compute Wasserstein Distance (W2) between two 2D Gaussian distributions
    derived from bounding boxes.
    
    For 2D Gaussians N(μ1, Σ1) and N(μ2, Σ2), the Wasserstein-2 distance is:
    W2^2 = ||μ1 - μ2||^2 + Tr(Σ1 + Σ2 - 2(Σ1^(1/2) * Σ2 * Σ1^(1/2))^(1/2))
    
    For diagonal covariance matrices (which we use for bounding boxes):
    Σ = diag(σ_x^2, σ_y^2) where σ_x = w/2, σ_y = h/2
    
    Args:
        bbox1: Tensor of shape (..., 4) in [x1, y1, x2, y2] format
        bbox2: Tensor of shape (..., 4) in [x1, y1, x2, y2] format
        eps: Small constant for numerical stability
        
    Returns:
        Wasserstein distance: Tensor of shape (...)
    """
    # Convert to center format
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2
    w1 = bbox1[..., 2] - bbox1[..., 0]
    h1 = bbox1[..., 3] - bbox1[..., 1]
    
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2
    w2 = bbox2[..., 2] - bbox2[..., 0]
    h2 = bbox2[..., 3] - bbox2[..., 1]
    
    # Center distance (mean difference)
    center_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    # Standard deviations (using w/2 and h/2 as per paper)
    sigma_x1 = w1 / 2
    sigma_y1 = h1 / 2
    sigma_x2 = w2 / 2
    sigma_y2 = h2 / 2
    
    # For diagonal covariance matrices, the formula simplifies to:
    # Tr(Σ1 + Σ2 - 2(Σ1^(1/2) * Σ2 * Σ1^(1/2))^(1/2))
    # = σ_x1^2 + σ_y1^2 + σ_x2^2 + σ_y2^2 - 2*sqrt(σ_x1^2*σ_x2^2) - 2*sqrt(σ_y1^2*σ_y2^2)
    # = σ_x1^2 + σ_x2^2 - 2*σ_x1*σ_x2 + σ_y1^2 + σ_y2^2 - 2*σ_y1*σ_y2
    # = (σ_x1 - σ_x2)^2 + (σ_y1 - σ_y2)^2
    
    cov_dist = (sigma_x1 - sigma_x2) ** 2 + (sigma_y1 - sigma_y2) ** 2
    
    # Total Wasserstein distance
    wd = torch.sqrt(center_dist + cov_dist + eps)
    
    return wd


def normalized_wasserstein_distance(bbox1, bbox2, constant=None, eps=1e-7):
    """
    Compute Normalized Wasserstein Distance (NWD) between bounding boxes.
    
    NWD is normalized to [0, 1] range similar to IoU:
    NWD = exp(-W2 / C)
    
    where C is a normalization constant based on object scale.
    
    Args:
        bbox1: Tensor of shape (..., 4) in [x1, y1, x2, y2] format
        bbox2: Tensor of shape (..., 4) in [x1, y1, x2, y2] format
        constant: Normalization constant C. If None, use adaptive constant based on bbox size
        eps: Small constant for numerical stability
        
    Returns:
        NWD: Tensor of shape (...) with values in [0, 1]
    """
    # Compute Wasserstein distance
    wd = wasserstein_distance_2d_gaussian(bbox1, bbox2, eps)
    
    # Compute normalization constant if not provided
    if constant is None:
        # Use average size of both boxes as normalization constant
        # This makes NWD scale-invariant
        w1 = bbox1[..., 2] - bbox1[..., 0]
        h1 = bbox1[..., 3] - bbox1[..., 1]
        w2 = bbox2[..., 2] - bbox2[..., 0]
        h2 = bbox2[..., 3] - bbox2[..., 1]
        
        # Diagonal length as normalization factor
        diag1 = torch.sqrt(w1 ** 2 + h1 ** 2)
        diag2 = torch.sqrt(w2 ** 2 + h2 ** 2)
        constant = (diag1 + diag2) / 2
    
    # Normalize: NWD = exp(-W2 / C)
    nwd = torch.exp(-wd / (constant + eps))
    
    return nwd


class NWDLoss(nn.Module):
    """
    NWD Loss for bounding box regression.
    Can be used as a drop-in replacement for IoU-based losses.
    
    Loss = 1 - NWD
    """
    def __init__(self, constant=None, eps=1e-7):
        super().__init__()
        self.constant = constant
        self.eps = eps
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: Predicted boxes (..., 4) in [x1, y1, x2, y2]
            target_boxes: Target boxes (..., 4) in [x1, y1, x2, y2]
            
        Returns:
            Loss scalar
        """
        nwd = normalized_wasserstein_distance(pred_boxes, target_boxes, self.constant, self.eps)
        loss = 1.0 - nwd
        return loss.mean()


def nwd_nms(boxes, scores, iou_threshold=0.5, use_nwd_threshold=True):
    """
    Non-Maximum Suppression using NWD instead of IoU.
    
    Args:
        boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        scores: Tensor of shape (N,)
        iou_threshold: Threshold for suppression (default 0.5)
        use_nwd_threshold: If True, use NWD threshold; otherwise use IoU threshold
        
    Returns:
        keep: Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Sort by scores
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0])
            break
            
        i = order[0]
        keep.append(i)
        
        # Compute NWD with remaining boxes
        if order.numel() == 1:
            break
            
        remaining_boxes = boxes[order[1:]]
        current_box = boxes[i:i+1]
        
        # Compute NWD
        nwd = normalized_wasserstein_distance(
            current_box.expand_as(remaining_boxes),
            remaining_boxes
        )
        
        # Keep boxes with NWD below threshold (less similar)
        mask = nwd <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def compute_nwd_matrix(boxes1, boxes2, constant=None, eps=1e-7):
    """
    Compute NWD matrix between two sets of boxes.
    Useful for label assignment.
    
    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2]
        boxes2: Tensor of shape (M, 4) in [x1, y1, x2, y2]
        constant: Normalization constant
        eps: Small constant
        
    Returns:
        NWD matrix of shape (N, M)
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Expand dimensions for broadcasting
    boxes1_exp = boxes1.unsqueeze(1).expand(N, M, 4)  # (N, M, 4)
    boxes2_exp = boxes2.unsqueeze(0).expand(N, M, 4)  # (N, M, 4)
    
    # Compute NWD for all pairs
    nwd_matrix = normalized_wasserstein_distance(boxes1_exp, boxes2_exp, constant, eps)
    
    return nwd_matrix


def nwd_anchor_assignment(gt_boxes, anchor_boxes, pos_threshold=0.7, neg_threshold=0.3):
    """
    Assign anchors to ground truth boxes using NWD instead of IoU.
    
    Args:
        gt_boxes: Ground truth boxes (N_gt, 4) in [x1, y1, x2, y2]
        anchor_boxes: Anchor boxes (N_anchor, 4) in [x1, y1, x2, y2]
        pos_threshold: NWD threshold for positive samples (default 0.7)
        neg_threshold: NWD threshold for negative samples (default 0.3)
        
    Returns:
        labels: Tensor of shape (N_anchor,) with values:
                1 for positive samples
                0 for negative samples
                -1 for ignore samples
        matched_gt_idx: Tensor of shape (N_anchor,) with matched GT index for each anchor
    """
    if gt_boxes.numel() == 0:
        labels = torch.zeros(anchor_boxes.shape[0], dtype=torch.long, device=anchor_boxes.device)
        matched_gt_idx = torch.zeros(anchor_boxes.shape[0], dtype=torch.long, device=anchor_boxes.device)
        return labels, matched_gt_idx
    
    # Compute NWD matrix (N_anchor, N_gt)
    nwd_matrix = compute_nwd_matrix(anchor_boxes, gt_boxes)
    
    # For each anchor, find best matching GT
    max_nwd, matched_gt_idx = nwd_matrix.max(dim=1)
    
    # Initialize labels as ignore (-1)
    labels = torch.full((anchor_boxes.shape[0],), -1, dtype=torch.long, device=anchor_boxes.device)
    
    # Assign positive labels (NWD > pos_threshold)
    labels[max_nwd >= pos_threshold] = 1
    
    # Assign negative labels (NWD < neg_threshold)
    labels[max_nwd < neg_threshold] = 0
    
    # For each GT, ensure at least one anchor is assigned (highest NWD)
    if gt_boxes.shape[0] > 0:
        gt_max_nwd, gt_best_anchor = nwd_matrix.max(dim=0)
        labels[gt_best_anchor] = 1
    
    return labels, matched_gt_idx


def compute_iou(box1, box2, eps=1e-7):
    """
    Compute IoU for comparison with NWD.
    
    Args:
        box1: Tensor (..., 4) in [x1, y1, x2, y2]
        box2: Tensor (..., 4) in [x1, y1, x2, y2]
        
    Returns:
        IoU: Tensor (...)
    """
    # Intersection
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Union
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + eps)
    return iou


# Test function
if __name__ == "__main__":
    print("Testing NWD implementation...")
    
    # Test case 1: Identical boxes
    box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    box2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    nwd = normalized_wasserstein_distance(box1, box2)
    print(f"Test 1 - Identical boxes: NWD = {nwd.item():.4f} (should be ~1.0)")
    
    # Test case 2: Tiny object with 1 pixel deviation (from paper example)
    box1 = torch.tensor([[0, 0, 6, 6]], dtype=torch.float32)  # 6x6 tiny object
    box2 = torch.tensor([[1, 1, 7, 7]], dtype=torch.float32)  # 1 pixel deviation
    nwd = normalized_wasserstein_distance(box1, box2)
    iou = compute_iou(box1, box2)
    print(f"Test 2 - Tiny object (6x6), 1px deviation: NWD = {nwd.item():.4f}, IoU = {iou.item():.4f}")
    
    # Test case 3: Normal object with same deviation
    box1 = torch.tensor([[0, 0, 36, 36]], dtype=torch.float32)  # 36x36 normal object
    box2 = torch.tensor([[1, 1, 37, 37]], dtype=torch.float32)  # 1 pixel deviation
    nwd = normalized_wasserstein_distance(box1, box2)
    iou = compute_iou(box1, box2)
    print(f"Test 3 - Normal object (36x36), 1px deviation: NWD = {nwd.item():.4f}, IoU = {iou.item():.4f}")
    
    # Test case 4: No overlap
    box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    box2 = torch.tensor([[20, 20, 30, 30]], dtype=torch.float32)
    nwd = normalized_wasserstein_distance(box1, box2)
    iou = compute_iou(box1, box2)
    print(f"Test 4 - No overlap: NWD = {nwd.item():.4f}, IoU = {iou.item():.4f}")
    
    # Test case 5: Batch processing
    boxes1 = torch.tensor([
        [0, 0, 10, 10],
        [5, 5, 15, 15],
        [10, 10, 20, 20]
    ], dtype=torch.float32)
    boxes2 = torch.tensor([
        [1, 1, 11, 11],
        [5, 5, 15, 15],
        [15, 15, 25, 25]
    ], dtype=torch.float32)
    nwd = normalized_wasserstein_distance(boxes1, boxes2)
    print(f"Test 5 - Batch processing: NWD = {nwd}")
    
    # Test NWD matrix computation
    gt_boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32)
    anchor_boxes = torch.tensor([
        [9, 9, 21, 21],
        [11, 11, 19, 19],
        [29, 29, 41, 41]
    ], dtype=torch.float32)
    nwd_matrix = compute_nwd_matrix(anchor_boxes, gt_boxes)
    print(f"Test 6 - NWD matrix:\n{nwd_matrix}")
    
    # Test anchor assignment
    labels, matched = nwd_anchor_assignment(gt_boxes, anchor_boxes, pos_threshold=0.7, neg_threshold=0.3)
    print(f"Test 7 - Anchor assignment:\nLabels: {labels}\nMatched GT: {matched}")
    
    print("\nAll tests completed!")


__all__ = [
    'wasserstein_distance_2d_gaussian',
    'normalized_wasserstein_distance',
    'NWDLoss',
    'nwd_nms',
    'compute_nwd_matrix',
    'nwd_anchor_assignment',
    'compute_iou'
]

