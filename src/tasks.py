"""
tasks.py
Custom YOLO model parser that supports GL-YOMO modules.
"""

import torch
import torch.nn as nn
import contextlib
import ast

from ultralytics.nn.modules import (
    AIFI, C1, C2, C2PSA, C3, C3TR, ELAN1, OBB, PSA, SPP, SPPELAN, SPPF,
    A2C2f, AConv, ADown, Bottleneck, BottleneckCSP, C2f, C2fAttn, C2fCIB,
    C2fPSA, C3k2, C3x, CBFuse, CBLinear, Classify, Concat, Conv, Conv2,
    ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus, HGBlock, HGStem,
    ImagePoolingAttn, Index, Pose, RepC3, RepConv, RepNCSPELAN4, RepVGGDW,
    ResNetLayer, RTDETRDecoder, SCDown, Segment, TorchVision, WorldDetect, v10Detect,
)
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.ops import make_divisible

# Import GL-YOMO custom modules
from glyomo_modules import (
    SAC, C3_SAC, EMA, AdaptiveConcat, DyHead,
    GhostConv, C3Ghost, GhostBottleneck, CPAM, TFE, SSFF
)

# Register modules to global scope for model parsing
_custom_modules = {
    'SAC': SAC,
    'C3_SAC': C3_SAC,
    'EMA': EMA,
    'AdaptiveConcat': AdaptiveConcat,
    'DyHead': DyHead,
    'GhostConv': GhostConv,
    'C3Ghost': C3Ghost,
    'GhostBottleneck': GhostBottleneck,
    'CPAM': CPAM,
    'TFE': TFE,
    'SSFF': SSFF,
}

for name, module in _custom_modules.items():
    globals()[name] = module


def custom_parse_model(d, ch, verbose=True):
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.
    Supports all GL-YOMO custom modules.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[1]
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    
    # Base modules that follow standard c1, c2, *args pattern
    base_modules = frozenset({
        Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
        SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f,
        C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR,
        C3Ghost, torch.nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, PSA,
        SCDown, C2fCIB, A2C2f,
        # GL-YOMO modules
        C3_SAC, SAC,
    })
    
    # Modules with repeat arguments
    repeat_modules = frozenset({
        BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x,
        RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        # GL-YOMO modules
        C3_SAC,
    })
    
    # Combine backbone, neck, head
    all_layers = d.get("backbone", []) + d.get("neck", []) + d.get("head", [])
    
    for i, (f, n, m, args) in enumerate(all_layers):
        # Get module class
        m = (
            getattr(torch.nn, m[3:]) if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:]) if "torchvision.ops." in m
            else globals().get(m, getattr(__import__('ultralytics.nn.modules', fromlist=[m]), m, None))
        )
        
        if m is None:
            raise ValueError(f"Unknown module: {m}")
        
        # Parse arguments
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
                
        elif m is CPAM:
            # CPAM: [channels, reduction]
            c1 = ch[f]
            c2 = c1  # CPAM preserves channels
            args = [c1, *args[1:]] if len(args) > 1 else [c1, 16]
            
        elif m is TFE:
            # TFE: [c1, c2]
            c1 = ch[f]
            c2 = args[0] if args else c1
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2]
            
        elif m is SSFF:
            # SSFF: [c1, c2, num_scales]
            c1 = ch[f]
            c2 = args[0] if args else c1
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2] + (args[1:] if len(args) > 1 else [])
            
        elif m is EMA:
            # EMA: [channels, beta]
            c1 = ch[f]
            c2 = c1  # EMA preserves channels
            args = [c1] + (args if args else [0.99])
            
        elif m is AIFI:
            args = [ch[f], *args]
            
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
                
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
            
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
            
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
            
        elif m in frozenset({Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
                
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
            
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
            
        elif m is CBFuse:
            c2 = ch[f[-1]]
            
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
            
        elif m is AdaptiveConcat:
            # AdaptiveConcat: [c2]
            if isinstance(f, list):
                c2 = args[0] if args else sum(ch[x] for x in f)
            else:
                c2 = args[0] if args else ch[f]
            args = [c2]
            
        elif m is DyHead:
            # DyHead: [nc]
            if isinstance(f, list):
                c2 = args[0] if args else sum(ch[x] for x in f)
            else:
                c2 = args[0] if args else ch[f]
            args = [nc if nc else args[0]]
            
        else:
            # Default handling
            if isinstance(f, list):
                c2 = ch[f[-1]]
            else:
                c2 = ch[f]

        # Build module
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")
        
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        
        if i == 0:
            ch = []
        ch.append(c2)
    
    return nn.Sequential(*layers), sorted(save)


__all__ = ['custom_parse_model'] + list(_custom_modules.keys())
