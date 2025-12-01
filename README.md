# UAV Object Detection Toolkit — YOLO with Attention & NWD

This repository contains code to train and run lightweight YOLO variants adapted for UAV (drone) imagery, with additional attention modules and a Normalized Wasserstein Distance (NWD) integration for improved tiny-object detection. It includes training/inference scripts, motion-based local/global detection logic and dataset utilities.

## Overview

- Attention modules (channel/spatial) integrated into YOLO backbones and heads to improve feature selection for small objects captured by UAVs.
- NWD (Normalized Wasserstein Distance) implemented as an alternative similarity/metric for bounding-box comparison and as a component in assignment/loss/NMS.
- GL-YOMO-style modules and motion analysis: multi-frame motion detection, template matching, Kalman filter and a global-local strategy to focus inference on ROIs.
- Utilities for dataset merging/splitting, training and inference workflows.

The implementation is intended for experiments with small / nano YOLO models and research into attention and alternative metrics for detection under UAV imaging conditions.

## Key algorithms and components

1) Attention blocks
- Squeeze-and-Excitation (SE)-style channel attention and CBAM-like channel+spatial attention are used to reweight feature maps.
- Custom modules (CPAM, SAC, TFE, SSFF, Ghost modules) implement lightweight feature encoding and attention-friendly blocks to keep models efficient.

2) Normalized Wasserstein Distance (NWD)
- NWD computes a similarity score from a Wasserstein distance between Gaussian approximations of bounding boxes and normalizes to a [0,1] range.
- Used for anchor assignment, bbox regression loss (NWDLoss), and a NWD-based NMS alternative.

3) Motion detection & GL-YOMO pipeline
- Multi-frame motion extraction (optical flow + frame differencing), multi-scale template matching (NCC), and displacement similarity are combined to detect targets across frames.
- An 8-state Kalman filter verifies detections and supports tracking; a global/local switching strategy narrows ROIs when appropriate to save computation.

## Files of interest

- `train.py` — training script using Ultralytics YOLO API with support for custom modules and NWD.
- `infer.py` — inference script for images, videos, streams or folders; supports GL-YOMO detector.
- `glyomo_modules.py` — implementation of attention blocks, Ghost modules, motion detection, Kalman filter, GLYOMODetector.
- `nwd_modules.py` & `nwd_yolo_patch.py` — NWD implementation, loss and patches to integrate NWD into YOLO.
- `sed_modules.py`, `motion_detector.py` — compatibility wrappers and helper exports.
- `yolo11_uav.yaml`, `yolo11_glyomo.yaml` — example model/dataset configs.
- `runs/`, `output/` — training logs, checkpoints and inference outputs.

## Requirements

See `requirements.txt` for a recommended list. Key packages include:
- torch, torchvision
- ultralytics (YOLO API)
- numpy, opencv-python, matplotlib, PyYAML, tqdm

Note: Install a torch wheel that matches your CUDA version (or use CPU-only wheel).

## Quick start — installation

1) Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) If your system has CUDA, install the matching torch and torchvision wheel from the official PyTorch instructions instead of the pip-installed defaults.

## Training

Prepare a dataset in YOLO format (images + labels, and a `data.yaml`/`data.yml` file). Edit `yolo11_uav.yaml` or `yolo11_glyomo.yaml` as needed.

Run a full training run (example):

```bash
python train.py --data /path/to/data.yml --weights '' --epochs 100 --imgsz 640 --batch 32
```

Quick smoke test:

```bash
python train.py --quick
```

Use `--resume` to resume from last checkpoint. Check `TRAIN_GUIDE.md` for additional tips.

## Inference

Run inference on a single image:

```bash
python infer.py --source ./images/img1.jpg --weights runs/detect/trainX/weights/best.pt --save
```

Run on a video or stream:

```bash
python infer.py --source video.mp4 --weights runs/detect/trainX/weights/best.pt --save
python infer.py --source 0 --weights runs/detect/trainX/weights/best.pt --save  # webcam
```

Batch process a folder:

```bash
python infer.py --source ./images/ --weights runs/detect/trainX/weights/best.pt --save
```

Control confidence/IoU/image size with `--conf`, `--iou`, `--imgsz`.

## Motion-detection usage

When working with videos, consider using the GL-YOMO detector (enabled automatically if custom modules are available). The motion-based ROI approach reduces the number of frames / regions passed to the detector and helps track small objects across frames.

## Outputs

- Trained models and logs saved under `runs/detect/<trainN>/`.
- Inference outputs saved in `output/` when `--save` is used.

## Tips and best practices

- Start with small models and smaller `imgsz` for fast iteration.
- Validate changes to attention or NWD on held-out validation sets and compare mAP, precision and recall.
- When debugging custom modules, use `--dry-run` or small batches (`--quick`).

## References

- Squeeze-and-Excitation Networks — Jie Hu, Li Shen, Gang Sun (2018): https://arxiv.org/abs/1709.01507
- CBAM: Convolutional Block Attention Module — Sanghyun Woo et al. (2018): https://arxiv.org/abs/1807.06521
- Attention Is All You Need — Vaswani et al. (2017): https://arxiv.org/abs/1706.03762
- YOLO family papers (YOLOv3 / YOLOv4) — see arXiv for original references
- A Normalized Gaussian Wasserstein Distance for Tiny Object Detection — (NWD paper): https://arxiv.org/abs/2110.13389
- Wasserstein distance and Optimal Transport survey: https://arxiv.org/abs/1908.09890


