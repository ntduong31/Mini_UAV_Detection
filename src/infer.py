#!/usr/bin/env python3
"""
infer.py - GL-YOMO UAV Detection Inference
Supports: Image, Video, Real-time Camera, Folder batch processing

Usage:
    python infer.py --source image.jpg              # Single image
    python infer.py --source video.mp4              # Video file
    python infer.py --source 0                      # Webcam (camera index)
    python infer.py --source rtsp://...             # RTSP stream
    python infer.py --source ./images/              # Folder of images
    python infer.py --source image.jpg --save       # Save output
    python infer.py --source video.mp4 --show       # Display window
"""

import os
import sys
import cv2
import time
import glob
import argparse
import warnings
from pathlib import Path
from typing import Optional, Union, List

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
import ultralytics.nn.tasks as yolo_tasks

# Suppress warnings
warnings.filterwarnings("ignore", message=".*adaptive_max_pool2d_backward_cuda.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# IMPORT GL-YOMO MODULES
# ============================================================================
try:
    from glyomo_modules import (
        SAC, C3_SAC, EMA, AdaptiveConcat, DyHead,
        GhostConv, C3Ghost, GhostBottleneck, CPAM, TFE, SSFF,
        GLYOMODetector, GLYOMOParams, KalmanFilter8State
    )
    from tasks import custom_parse_model
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] GL-YOMO modules not available: {e}")
    MODULES_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def register_custom_modules():
    """Register custom GL-YOMO modules into ultralytics."""
    if not MODULES_AVAILABLE:
        return False
    
    try:
        for name, module in [
            ('SAC', SAC), ('C3_SAC', C3_SAC), ('EMA', EMA),
            ('AdaptiveConcat', AdaptiveConcat), ('DyHead', DyHead),
            ('GhostConv', GhostConv), ('C3Ghost', C3Ghost),
            ('GhostBottleneck', GhostBottleneck), ('CPAM', CPAM),
            ('TFE', TFE), ('SSFF', SSFF)
        ]:
            setattr(yolo_tasks, name, module)
        
        ultralytics.nn.tasks.parse_model = custom_parse_model
        return True
    except Exception as e:
        print(f"[ERROR] Failed to register modules: {e}")
        return False


def get_source_type(source: str) -> str:
    """Determine source type: image, video, camera, stream, or folder."""
    source = str(source).strip()
    
    # Camera index
    if source.isdigit():
        return "camera"
    
    # RTSP/HTTP stream
    if source.lower().startswith(('rtsp://', 'http://', 'https://')):
        return "stream"
    
    # Folder
    if os.path.isdir(source):
        return "folder"
    
    # File - check extension
    ext = Path(source).suffix.lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    if ext in image_exts:
        return "image"
    elif ext in video_exts:
        return "video"
    else:
        # Try to determine by opening
        if os.path.isfile(source):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, _ = cap.read()
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if ret:
                    return "video" if frame_count > 1 else "image"
        return "unknown"


def get_images_from_folder(folder: str) -> List[str]:
    """Get all image files from a folder."""
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    images = []
    for ext in image_exts:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(images)


def draw_detections(frame: np.ndarray, results, conf_thresh: float = 0.3) -> np.ndarray:
    """Draw detection boxes on frame."""
    annotated = frame.copy()
    
    if results is None or len(results) == 0:
        return annotated
    
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return annotated
    
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls = int(box.cls[0])
        
        # Draw box
        color = (0, 255, 0)  # Green for UAV
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"UAV {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return annotated


def add_info_overlay(frame: np.ndarray, fps: float = 0, 
                     frame_num: int = 0, total_frames: int = 0,
                     detections: int = 0) -> np.ndarray:
    """Add info overlay to frame."""
    h, w = frame.shape[:2]
    
    # Info text
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Detections: {detections}",
    ]
    if total_frames > 0:
        info_lines.append(f"Frame: {frame_num}/{total_frames}")
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (200, 20 + len(info_lines) * 25), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Draw text
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (15, 30 + i * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame


# ============================================================================
# INFERENCE CLASS
# ============================================================================

class GLYOMOInference:
    """GL-YOMO Inference Engine for UAV Detection."""
    
    def __init__(self, 
                 weights: str = "./runs/detect/train3/weights/best.pt",
                 device: str = "auto",
                 conf: float = 0.3,
                 iou: float = 0.45,
                 imgsz: int = 640,
                 use_glyomo: bool = True):
        """
        Initialize inference engine.
        
        Args:
            weights: Path to model weights
            device: Device to use (auto, cuda, cpu)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Input image size
            use_glyomo: Use GL-YOMO detector with motion analysis
        """
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.use_glyomo = use_glyomo and MODULES_AVAILABLE
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Weights: {weights}")
        
        # Register custom modules
        if MODULES_AVAILABLE:
            register_custom_modules()
            print("[INFO] GL-YOMO modules registered")
        
        # Load model
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")
        
        self.model = YOLO(weights)
        self.model.to(self.device)
        print(f"[INFO] Model loaded: {self.model.model.__class__.__name__}")
        
        # Initialize GL-YOMO detector if available
        self.glyomo_detector = None
        if self.use_glyomo:
            try:
                params = GLYOMOParams(
                    Ng=30, Nl=60,
                    roi_base_size=300,
                    tau_g=0.3, tau_l=0.1,
                    k2=0.6, k3=0.4,
                    scales=[0.7, 1.0, 1.3]
                )
                self.glyomo_detector = GLYOMODetector(self.model, params)
                print("[INFO] GL-YOMO detector initialized")
            except Exception as e:
                print(f"[WARNING] GL-YOMO detector init failed: {e}")
                self.glyomo_detector = None
    
    def predict_image(self, image: np.ndarray) -> tuple:
        """
        Run inference on single image.
        
        Returns:
            (annotated_image, results, detections_count)
        """
        # Run YOLO inference
        results = self.model(image, conf=self.conf, iou=self.iou, 
                            imgsz=self.imgsz, verbose=False, device=self.device)
        
        # Count detections
        n_detections = 0
        if results and len(results) > 0 and results[0].boxes is not None:
            n_detections = len([b for b in results[0].boxes if float(b.conf[0]) >= self.conf])
        
        # Draw detections
        annotated = draw_detections(image, results, self.conf)
        
        return annotated, results, n_detections
    
    def process_image(self, source: str, save: bool = False, 
                      save_dir: str = "./output") -> dict:
        """Process a single image file."""
        print(f"[INFO] Processing image: {source}")
        
        # Read image
        image = cv2.imread(source)
        if image is None:
            return {"error": f"Cannot read image: {source}"}
        
        # Inference
        start = time.time()
        annotated, results, n_det = self.predict_image(image)
        inference_time = time.time() - start
        
        print(f"[INFO] Detections: {n_det}, Time: {inference_time*1000:.1f}ms")
        
        # Save output
        if save:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"result_{Path(source).name}")
            cv2.imwrite(out_path, annotated)
            print(f"[INFO] Saved: {out_path}")
        
        return {
            "source": source,
            "detections": n_det,
            "time_ms": inference_time * 1000,
            "image_shape": image.shape
        }
    
    def process_video(self, source: Union[str, int], save: bool = False,
                      save_dir: str = "./output", max_frames: int = 0) -> dict:
        """
        Process video file, camera, or stream.
        
        Args:
            source: Video path, camera index, or stream URL
            save: Save output video
            save_dir: Output directory
            max_frames: Max frames to process (0 = all)
        """
        # Open video source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {"error": f"Cannot open video source: {source}"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        is_stream = isinstance(source, int) or str(source).startswith(('rtsp://', 'http://'))
        
        print(f"[INFO] Video: {width}x{height} @ {fps:.1f}fps")
        if total_frames > 0:
            print(f"[INFO] Total frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if save:
            os.makedirs(save_dir, exist_ok=True)
            if isinstance(source, int):
                out_name = f"camera_{source}_output.mp4"
            else:
                out_name = f"result_{Path(str(source)).stem}.mp4"
            out_path = os.path.join(save_dir, out_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            print(f"[INFO] Saving to: {out_path}")
        
        # Process frames
        frame_num = 0
        total_detections = 0
        fps_counter = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if is_stream:
                        print("[WARNING] Stream interrupted, retrying...")
                        time.sleep(1)
                        continue
                    break
                
                frame_num += 1
                if max_frames > 0 and frame_num > max_frames:
                    break
                
                # Inference
                start = time.time()
                annotated, results, n_det = self.predict_image(frame)
                inference_time = time.time() - start
                
                total_detections += n_det
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                fps_counter.append(current_fps)
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                avg_fps = sum(fps_counter) / len(fps_counter)
                
                # Add overlay
                annotated = add_info_overlay(
                    annotated, fps=avg_fps, 
                    frame_num=frame_num, 
                    total_frames=total_frames if not is_stream else 0,
                    detections=n_det
                )
                
                # Save frame
                if writer:
                    writer.write(annotated)
                
                # Progress
                if frame_num % 100 == 0:
                    if total_frames > 0:
                        progress = frame_num / total_frames * 100
                        print(f"[INFO] Progress: {progress:.1f}% ({frame_num}/{total_frames})")
                    else:
                        print(f"[INFO] Processed: {frame_num} frames, FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        avg_fps_final = sum(fps_counter) / len(fps_counter) if fps_counter else 0
        
        return {
            "source": str(source),
            "frames_processed": frame_num,
            "total_detections": total_detections,
            "avg_fps": avg_fps_final,
            "video_fps": fps,
            "resolution": f"{width}x{height}"
        }
    
    def process_folder(self, folder: str, save: bool = False,
                       save_dir: str = "./output") -> dict:
        """Process all images in a folder."""
        images = get_images_from_folder(folder)
        if not images:
            return {"error": f"No images found in: {folder}"}
        
        print(f"[INFO] Found {len(images)} images in {folder}")
        
        results_list = []
        total_detections = 0
        total_time = 0
        
        for i, img_path in enumerate(images):
            result = self.process_image(img_path, save=save, save_dir=save_dir)
            if "error" not in result:
                results_list.append(result)
                total_detections += result["detections"]
                total_time += result["time_ms"]
            
            if (i + 1) % 10 == 0:
                print(f"[INFO] Processed: {i+1}/{len(images)}")
        
        return {
            "source": folder,
            "total_images": len(images),
            "processed": len(results_list),
            "total_detections": total_detections,
            "total_time_ms": total_time,
            "avg_time_ms": total_time / len(results_list) if results_list else 0
        }
    
    def run(self, source: str, save: bool = False, 
            save_dir: str = "./output", max_frames: int = 0) -> dict:
        """
        Run inference on any source type.
        
        Args:
            source: Image, video, camera index, stream URL, or folder
            save: Save output
            save_dir: Output directory
            max_frames: Max frames for video (0 = all)
        """
        source_type = get_source_type(source)
        print(f"[INFO] Source type: {source_type}")
        
        if source_type == "image":
            return self.process_image(source, save, save_dir)
        elif source_type in ("video", "camera", "stream"):
            return self.process_video(source, save, save_dir, max_frames)
        elif source_type == "folder":
            return self.process_folder(source, save, save_dir)
        else:
            return {"error": f"Unknown source type: {source}"}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GL-YOMO UAV Detection Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --source image.jpg                    # Single image
  python infer.py --source video.mp4 --save             # Video with save
  python infer.py --source 0 --show                     # Webcam with display
  python infer.py --source ./images/ --save             # Folder batch
  python infer.py --source rtsp://... --show --save     # RTSP stream
        """
    )
    
    parser.add_argument("--source", type=str, required=True,
                        help="Image, video, camera index (0,1..), stream URL, or folder")
    parser.add_argument("--weights", type=str, 
                        default="./runs/detect/train3/weights/best.pt",
                        help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device (default: auto)")
    parser.add_argument("--save", action="store_true",
                        help="Save output")
    parser.add_argument("--save-dir", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames for video (0 = all)")
    
    args = parser.parse_args()
    
    # Initialize engine
    try:
        engine = GLYOMOInference(
            weights=args.weights,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return 1
    
    # Run inference
    result = engine.run(
        source=args.source,
        save=args.save,
        save_dir=args.save_dir,
        max_frames=args.max_frames
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("INFERENCE RESULTS")
    print("=" * 50)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
