"""
train.py
Complete training script for NWD-enhanced GL-YOMO YOLO
Combines GL-YOMO modules with Normalized Wasserstein Distance for tiny object detection

Reference: 
- NWD Paper: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
- GL-YOMO: "Real-Time Detection for Small UAVs: Combining YOLO and Multi-frame Motion Analysis"
"""

import os
import argparse
import sys

# --- CPU threading configuration (can be overridden by CLI args) ---
# Set environment variables early so BLAS/OMP libraries pick them up before they are loaded.
_DEFAULT_CPU_THREADS = max(1, (os.cpu_count() or 4) // 2)
os.environ.setdefault('OMP_NUM_THREADS', str(_DEFAULT_CPU_THREADS))
os.environ.setdefault('MKL_NUM_THREADS', str(_DEFAULT_CPU_THREADS))

import torch
torch.use_deterministic_algorithms(False)  # Tắt chế độ deterministic
torch.backends.cudnn.benchmark = True

# Fix matplotlib backend for headless environment
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
import ultralytics
import ultralytics.nn.tasks as yolo_tasks
from sed_modules import (SAC, C3_SAC, EMA, AdaptiveConcat, DyHead,
                         GhostConv, C3Ghost, GhostBottleneck, CPAM, TFE, SSFF)
from tasks import custom_parse_model
import warnings

# Tắt cảnh báo cụ thể từ adaptive_max_pool2d_backward_cuda
warnings.filterwarnings("ignore", message=".*adaptive_max_pool2d_backward_cuda.*")

# Import and apply NWD patches
from nwd_yolo_patch import patch_yolo_with_nwd, NWDBboxLoss, NWDTaskAlignedAssigner
from nwd_modules import normalized_wasserstein_distance

print("="*80)
print("🚀 NWD-Enhanced GL-YOMO Training")
print("="*80)
print("\n📚 Papers:")
print("   - NWD: 'A Normalized Gaussian Wasserstein Distance for Tiny Object Detection'")
print("   - GL-YOMO: 'Real-Time Detection for Small UAVs'")
print("🔗 NWD Reference: https://arxiv.org/abs/2110.13389")
print("\n🔧 Loading NWD modules...")

# Patch YOLO with NWD
patch_yolo_with_nwd()

# Register custom GL-YOMO modules
yolo_tasks.SAC = SAC
yolo_tasks.C3_SAC = C3_SAC
yolo_tasks.EMA = EMA
yolo_tasks.AdaptiveConcat = AdaptiveConcat
yolo_tasks.DyHead = DyHead
yolo_tasks.GhostConv = GhostConv
yolo_tasks.C3Ghost = C3Ghost
yolo_tasks.GhostBottleneck = GhostBottleneck
yolo_tasks.CPAM = CPAM
yolo_tasks.TFE = TFE
yolo_tasks.SSFF = SSFF

# Đăng ký vào globals để YOLO parser nhận dạng
globals().update(dict(SAC=SAC, C3_SAC=C3_SAC, EMA=EMA,
                      AdaptiveConcat=AdaptiveConcat, DyHead=DyHead,
                      GhostConv=GhostConv, C3Ghost=C3Ghost,
                      GhostBottleneck=GhostBottleneck,
                      CPAM=CPAM, TFE=TFE, SSFF=SSFF))

ultralytics.nn.tasks.parse_model = custom_parse_model

print("✅ All modules loaded successfully!")
print("✅ NWD integration completed!")
print("="*80)
print()


def get_next_train_name(project_dir):
    """Tìm tên train tiếp theo (train1, train2, train3, ...)"""
    os.makedirs(project_dir, exist_ok=True)
    existing = [d for d in os.listdir(project_dir) if d.startswith('train') and os.path.isdir(os.path.join(project_dir, d))]
    
    # Tìm số lớn nhất
    max_num = 0
    for name in existing:
        try:
            num = int(name.replace('train', ''))
            max_num = max(max_num, num)
        except ValueError:
            pass
    
    return f"train{max_num + 1}"

# Example usage:
# python3 train.py --data /AI/AI/App_Uav/merged_dataset/data.yml --epochs 100 --imgsz 640 --batch 32 --device 0 --workers 4 --cache
# python3 train.py --quick  # Quick test with 1 epoch
# python3 train.py --resume # Resume from last checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train NWD-Enhanced GL-YOMO for Tiny Object Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full training
  python3 train.py --epochs 100 --batch 32 --workers 4 --cache
  
  # Quick test (1 epoch)
  python3 train.py --quick
  
  # Resume training
  python3 train.py --resume
  
  # Custom dataset
  python3 train.py --data /path/to/data.yml --epochs 50
        '''
    )
    
    # Dataset & Model
    parser.add_argument('--data', default="../merged_dataset/data.yml", 
                       help='Path to data yaml file')
    parser.add_argument('--weights', type=str, 
                       default="./runs/detect/train6/weights/best.pt",
                       help='Path to initial weights (default: best GL-YOMO weights)')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32, 
                       help='Batch size (reduce if OOM)')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Input image size')
    
    # Hardware Configuration
    parser.add_argument('--device', default=0, 
                       help='GPU device id or cpu')
    parser.add_argument('--workers', type=int, default=4, 
                       help='DataLoader workers (0 for single process)')
    parser.add_argument('--num-threads', type=int, default=None, 
                       help='Set OMP/MKL/PyTorch intra-op threads to this value')
    parser.add_argument('--auto-distribute', action='store_true', 
                       help='Auto-distribute threads across workers and main process')
    
    # Training Options
    parser.add_argument('--cache', action='store_true', 
                       help='Cache images in memory/disk for faster training')
    parser.add_argument('--rect', action='store_true', 
                       help='Use rectangular training')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision (can be unstable with custom modules)')
    parser.add_argument('--fraction', type=float, default=None, 
                       help='Fraction of dataset to use (e.g., 0.1 for 10%%)')
    
    # Resume & Save
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from last checkpoint')
    parser.add_argument('--project-dir', 
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect"), 
                       help='Root project directory to save runs')
    parser.add_argument('--save-period', type=int, default=20,
                       help='Save checkpoint every N epochs')
    
    # Testing & Debug
    parser.add_argument('--quick', action='store_true', 
                       help='Quick smoke test (1 epoch, small batch)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Print configuration and exit without training')
    
    args = parser.parse_args()

    # Prepare project dir and session name
    PROJECT_DIR = args.project_dir
    train_name = get_next_train_name(PROJECT_DIR)
    
    print("="*80)
    print(f"📁 Training session: {train_name}")
    print(f"💾 Results will be saved to: {os.path.join(PROJECT_DIR, train_name)}")
    print(f"🎯 Dataset: {args.data}")
    print(f"🔢 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch}")
    print(f"🖼️  Image size: {args.imgsz}")
    print(f"⚙️  Device: {args.device}")
    print(f"👷 Workers: {args.workers if args.workers is not None else 'auto'}")
    if args.weights:
        print(f"⚖️  Initial weights: {args.weights}")
    print("="*80)
    print()

    # Configure threading
    workers = args.workers
    
    if args.num_threads is not None:
        # Optionally auto-distribute threads across worker processes and main process
        if args.auto_distribute:
            # Determine number of worker processes that will be spawned
            n_workers = workers
            total = int(args.num_threads)
            # Ensure at least 1 thread per process
            parts = max(1, n_workers + 1)
            per_part = max(1, total // parts)
            # Give the remainder to the main process to keep it responsive
            remainder = total - per_part * n_workers
            main_threads = max(1, remainder)
            worker_threads = per_part

            # Set environment variables that worker processes will inherit
            os.environ['OMP_NUM_THREADS'] = str(worker_threads)
            os.environ['MKL_NUM_THREADS'] = str(worker_threads)

            # Configure PyTorch threads for main process
            try:
                torch.set_num_threads(int(main_threads))
                torch.set_num_interop_threads(int(main_threads))
            except Exception:
                pass

            print(f"🧵 Auto-distribute: total={total}, workers={n_workers}, main_threads={main_threads}, worker_threads={worker_threads}")
        else:
            os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
            os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
            try:
                torch.set_num_threads(int(args.num_threads))
                torch.set_num_interop_threads(int(args.num_threads))
            except Exception:
                pass
    else:
        # Ensure torch uses a reasonable default
        try:
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', _DEFAULT_CPU_THREADS)))
        except Exception:
            pass

    print(f"🧵 PyTorch threads: {torch.get_num_threads()}")
    print(f"🧵 OMP threads: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"👷 DataLoader workers: {workers}\n")

    # Quick mode for smoke test
    if args.quick:
        print("⚡ Quick mode enabled: 1 epoch, batch=4")
        args.epochs = 1
        args.batch = 4
        args.workers = 0

    if args.dry_run:
        print('✋ Dry-run mode: Configuration displayed, exiting without training')
        sys.exit(0)

    # Load model with NWD enhancements
    print("📥 Loading model...")
    try:
        if args.weights and os.path.exists(args.weights):
            model = YOLO(args.weights, task="detect")
            print(f"✅ Model loaded from: {args.weights}\n")
        else:
            print("⚠️  Weights file not found, loading default YOLO11n...")
            model = YOLO("yolo11n.pt", task="detect")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Falling back to default YOLO11n model...")
        model = YOLO("yolo11n.pt", task="detect")
    
    # Build training configuration
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        verbose=True,
        amp=args.amp,  # Use user-specified AMP setting
        save=True,
        save_period=args.save_period,
        rect=args.rect,
        device=args.device,
        workers=workers,
        project=PROJECT_DIR,
        name=train_name,
        exist_ok=False,
        patience=0 if args.quick else 50,  # No early stopping in quick mode
    )
    
    # Add optional parameters
    if args.fraction is not None:
        train_kwargs['fraction'] = args.fraction
    if args.cache:
        train_kwargs['cache'] = True
    
    # Resume training
    if args.resume:
        train_kwargs['resume'] = True
        train_kwargs['exist_ok'] = True
        print("🔄 Resuming training from last checkpoint...\n")
    
    # Display training configuration
    print("="*80)
    print("🚀 Starting NWD-enhanced GL-YOMO training...")
    print("="*80)
    print("\n📊 Training Configuration:")
    for key, value in sorted(train_kwargs.items()):
        print(f"   {key}: {value}")
    print()
    
    try:
        # Start training
        results = model.train(**train_kwargs)
        
        print("\n" + "="*80)
        print("✅ Training completed successfully!")
        print("="*80)
        print("\n📈 Training Summary:")
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                print(f"   {key}: {value}")
        
        print(f"\n💾 Model saved to: {os.path.join(PROJECT_DIR, train_name)}")
        print("\n🎉 NWD-enhanced training completed without errors!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("⚠️  Training interrupted by user (Ctrl+C)")
        print("="*80)
        print(f"\n💾 Partial results saved to: {os.path.join(PROJECT_DIR, train_name)}")
        print("💡 You can resume training with: python3 train.py --resume")
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ Training failed with error:")
        print("="*80)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Debugging information:")
        print(f"   - Data path: {args.data}")
        print(f"   - Model weights: {args.weights}")
        print(f"   - Device: {args.device}")
        print(f"   - Batch size: {args.batch}")
        print(f"   - Workers: {workers}")
        print("\n💡 Try:")
        print("   - Reduce batch size: --batch 16 or --batch 8")
        print("   - Use fewer workers: --workers 2 or --workers 0")
        print("   - Quick test: --quick")
        print("="*80)
        sys.exit(1)
