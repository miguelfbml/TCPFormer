"""
Compare Ground Truth and YOLO 2D poses with comprehensive metrics (MPJPE, PCK, AUC)
Creates side-by-side visualization showing GT vs YOLO predictions with frame-by-frame MPJPE

Usage:
python compare_gt_yolo_2d.py --sequence TS1 --model-path runs/pose/mpi_yolo_pose_full/weights/best.pt --num-frames 50 --save-video
python compare_gt_yolo_2d.py --sequence TS1 --model-path runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt --num-frames 50 --save-video
python compare_gt_yolo_2d.py --all --model-path runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt
python compare_gt_yolo_2d.py --all --model-path runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt --num-frames 50
python compare_gt_yolo_2d.py --sequence TS6 --model-path runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt --save-video
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from tqdm import tqdm
import glob
import gc
import shutil
try:
    from PIL import Image
except ImportError:
    Image = None

# GPU detection and setup
def check_gpu_availability(args_device):
    """Check GPU availability and set device"""
    gpu_available = torch.cuda.is_available()
    print(f"\n{'='*60}\nSYSTEM INFORMATION\n{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {gpu_available}")

    if args_device == 'auto':
        device = f'cuda:{torch.cuda.current_device()}' if gpu_available else 'cpu'
    else:
        device = args_device
        if device.startswith('cuda') and not gpu_available:
            print("⚠️ CUDA device specified but not available, falling back to CPU")
            device = 'cpu'

    if gpu_available and device.startswith('cuda'):
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        print(f"GPU count: {gpu_count}\nCurrent GPU: {current_device}\nGPU name: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f} GB")
        print(f"CUDA compute capability: {torch.cuda.get_device_capability(current_device)}")
        torch.cuda.set_per_process_memory_fraction(0.8)
        print(f"GPU memory fraction set to: 80%")

    print(f"Selected device: {device}\n{'='*60}")
    return device, gpu_available

# Metric calculation functions
def calculate_torso_diameter_2d(poses_2d):
    """Calculate torso diameter from 2D poses using shoulder width and torso height"""
    if len(poses_2d.shape) != 3 or poses_2d.shape[1] != 17:
        print(f"Warning: Unexpected pose shape {poses_2d.shape}, using default torso diameter")
        return np.ones(poses_2d.shape[0]) * 100.0

    left_shoulder = poses_2d[:, 2, :]  # LShoulder
    right_shoulder = poses_2d[:, 5, :]  # RShoulder
    spine_shoulder = poses_2d[:, 1, :]  # SpineShoulder
    sacrum = poses_2d[:, 14, :]  # Sacrum
    left_hip = poses_2d[:, 8, :]  # LHip
    right_hip = poses_2d[:, 11, :]  # RHip

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    hip_width = np.linalg.norm(left_hip - right_hip, axis=1)
    torso_height = np.linalg.norm(spine_shoulder - sacrum, axis=1)
    torso_diameter = np.maximum.reduce([shoulder_width, hip_width, torso_height * 0.5])
    torso_diameter = np.clip(torso_diameter, 50.0, 300.0)
    torso_diameter[torso_diameter < 1e-6] = 100.0  # Default for invalid frames
    return torso_diameter

def compute_pck_2d(pred_poses, gt_poses, torso_diameters, fixed_threshold=150.0):
    """Compute PCK (Percentage of Correct Keypoints) for 2D poses"""
    if pred_poses.shape != gt_poses.shape:
        print(f"Warning: Shape mismatch between predictions {pred_poses.shape} and GT {gt_poses.shape}")
        return {f'PCK@{pct}%_torso': 0.0 for pct in [20, 50, 80, 100]} | {f'PCK@100%_{int(fixed_threshold)}px': 0.0}

    joint_distances = np.linalg.norm(pred_poses - gt_poses, axis=2)
    results = {}
    for pct in [20, 50, 80, 100]:
        threshold = torso_diameters[:, np.newaxis] * (pct / 100.0)
        correct = joint_distances < threshold
        results[f'PCK@{pct}%_torso'] = np.mean(correct)
    correct_fixed = joint_distances < fixed_threshold
    results[f'PCK@100%_{int(fixed_threshold)}px'] = np.mean(correct_fixed)
    return results

def compute_auc_2d(pred_poses, gt_poses, max_threshold=150.0, num_thresholds=50):
    """Compute AUC (Area Under Curve) for 2D poses"""
    if pred_poses.shape != gt_poses.shape:
        print(f"Warning: Shape mismatch for AUC calculation")
        return 0.0

    joint_distances = np.linalg.norm(pred_poses - gt_poses, axis=2)
    thresholds = np.linspace(0, max_threshold, num_thresholds)
    pck_values = [np.mean(joint_distances < threshold) for threshold in thresholds]
    return np.trapz(pck_values, thresholds) / max_threshold

# MPI-INF-3DHP joint names and connections
JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand', 'RShoulder', 'RElbow', 'RHand',
    'LHip', 'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle', 'Sacrum', 'Spine', 'Neck'
]
CONNECTIONS_2D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]
TEST_SEQUENCES = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

def get_available_sequences():
    """Get list of available test sequences from data file"""
    test_data_paths = [
        '../../motion3d/data_test_3dhp.npz',
        '../../../motion3d/data_test_3dhp.npz',
        '../../../../motion3d/data_test_3dhp.npz'
    ]
    for data_path in test_data_paths:
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)['data'].item()
            return list(data.keys())
    return TEST_SEQUENCES

def load_test_3d_data_from_dataset(sequence_name):
    """Load test data for a specific sequence"""
    test_data_paths = [
        '../../motion3d/data_test_3dhp.npz',
        '../../../motion3d/data_test_3dhp.npz',
        '../../../../motion3d/data_test_3dhp.npz'
    ]
    for data_path in test_data_paths:
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)['data'].item()
            if sequence_name in data:
                seq_data = data[sequence_name]
                return seq_data['data_2d'], seq_data['data_3d'], sequence_name
    return None, None, None

def load_test_frames_batch(sequence_name, start_frame=0, num_frames=None):
    """Load test frames for the sequence in batches"""
    test_image_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
        '../motion3d/mpi_inf_3dhp_test_set',
        '../../motion3d/mpi_inf_3dhp_test_set',
        '../../../motion3d/mpi_inf_3dhp_test_set'
    ]
    for base_path in test_image_paths:
        image_folder = os.path.join(base_path, sequence_name, 'imageSequence')
        if os.path.exists(image_folder):
            image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
            image_files.sort()
            if image_files:
                end_frame = start_frame + num_frames if num_frames is not None else len(image_files)
                batch_files = image_files[start_frame:end_frame]
                frames = [cv2.imread(img_path) for img_path in batch_files if cv2.imread(img_path) is not None]
                return frames, len(image_files)
    return None, 0

def estimate_yolo_poses_batch(model, frames, img_size=640, device='cpu'):
    """Estimate poses using YOLO model with memory management"""
    yolo_poses = []
    confidences = []
    batch_size = 32 if device.startswith('cuda') else 16
    print(f"Processing {len(frames)} frames in batches of {batch_size} on {device}")

    for i in tqdm(range(0, len(frames), batch_size), desc="YOLO Inference"):
        batch_frames = frames[i:i+batch_size]
        for frame in batch_frames:
            try:
                results = model.predict(frame, verbose=False, imgsz=img_size, conf=0.2, device=device)
                if (results and len(results) > 0 and hasattr(results[0], 'keypoints') and 
                    results[0].keypoints is not None and len(results[0].keypoints.xy) > 0):
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
                    if keypoints.shape[0] == 17:
                        yolo_poses.append(keypoints)
                        confidences.append(conf)
                    else:
                        padded_kpts = np.zeros((17, 2))
                        padded_conf = np.zeros(17)
                        n_kpts = min(17, keypoints.shape[0])
                        padded_kpts[:n_kpts] = keypoints[:n_kpts]
                        padded_conf[:n_kpts] = conf[:n_kpts] if len(conf) > 0 else 0.5
                        yolo_poses.append(padded_kpts)
                        confidences.append(padded_conf)
                else:
                    yolo_poses.append(np.zeros((17, 2)))
                    confidences.append(np.zeros(17))
            except Exception as e:
                print(f"Error in YOLO inference: {e}")
                yolo_poses.append(np.zeros((17, 2)))
                confidences.append(np.zeros(17))
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()
    return np.array(yolo_poses), np.array(confidences)

def convert_coordinates_to_pixels(poses_2d, frames):
    """Convert normalized coordinates to pixel coordinates"""
    if not frames:
        return poses_2d
    img_height, img_width = frames[0].shape[:2]
    if np.max(poses_2d[:, :, :2]) > 1.0:
        return poses_2d
    poses_pixel = poses_2d.copy()
    poses_pixel[:, :, 0] *= img_width
    poses_pixel[:, :, 1] *= img_height
    return poses_pixel

def make_root_relative_2d_pixel(poses_2d_pixel, root_joint_idx=14):
    """Make poses root-relative in pixel domain"""
    root_relative_poses = poses_2d_pixel.copy()
    for frame_idx in range(poses_2d_pixel.shape[0]):
        root_pos = poses_2d_pixel[frame_idx, root_joint_idx, :2]
        root_relative_poses[frame_idx, :, :2] -= root_pos
    return root_relative_poses

def compute_mpjpe_2d(gt_poses_2d, yolo_poses_2d):
    """Compute comprehensive metrics including MPJPE, PCK, and AUC for 2D poses"""
    min_frames = min(len(gt_poses_2d), len(yolo_poses_2d))
    gt_poses = gt_poses_2d[:min_frames]
    yolo_poses = yolo_poses_2d[:min_frames]
    valid_frames, valid_gt_list, valid_yolo_list, frame_mpjpe = [], [], [], []

    for frame_idx in range(min_frames):
        gt_frame = gt_poses[frame_idx]
        yolo_frame = yolo_poses[frame_idx]
        gt_valid = not np.all(gt_frame == 0)
        yolo_valid = not np.all(yolo_frame == 0)
        if gt_valid and yolo_valid:
            valid_frames.append(frame_idx)
            valid_gt_list.append(gt_frame)
            valid_yolo_list.append(yolo_frame)
            joint_diffs = np.linalg.norm(gt_frame - yolo_frame, axis=1)
            frame_mpjpe.append(np.mean(joint_diffs))
        else:
            frame_mpjpe.append(np.nan)

    if not valid_gt_list:
        return None

    valid_gt = np.array(valid_gt_list)
    valid_yolo = np.array(valid_yolo_list)
    avg_mpjpe = np.mean([e for e in frame_mpjpe if not np.isnan(e)])
    joint_errors = np.mean(np.linalg.norm(valid_gt - valid_yolo, axis=2), axis=0)
    torso_diameters = calculate_torso_diameter_2d(valid_gt)
    pck_results = compute_pck_2d(valid_yolo, valid_gt, torso_diameters, fixed_threshold=150.0)
    auc = compute_auc_2d(valid_yolo, valid_gt, max_threshold=150.0)

    return {
        'avg_mpjpe': float(avg_mpjpe),
        'frame_mpjpe': frame_mpjpe,
        'joint_errors': [float(x) for x in joint_errors],
        'pck_results': pck_results,
        'auc': float(auc),
        'valid_frames': len(valid_frames),
        'total_frames': min_frames,
        'torso_diameters': torso_diameters
    }

def process_single_sequence(model, sequence_name, args, device='cpu'):
    """Process a single sequence with batched processing for all frames or limited frames"""
    print(f"\n{'='*60}\nProcessing sequence: {sequence_name}\n{'='*60}")

    gt_poses_2d, gt_poses_3d, seq_name = load_test_3d_data_from_dataset(sequence_name)
    if gt_poses_2d is None:
        print(f"❌ Failed to load ground truth data for {sequence_name}")
        return None

    total_gt_frames = len(gt_poses_2d)
    num_frames_to_use = min(args.num_frames, total_gt_frames) if args.num_frames is not None else total_gt_frames
    print(f"✓ Processing {num_frames_to_use} frames for sequence {sequence_name}")

    batch_size = 300 if device.startswith('cuda') else 200
    all_frame_mpjpe, all_valid_gt, all_valid_yolo = [], [], []
    total_valid_frames, total_processed_frames = 0, 0

    for start_idx in range(0, num_frames_to_use, batch_size):
        end_idx = min(start_idx + batch_size, num_frames_to_use)
        batch_frames_count = end_idx - start_idx
        print(f"Processing batch {start_idx//batch_size + 1}/{(num_frames_to_use + batch_size - 1)//batch_size}: frames {start_idx+1}-{end_idx}")

        gt_batch = gt_poses_2d[start_idx:end_idx]
        frames_batch, _ = load_test_frames_batch(sequence_name, start_idx, batch_frames_count)
        if frames_batch is None:
            print(f"❌ Failed to load frames for batch {start_idx}-{end_idx}")
            continue

        min_frames = min(len(frames_batch), len(gt_batch))
        frames_batch = frames_batch[:min_frames]
        gt_batch = gt_batch[:min_frames]

        yolo_batch, _ = estimate_yolo_poses_batch(model, frames_batch, args.img_size, device)
        gt_batch_pixel = convert_coordinates_to_pixels(gt_batch, frames_batch)
        gt_batch_root_rel = make_root_relative_2d_pixel(gt_batch_pixel, root_joint_idx=14)
        yolo_batch_root_rel = make_root_relative_2d_pixel(yolo_batch, root_joint_idx=14)

        for frame_idx in range(min_frames):
            gt_frame = gt_batch_root_rel[frame_idx]
            yolo_frame = yolo_batch_root_rel[frame_idx]
            gt_valid = not np.all(gt_frame == 0)
            yolo_valid = not np.all(yolo_frame == 0)
            if gt_valid and yolo_valid:
                all_valid_gt.append(gt_frame)
                all_valid_yolo.append(yolo_frame)
                total_valid_frames += 1
                joint_diffs = np.linalg.norm(gt_frame - yolo_frame, axis=1)
                all_frame_mpjpe.append(np.mean(joint_diffs))
            else:
                all_frame_mpjpe.append(np.nan)
            total_processed_frames += 1

        del frames_batch, gt_batch, yolo_batch, gt_batch_pixel, gt_batch_root_rel, yolo_batch_root_rel
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

    if not all_valid_gt:
        print(f"❌ No valid frames found for {sequence_name}")
        return None

    valid_gt = np.array(all_valid_gt)
    valid_yolo = np.array(all_valid_yolo)
    avg_mpjpe = np.mean([e for e in all_frame_mpjpe if not np.isnan(e)])
    joint_errors = np.mean(np.linalg.norm(valid_gt - valid_yolo, axis=2), axis=0)
    torso_diameters = calculate_torso_diameter_2d(valid_gt)
    pck_results = compute_pck_2d(valid_yolo, valid_gt, torso_diameters, fixed_threshold=150.0)
    auc = compute_auc_2d(valid_yolo, valid_gt, max_threshold=150.0)

    metrics = {
        'avg_mpjpe': float(avg_mpjpe),
        'frame_mpjpe': all_frame_mpjpe,
        'joint_errors': [float(x) for x in joint_errors],
        'pck_results': pck_results,
        'auc': float(auc),
        'valid_frames': total_valid_frames,
        'total_frames': total_processed_frames,
        'torso_diameters': torso_diameters,
        'sequence': sequence_name
    }

    print(f"✓ Metrics computed for {sequence_name}")
    print(f"  MPJPE: {metrics['avg_mpjpe']:.2f} pixels")
    print(f"  Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
    return metrics

def calculate_visualization_bounds_streaming(model, sequence_name, gt_poses_2d, total_frames, args, device):
    """Calculate visualization bounds by sampling frames"""
    sample_indices = np.linspace(0, total_frames-1, min(100, total_frames), dtype=int)
    all_points = []

    for idx in sample_indices:
        frames_batch, _ = load_test_frames_batch(sequence_name, idx, 1)
        if not frames_batch:
            continue
        frame = frames_batch[0]
        gt_pose = gt_poses_2d[idx:idx+1]
        yolo_poses, _ = estimate_yolo_poses_batch(model, [frame], args.img_size, device)
        gt_pixel = convert_coordinates_to_pixels(gt_pose, [frame])
        gt_root_rel = make_root_relative_2d_pixel(gt_pixel, root_joint_idx=14)
        yolo_root_rel = make_root_relative_2d_pixel(yolo_poses, root_joint_idx=14)
        gt_points = gt_root_rel[0].reshape(-1, 2)
        yolo_points = yolo_root_rel[0].reshape(-1, 2)
        valid_gt = gt_points[~np.all(gt_points == 0, axis=1)]
        valid_yolo = yolo_points[~np.all(yolo_points == 0, axis=1)]
        all_points.extend(valid_gt)
        all_points.extend(valid_yolo)

    if all_points:
        all_points = np.array(all_points)
        x_range = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
        y_range = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
        x_padding = max((x_range[1] - x_range[0]) * 0.1, 50)
        y_padding = max((y_range[1] - y_range[0]) * 0.1, 50)
        return {
            'x_min': x_range[0] - x_padding,
            'x_max': x_range[1] + x_padding,
            'y_min': y_range[0] - y_padding,
            'y_max': y_range[1] + y_padding
        }
    return {'x_min': -500, 'x_max': 500, 'y_min': -500, 'y_max': 500}

def create_single_frame_image(gt_frame, yolo_frame, frame_idx, total_frames, frame_mpjpe, sequence_name, temp_dir, bounds):
    """Create a single frame comparison image"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        title = f'2D Pose Comparison: GT vs YOLO - {sequence_name}\nFrame {frame_idx+1}/{total_frames}'
        if not np.isnan(frame_mpjpe):
            title += f' | Frame MPJPE: {frame_mpjpe:.1f}px'
        fig.suptitle(title, fontsize=14)

        for ax in [ax1, ax2]:
            ax.set_xlim(bounds['x_min'], bounds['x_max'])
            ax.set_ylim(bounds['y_min'], bounds['y_max'])
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        ax1.set_title('Ground Truth', fontsize=12)
        if not np.all(gt_frame == 0):
            for connection in CONNECTIONS_2D:
                joint1, joint2 = connection
                if joint1 < len(gt_frame) and joint2 < len(gt_frame):
                    x1, y1 = gt_frame[joint1]
                    x2, y2 = gt_frame[joint2]
                    ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
            for joint_idx, (x, y) in enumerate(gt_frame):
                if joint_idx != 14:
                    ax1.scatter(x, y, c='blue', s=40, alpha=0.9, edgecolors='darkblue', linewidth=1)
        else:
            ax1.text((bounds['x_min']+bounds['x_max'])/2, (bounds['y_min']+bounds['y_max'])/2, 
                     'No GT Data', ha='center', va='center', fontsize=14, color='red')

        ax2.set_title('YOLO Estimation', fontsize=12)
        if not np.all(yolo_frame == 0):
            for connection in CONNECTIONS_2D:
                joint1, joint2 = connection
                if joint1 < len(yolo_frame) and joint2 < len(yolo_frame):
                    x1, y1 = yolo_frame[joint1]
                    x2, y2 = yolo_frame[joint2]
                    ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
            for joint_idx, (x, y) in enumerate(yolo_frame):
                if joint_idx != 14:
                    ax2.scatter(x, y, c='red', s=40, alpha=0.9, edgecolors='darkred', linewidth=1)
        else:
            ax2.text((bounds['x_min']+bounds['x_max'])/2, (bounds['y_min']+bounds['y_max'])/2, 
                     'No YOLO Detection', ha='center', va='center', fontsize=14, color='red')

        for ax in [ax1, ax2]:
            ax.scatter(0, 0, c='green', s=100, marker='*', alpha=1.0, edgecolors='darkgreen', linewidth=2)
        ax1.set_xlabel('X (pixels)', fontsize=10)
        ax1.set_ylabel('Y (pixels)', fontsize=10)
        ax2.set_xlabel('X (pixels)', fontsize=10)
        ax2.set_ylabel('Y (pixels)', fontsize=10)

        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:06d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return frame_path
    except Exception as e:
        print(f"Error creating frame {frame_idx}: {e}")
        plt.close('all')
        return None

def create_gif_from_images(frame_paths, output_path, total_frames):
    """Create GIF from individual frame images"""
    if not Image:
        print("❌ PIL/Pillow required for GIF creation. Install with: pip install Pillow")
        return
    try:
        images = [Image.open(path) for path in frame_paths if os.path.exists(path)]
        if images:
            duration = 300 if total_frames <= 200 else 200 if total_frames <= 500 else 100 if total_frames <= 1000 else 50
            images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=True)
            print(f"✓ GIF created: {output_path}\n  Frames: {len(images)}\n  Duration per frame: {duration}ms")
        else:
            print("❌ No valid images to create GIF")
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")

def create_streaming_visualization(model, sequence_name, gt_poses_2d, total_frames, args, device):
    """Create visualization by streaming frames and saving as GIF"""
    if not args.save_video:
        print("ℹ️ Visualization skipped as --save-video not specified")
        return

    if not Image:
        print("❌ PIL/Pillow required for GIF creation. Install with: pip install Pillow")
        return

    print(f"🎬 Creating visualization for {total_frames} frames...")
    temp_dir = os.path.join(args.output_dir, f"temp_{sequence_name}")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    frame_paths = []

    bounds = calculate_visualization_bounds_streaming(model, sequence_name, gt_poses_2d, total_frames, args, device)
    batch_size = 50

    for start_idx in tqdm(range(0, total_frames, batch_size), desc="Creating visualization frames"):
        end_idx = min(start_idx + batch_size, total_frames)
        batch_frames_count = end_idx - start_idx
        frames_batch, _ = load_test_frames_batch(sequence_name, start_idx, batch_frames_count)
        if frames_batch is None:
            print(f"⚠️ Skipping batch {start_idx}-{end_idx} due to loading issues")
            continue

        gt_batch = gt_poses_2d[start_idx:end_idx]
        min_frames = min(len(frames_batch), len(gt_batch))
        frames_batch = frames_batch[:min_frames]
        gt_batch = gt_batch[:min_frames]
        yolo_batch, _ = estimate_yolo_poses_batch(model, frames_batch, args.img_size, device)
        gt_batch_pixel = convert_coordinates_to_pixels(gt_batch, frames_batch)
        gt_batch_root_rel = make_root_relative_2d_pixel(gt_batch_pixel, root_joint_idx=14)
        yolo_batch_root_rel = make_root_relative_2d_pixel(yolo_batch, root_joint_idx=14)

        for i in range(min_frames):
            frame_idx = start_idx + i
            gt_frame = gt_batch_root_rel[i]
            yolo_frame = yolo_batch_root_rel[i]
            frame_mpjpe = np.mean(np.linalg.norm(gt_frame - yolo_frame, axis=1)) if not (np.all(gt_frame == 0) or np.all(yolo_frame == 0)) else np.nan
            frame_path = create_single_frame_image(gt_frame, yolo_frame, frame_idx, total_frames, frame_mpjpe, sequence_name, temp_dir, bounds)
            if frame_path:
                frame_paths.append(frame_path)

        del frames_batch, gt_batch, yolo_batch, gt_batch_pixel, gt_batch_root_rel, yolo_batch_root_rel
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

    if frame_paths:
        model_name_clean = os.path.splitext(os.path.basename(args.model_path))[0]
        gif_path = os.path.join(args.output_dir, f'{sequence_name}_gt_vs_yolo_{model_name_clean}_all_frames.gif')
        create_gif_from_images(frame_paths, gif_path, total_frames)
        if frame_paths:
            static_path = os.path.join(args.output_dir, f'{sequence_name}_gt_vs_yolo_{model_name_clean}_all_frames.png')
            shutil.copy2(frame_paths[0], static_path)
            print(f"✓ Static image saved to: {static_path}")
        shutil.rmtree(temp_dir)
        print(f"✓ Visualization complete! GIF saved to: {gif_path}")
    else:
        print("❌ No frames could be processed for visualization")
        shutil.rmtree(temp_dir)

def print_sequence_results(metrics):
    """Print results for a single sequence"""
    if not metrics:
        return
    print(f"\nResults for {metrics['sequence']}:")
    print(f"  MPJPE: {metrics['avg_mpjpe']:.2f} pixels")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
    print(f"  PCK metrics:")
    for key, value in metrics['pck_results'].items():
        print(f"    {key}: {value*100:.2f}%")

def print_summary_results(all_metrics, model_name):
    """Print summary results for all sequences"""
    if not all_metrics:
        print("No results to summarize.")
        return

    print(f"\n{'='*80}\nCOMPREHENSIVE SUMMARY - YOLO vs Ground Truth (2D) [Confidence=0.2]\nModel: {model_name}\n{'='*80}")
    valid_metrics = [m for m in all_metrics if m is not None]
    if not valid_metrics:
        print("No valid metrics found.")
        return

    total_mpjpe_sum = sum(m['avg_mpjpe'] * m['valid_frames'] for m in valid_metrics)
    total_weighted_frames = sum(m['valid_frames'] for m in valid_metrics)
    overall_mpjpe = total_mpjpe_sum / total_weighted_frames if total_weighted_frames > 0 else 0
    avg_mpjpe = np.mean([m['avg_mpjpe'] for m in valid_metrics])
    avg_auc = np.mean([m['auc'] for m in valid_metrics])
    total_valid_frames = sum(m['valid_frames'] for m in valid_metrics)
    total_frames = sum(m['total_frames'] for m in valid_metrics)

    print(f"\nOVERALL METRICS:\n  Sequences processed: {len(valid_metrics)}\n  Total valid frames: {total_valid_frames}/{total_frames}")
    print(f"  Overall MPJPE (weighted): {overall_mpjpe:.2f} pixels\n  Average MPJPE (per sequence): {avg_mpjpe:.2f} pixels")
    print(f"  Average AUC: {avg_auc:.4f}")
    print(f"\nPCK METRICS (averaged across sequences):")
    for key in valid_metrics[0]['pck_results'].keys():
        avg_pck = np.mean([m['pck_results'][key] for m in valid_metrics])
        print(f"  {key}: {avg_pck*100:.2f}%")
    print(f"\nPER-SEQUENCE BREAKDOWN:\n{'Sequence':<10} {'MPJPE':<12} {'AUC':<10} {'Valid/Total':<12}\n{'-'*50}")
    for metrics in valid_metrics:
        print(f"{metrics['sequence']:<10} {metrics['avg_mpjpe']:<12.2f} {metrics['auc']:<10.4f} {metrics['valid_frames']}/{metrics['total_frames']:<12}")
    print(f"{'='*80}\n🎯 FINAL OVERALL MPJPE: {overall_mpjpe:.2f} pixels (Confidence=0.2)\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Compare Ground Truth and YOLO 2D poses with comprehensive metrics')
    parser.add_argument('--sequence', type=str, default='TS1', help='Sequence to compare (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--num-frames', type=int, default=None, help='Number of frames to process (default: all frames)')
    parser.add_argument('--all', action='store_true', help='Run evaluation on all available sequences')
    parser.add_argument('--save-video', action='store_true', help='Save comparison as GIF for each sequence')
    parser.add_argument('--output-dir', type=str, default='comparison_output', help='Directory to save outputs')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size for YOLO inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    args = parser.parse_args()

    print("🎯 Ground Truth vs YOLO 2D Pose Comparison with Comprehensive Metrics [Confidence=0.2]\n" + "="*80)
    if args.save_video and not Image:
        print("❌ PIL/Pillow required for GIF creation. Install with: pip install Pillow")
        return

    device, gpu_available = check_gpu_availability(args.device)
    print(f"Model: {args.model_path}\nDevice: {device}\nYOLO Confidence Threshold: 0.2")
    print(f"Mode: {'All sequences' if args.all else f'Sequence {args.sequence}'} with {args.num_frames if args.num_frames else 'all'} frames")
    print(f"Input size: {args.img_size}\nMetrics: MPJPE, PCK, AUC\nCoordinate system: Root-relative poses in pixel domain\n" + "="*80)

    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return

    print(f"🤖 Loading YOLO model from {args.model_path}...")
    try:
        model = YOLO(args.model_path)
        if gpu_available and device.startswith('cuda'):
            model.to(device)
        print("✓ YOLO model loaded successfully")
        print(f"Model device: {next(model.model.parameters()).device if hasattr(model, 'model') else 'Unknown'}")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return

    model_name = os.path.basename(args.model_path)
    all_metrics = []

    if args.all:
        print(f"\n🔄 Processing all available sequences...")
        available_sequences = get_available_sequences()
        print(f"Available sequences: {available_sequences}")
        for sequence in available_sequences:
            try:
                metrics = process_single_sequence(model, sequence, args, device)
                all_metrics.append(metrics)
                if metrics:
                    print_sequence_results(metrics)
                    if args.save_video:
                        gt_poses_2d, _, _ = load_test_3d_data_from_dataset(sequence)
                        if gt_poses_2d is not None:
                            total_frames = min(args.num_frames, len(gt_poses_2d)) if args.num_frames is not None else len(gt_poses_2d)
                            create_streaming_visualization(model, sequence, gt_poses_2d, total_frames, args, device)
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"❌ Error processing sequence {sequence}: {e}")
                all_metrics.append(None)
        print_summary_results(all_metrics, model_name)
    else:
        metrics = process_single_sequence(model, args.sequence, args, device)
        all_metrics.append(metrics)
        if metrics:
            print(f"\n{'='*60}\nDETAILED RESULTS FOR {args.sequence} [Confidence=0.2]\n{'='*60}")
            print(f"Model: {model_name}\nDevice: {device}\nValid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
            print(f"\nMPJPE: {metrics['avg_mpjpe']:.2f} pixels")
            print(f"\nPCK (Percentage of Correct Keypoints):")
            for key, value in metrics['pck_results'].items():
                print(f"  {key}: {value*100:.2f}%")
            print(f"\nAUC: {metrics['auc']:.4f}")
            print(f"\nJoint-wise errors (top 5 worst):")
            joint_error_pairs = [(i, metrics['joint_errors'][i], JOINT_NAMES[i]) for i in range(17)]
            joint_error_pairs.sort(key=lambda x: x[1], reverse=True)
            for i, (joint_idx, error, name) in enumerate(joint_error_pairs[:5]):
                print(f"  {name} (joint {joint_idx}): {error:.2f} pixels")
            print(f"{'='*60}")
            if args.save_video:
                gt_poses_2d, _, _ = load_test_3d_data_from_dataset(args.sequence)
                if gt_poses_2d is not None:
                    total_frames = min(args.num_frames, len(gt_poses_2d)) if args.num_frames is not None else len(gt_poses_2d)
                    create_streaming_visualization(model, args.sequence, gt_poses_2d, total_frames, args, device)  # Fixed: Changed 'sequence' to 'args.sequence'
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()