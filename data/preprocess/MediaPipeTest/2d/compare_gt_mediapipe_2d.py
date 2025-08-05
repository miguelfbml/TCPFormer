'''
# Single sequence with visualization
python compare_gt_mediapipe_2d.py --sequence TS1 --save-video

# Process all sequences (comprehensive metrics)
python compare_gt_mediapipe_2d.py --all

# Interactive visualization with 100 frames
python compare_gt_mediapipe_2d.py --sequence TS2 --num-frames 100
'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from tqdm import tqdm
import torch

# Navigate to project root
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

# Import utility functions from your workspace
from utils.utils_3dhp import AccumLoss

# MPI-INF-3DHP skeleton connections for 2D visualization
connections_2d = [
    (1, 2), (2, 3), (3, 4),  # Right arm
    (1, 5), (5, 6), (6, 7),          # Left arm
    (14, 8), (8, 9), (9, 10),  # Right leg
    (14, 11), (11, 12), (12, 13),    # Left leg
    (0, 16), (16, 1), (1, 15), (15,14)       # Spine and head
]

# Joint names for reference
JOINT_NAMES = [
    'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

def load_datasets():
    """Load both ground truth and MediaPipe datasets"""
    gt_path = 'nas-ctm01/homes/mfbrandao/TCPFormerForked/data/preprocess/MediaPipeTest/data1/motion3d/data_test_3dhp.npz'
    mp_path = 'nas-ctm01/homes/mfbrandao/TCPFormerForked/data/preprocess/MediaPipeTest/data1/motion3d/data_test_3dhp_mediapipe.npz'

    print("Loading datasets...")
    
    if not os.path.exists(gt_path):
        print(f"ERROR: Ground truth dataset not found: {gt_path}")
        return None, None
    
    gt_data = np.load(gt_path, allow_pickle=True)['data'].item()
    print(f"✓ Ground truth loaded: {list(gt_data.keys())}")
    
    if not os.path.exists(mp_path):
        print(f"ERROR: MediaPipe dataset not found: {mp_path}")
        return None, None
    
    mp_data = np.load(mp_path, allow_pickle=True)['data'].item()
    print(f"✓ MediaPipe loaded: {list(mp_data.keys())}")
    
    return gt_data, mp_data

def denormalize_poses_2d_correct(poses_2d, seq_name):
    """
    Denormalize 2D poses to pixel coordinates, handling various input ranges
    """
    # Get image dimensions for this sequence
    if seq_name in ['TS5', 'TS6']:
        width, height = 1920, 1080
    else:
        width, height = 2048, 2048
    
    denorm_poses = poses_2d.copy()
    
    # Check input range
    x_min, x_max = np.min(poses_2d[:, :, 0]), np.max(poses_2d[:, :, 0])
    y_min, y_max = np.min(poses_2d[:, :, 1]), np.max(poses_2d[:, :, 1])
    
    # If poses are not in [-1, 1], normalize them first assuming they are in pixel coordinates
    if x_max > 1.0 or x_min < -1.0 or y_max > 1.0 or y_min < -1.0:
        # Normalize to [0, 1] based on image dimensions
        denorm_poses[:, :, 0] = (poses_2d[:, :, 0] / width)
        denorm_poses[:, :, 1] = (poses_2d[:, :, 1] / height)
        # Then convert to [-1, 1]
        denorm_poses[:, :, 0] = denorm_poses[:, :, 0] * 2.0 - 1.0
        denorm_poses[:, :, 1] = denorm_poses[:, :, 1] * 2.0 - 1.0
    
    # Denormalize from [-1, 1] to pixel coordinates
    denorm_poses[:, :, 0] = (denorm_poses[:, :, 0] + 1.0) * width / 2.0
    denorm_poses[:, :, 1] = (denorm_poses[:, :, 1] + 1.0) * height / 2.0
    
    return denorm_poses

def make_root_relative_2d_pixel(poses_2d, root_joint_idx=14):
    """
    Make poses root-relative by subtracting root joint position from all joints
    """
    root_relative_poses = poses_2d.copy()
    
    for frame_idx in range(poses_2d.shape[0]):
        frame = poses_2d[frame_idx]
        root_pos = frame[root_joint_idx]
        
        if not np.all(frame == 0):
            root_relative_poses[frame_idx] = frame - root_pos[np.newaxis, :]
            root_relative_poses[frame_idx, root_joint_idx] = [0.0, 0.0]
    
    return root_relative_poses

def calculate_torso_diameter_2d(poses_2d, left_shoulder_idx=11, right_shoulder_idx=14, 
                               left_hip_idx=4, right_hip_idx=1):
    """
    Calculate torso diameter for 2D PCK metric
    Similar to 3D version but for 2D coordinates
    """
    # poses_2d shape: (N, 17, 2)
    N = poses_2d.shape[0]
    torso_diameters = np.zeros(N)
    
    for i in range(N):
        frame = poses_2d[i]
        
        # Check if frame has valid data
        if np.all(frame == 0):
            torso_diameters[i] = 0
            continue
            
        # Calculate shoulder distance
        left_shoulder = frame[left_shoulder_idx]
        right_shoulder = frame[right_shoulder_idx]
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Calculate hip distance
        left_hip = frame[left_hip_idx]
        right_hip = frame[right_hip_idx]
        hip_dist = np.linalg.norm(left_hip - right_hip)
        
        # Torso diameter is average of shoulder and hip distances
        torso_diameter = (shoulder_dist + hip_dist) / 2.0
        torso_diameters[i] = torso_diameter
    
    return torso_diameters

def compute_pck_2d(pred, gt, torso_diameters=None, fixed_threshold=150.0, pck_thresholds=[0.9, 0.8, 0.7]):
    """
    Compute PCK metrics for 2D poses
    Similar to 3D version but adapted for 2D coordinates
    """
    # pred, gt shape: (N, 17, 2)
    N, J, _ = pred.shape
    
    # Calculate joint errors (Euclidean distance in 2D)
    joint_errors = np.linalg.norm(pred - gt, axis=2)  # (N, 17)
    
    pck_results = {}
    
    # Torso-based PCK if torso diameters are provided
    if torso_diameters is not None:
        for percentage in [10, 20, 50, 80]:
            threshold = torso_diameters[:, None] * (percentage / 100.0)  # (N, 1)
            correct = joint_errors < threshold  # (N, 17)
            pck = np.mean(correct)  # Overall average across all joints and frames
            pck_results[f'PCK@{percentage}%_torso'] = pck
    
    # Fixed threshold PCK (e.g., 150 pixels)
    if fixed_threshold is not None:
        for percentage in [10, 20, 50, 80]:
            threshold = fixed_threshold * (percentage / 100.0)
            correct = joint_errors < threshold
            pck = np.mean(correct)
            pck_results[f'PCK@{percentage}%_{int(fixed_threshold)}px'] = pck
    
    return pck_results

def compute_auc_2d(pred, gt, max_threshold=150.0, num_thresholds=31):
    """
    Compute AUC metric for 2D poses
    Similar to 3D version but for 2D coordinates
    """
    # pred, gt shape: (N, 17, 2)
    joint_errors = np.linalg.norm(pred - gt, axis=2)  # (N, 17)
    thresholds = np.linspace(0, max_threshold, num_thresholds)
    
    pck_values = []
    for threshold in thresholds:
        correct = joint_errors < threshold
        pck = np.mean(correct)
        pck_values.append(pck)
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(pck_values, thresholds) / max_threshold
    return auc

def compute_mpjpe_2d(gt_poses_2d, mp_poses_2d):
    """
    Compute comprehensive metrics including MPJPE, PCK, and AUC for 2D poses
    """
    min_frames = min(len(gt_poses_2d), len(mp_poses_2d))
    gt_poses = gt_poses_2d[:min_frames]
    mp_poses = mp_poses_2d[:min_frames]
    
    # Find valid frames (non-zero poses)
    valid_frames = []
    valid_gt_list = []
    valid_mp_list = []
    frame_mpjpe = []
    
    for frame_idx in range(min_frames):
        gt_frame = gt_poses[frame_idx]
        mp_frame = mp_poses[frame_idx]
        
        gt_valid = not np.all(gt_frame == 0)
        mp_valid = not np.all(mp_frame == 0)
        
        if gt_valid and mp_valid:
            valid_frames.append(frame_idx)
            valid_gt_list.append(gt_frame)
            valid_mp_list.append(mp_frame)
            
            # Calculate frame MPJPE
            joint_diffs = np.linalg.norm(gt_frame - mp_frame, axis=1)
            frame_error = np.mean(joint_diffs)
            frame_mpjpe.append(frame_error)
        else:
            frame_mpjpe.append(np.nan)
    
    if len(valid_gt_list) == 0:
        return None
    
    # Convert to numpy arrays
    valid_gt = np.array(valid_gt_list)  # (V, 17, 2)
    valid_mp = np.array(valid_mp_list)  # (V, 17, 2)
    
    # Calculate MPJPE
    avg_mpjpe = np.mean([e for e in frame_mpjpe if not np.isnan(e)])
    
    # Calculate joint-wise errors
    joint_errors = np.mean(np.linalg.norm(valid_gt - valid_mp, axis=2), axis=0)  # (17,)
    
    # Calculate torso diameters for PCK
    torso_diameters = calculate_torso_diameter_2d(valid_gt)
    
    # Compute PCK metrics
    pck_results = compute_pck_2d(valid_mp, valid_gt, torso_diameters, fixed_threshold=150.0)
    
    # Compute AUC
    auc = compute_auc_2d(valid_mp, valid_gt, max_threshold=150.0)
    
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

def process_single_sequence(gt_data, mp_data, seq_name):
    """Process a single sequence and return its comprehensive metrics"""
    # Extract 2D poses
    gt_poses_2d_norm = gt_data[seq_name]['data_2d']
    mp_poses_2d_norm = mp_data[seq_name]['data_2d']
    
    # Denormalize to pixel coordinates
    gt_poses_2d_pixel = denormalize_poses_2d_correct(gt_poses_2d_norm, seq_name)
    mp_poses_2d_pixel = denormalize_poses_2d_correct(mp_poses_2d_norm, seq_name)
    
    # Make root-relative
    gt_poses_2d_root_rel = make_root_relative_2d_pixel(gt_poses_2d_pixel, root_joint_idx=14)
    mp_poses_2d_root_rel = make_root_relative_2d_pixel(mp_poses_2d_pixel, root_joint_idx=14)
    
    # Compute comprehensive metrics
    metrics = compute_mpjpe_2d(gt_poses_2d_root_rel, mp_poses_2d_root_rel)
    
    return metrics

def process_all_sequences(gt_data, mp_data):
    """Process all sequences and compute overall average metrics including PCK and AUC"""
    print("\n" + "="*70)
    print("PROCESSING ALL SEQUENCES - COMPUTING COMPREHENSIVE METRICS")
    print("="*70)
    
    all_sequences = sorted(gt_data.keys())
    sequence_results = {}
    
    # Overall statistics
    total_valid_frames = 0
    total_frame_errors = []
    total_joint_errors = np.zeros(17)
    
    # For PCK and AUC averaging
    total_pck_sums = {
        'PCK@10%_torso': 0.0, 'PCK@20%_torso': 0.0, 'PCK@50%_torso': 0.0, 'PCK@80%_torso': 0.0,
        'PCK@10%_150px': 0.0, 'PCK@20%_150px': 0.0, 'PCK@50%_150px': 0.0, 'PCK@80%_150px': 0.0
    }
    total_auc_sum = 0.0
    valid_sequences = 0
    
    print(f"Found {len(all_sequences)} sequences: {all_sequences}")
    print("\nProcessing sequences...")
    
    for seq_name in tqdm(all_sequences, desc="Sequences"):
        if seq_name in mp_data:
            metrics = process_single_sequence(gt_data, mp_data, seq_name)
            
            if metrics:
                sequence_results[seq_name] = metrics
                valid_sequences += 1
                
                # Accumulate statistics
                total_valid_frames += metrics['valid_frames']
                
                # Add valid frame errors to total
                valid_frame_errors = [e for e in metrics['frame_mpjpe'] if not np.isnan(e)]
                total_frame_errors.extend(valid_frame_errors)
                
                # Add joint errors (weighted by number of valid frames)
                joint_errors_weighted = np.array(metrics['joint_errors']) * metrics['valid_frames']
                total_joint_errors += joint_errors_weighted
                
                # Accumulate PCK results
                for key in total_pck_sums:
                    if key in metrics['pck_results']:
                        total_pck_sums[key] += metrics['pck_results'][key]
                
                # Accumulate AUC
                total_auc_sum += metrics['auc']
                
                print(f"  ✓ {seq_name}: {metrics['avg_mpjpe']:.1f} px, "
                      f"AUC: {metrics['auc']:.4f}, "
                      f"PCK@50%_150px: {metrics['pck_results']['PCK@50%_150px']*100:.1f}% "
                      f"({metrics['valid_frames']}/{metrics['total_frames']} frames)")
            else:
                print(f"  ✗ {seq_name}: No valid data")
        else:
            print(f"  ✗ {seq_name}: Not found in MediaPipe data")
    
    # Compute overall statistics
    if total_valid_frames > 0 and len(total_frame_errors) > 0 and valid_sequences > 0:
        overall_avg_mpjpe = np.mean(total_frame_errors)
        overall_joint_errors = total_joint_errors / total_valid_frames
        
        # Average PCK results
        overall_pck_results = {}
        for key in total_pck_sums:
            overall_pck_results[key] = total_pck_sums[key] / valid_sequences
        
        # Average AUC
        overall_auc = total_auc_sum / valid_sequences
        
        print(f"\n" + "="*70)
        print("OVERALL RESULTS ACROSS ALL SEQUENCES")
        print("="*70)
        print(f"Total valid frames: {total_valid_frames:,}")
        print(f"Total sequences processed: {valid_sequences}")
        print(f"Overall Average MPJPE: {overall_avg_mpjpe:.1f} pixels")
        print(f"Overall AUC: {overall_auc:.4f}")
        
        print(f"\nOverall PCK Results:")
        for key, value in overall_pck_results.items():
            print(f"  {key}: {value*100:.2f}%")
        
        print(f"\nOverall Joint Errors (top 5):")
        joint_error_pairs = [(i, overall_joint_errors[i], JOINT_NAMES[i]) for i in range(17)]
        joint_error_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (joint_idx, error, name) in enumerate(joint_error_pairs[:5]):
            print(f"  {name} (joint {joint_idx}): {error:.1f} pixels")
        
        print(f"\nPer-sequence breakdown:")
        for seq_name in sorted(sequence_results.keys()):
            metrics = sequence_results[seq_name]
            print(f"  {seq_name}: MPJPE={metrics['avg_mpjpe']:.1f}px, "
                  f"AUC={metrics['auc']:.3f}, "
                  f"PCK@50%_150px={metrics['pck_results']['PCK@50%_150px']*100:.1f}% "
                  f"({metrics['valid_frames']:,} frames)")
        
        return {
            'overall_avg_mpjpe': overall_avg_mpjpe,
            'overall_auc': overall_auc,
            'overall_pck_results': overall_pck_results,
            'sequence_results': sequence_results
        }
    else:
        print("ERROR: No valid data found across all sequences!")
        return None

def analyze_coordinate_ranges(gt_poses_2d, mp_poses_2d, seq_name):
    """Analyze coordinate ranges for both datasets"""
    print(f"\nCoordinate analysis for {seq_name}:")
    print(f"Ground Truth 2D:")
    print(f"  Shape: {gt_poses_2d.shape}")
    print(f"  X range: [{np.min(gt_poses_2d[:, :, 0]):.1f}, {np.max(gt_poses_2d[:, :, 0]):.1f}] pixels")
    print(f"  Y range: [{np.min(gt_poses_2d[:, :, 1]):.1f}, {np.max(gt_poses_2d[:, :, 1]):.1f}] pixels")
    
    print(f"MediaPipe 2D:")
    print(f"  Shape: {mp_poses_2d.shape}")
    print(f"  X range: [{np.min(mp_poses_2d[:, :, 0]):.1f}, {np.max(mp_poses_2d[:, :, 0]):.1f}] pixels")
    print(f"  Y range: [{np.min(mp_poses_2d[:, :, 1]):.1f}, {np.max(mp_poses_2d[:, :, 1]):.1f}] pixels")
    
    gt_zeros = np.sum(np.all(gt_poses_2d == 0, axis=(1, 2)))
    mp_zeros = np.sum(np.all(mp_poses_2d == 0, axis=(1, 2)))
    print(f"Zero poses: GT={gt_zeros}/{len(gt_poses_2d)}, MP={mp_zeros}/{len(mp_poses_2d)}")

def create_comparison_visualization(gt_poses_2d, mp_poses_2d, seq_name, metrics, args):
    """Create side-by-side comparison visualization with comprehensive metrics"""
    min_frames = min(len(gt_poses_2d), len(mp_poses_2d), args.num_frames)
    
    if min_frames == 0:
        print("No frames to visualize!")
        return None, None, 0
    
    gt_poses = gt_poses_2d[:min_frames]
    mp_poses = mp_poses_2d[:min_frames]
    frame_mpjpe = metrics['frame_mpjpe'][:min_frames] if metrics else [np.nan] * min_frames
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Enhanced title with all metrics
    if metrics:
        title = (f'2D Pose Comparison: GT vs MediaPipe - {seq_name} (Root-Relative, Pixels)\n'
                f'Avg MPJPE: {metrics["avg_mpjpe"]:.1f}px | '
                f'AUC: {metrics["auc"]:.4f} | '
                f'PCK@50%_150px: {metrics["pck_results"]["PCK@50%_150px"]*100:.1f}%')
    else:
        title = f'2D Pose Comparison: GT vs MediaPipe - {seq_name} (Root-Relative, Pixels)'
    
    fig.suptitle(title, fontsize=14)
    
    all_gt_non_root = gt_poses[:, :, :].reshape(-1, 2)
    all_mp_non_root = mp_poses[:, :, :].reshape(-1, 2)
    
    valid_gt = all_gt_non_root[~np.all(all_gt_non_root == 0, axis=1)]
    valid_mp = all_mp_non_root[~np.all(all_mp_non_root == 0, axis=1)]
    
    if len(valid_gt) > 0 and len(valid_mp) > 0:
        all_points = np.vstack([valid_gt, valid_mp])
        x_range = np.max(np.abs(all_points[:, 0]))
        y_range = np.max(np.abs(all_points[:, 1]))
        max_range = max(x_range, y_range) * 1.1
        x_min, x_max = -max_range, max_range
        y_min, y_max = -max_range, max_range
    else:
        x_min, x_max = -500, 500
        y_min, y_max = -500, 500
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        for ax in [ax1, ax2]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (pixels, root-relative)')
            ax.set_ylabel('Y (pixels, root-relative)')
            
            ax.scatter(0, 0, c='green', s=100, marker='*', alpha=1.0, 
                      edgecolors='darkgreen', linewidth=2, zorder=10)
            ax.text(20, 20, 'Root(14)', fontsize=10, ha='left', va='bottom', 
                   color='green', weight='bold')
        
        ax1.set_title(f'Ground Truth\nFrame {frame_idx+1}/{min_frames}', fontsize=14)
        gt_frame = gt_poses[frame_idx]
        
        gt_valid = not np.all(gt_frame == 0)
        if gt_valid:
            for connection in connections_2d:
                joint1, joint2 = connection
                if joint1 < len(gt_frame) and joint2 < len(gt_frame):
                    x1, y1 = gt_frame[joint1]
                    x2, y2 = gt_frame[joint2]
                    ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
            
            for joint_idx, (x, y) in enumerate(gt_frame):
                if joint_idx != 14:
                    ax1.scatter(x, y, c='blue', s=60, alpha=0.9, edgecolors='darkblue', linewidth=1)
                    ax1.text(x+15, y+15, str(joint_idx), fontsize=9, ha='left', va='bottom', 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        else:
            ax1.text(0, 50, 'No GT Data', ha='center', va='center', fontsize=16, color='red')
        
        # Enhanced frame title with frame-specific MPJPE
        frame_mpjpe_val = frame_mpjpe[frame_idx] if not np.isnan(frame_mpjpe[frame_idx]) else 0
        ax2.set_title(f'MediaPipe Estimation\nFrame {frame_idx+1}/{min_frames} | '
                     f'Frame MPJPE: {frame_mpjpe_val:.1f}px', fontsize=14)
        mp_frame = mp_poses[frame_idx]
        
        mp_valid = not np.all(mp_frame == 0)
        if mp_valid:
            for connection in connections_2d:
                joint1, joint2 = connection
                if joint1 < len(mp_frame) and joint2 < len(mp_frame):
                    x1, y1 = mp_frame[joint1]
                    x2, y2 = mp_frame[joint2]
                    ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
            
            for joint_idx, (x, y) in enumerate(mp_frame):
                if joint_idx != 14:
                    ax2.scatter(x, y, c='red', s=60, alpha=0.9, edgecolors='darkred', linewidth=1)
                    ax2.text(x+15, y+15, str(joint_idx), fontsize=9, ha='left', va='bottom', 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        else:
            ax2.text(0, 50, 'No MediaPipe Data', ha='center', va='center', fontsize=16, color='red')
        
        plt.tight_layout()
        return [ax1, ax2]
    
    return update, fig, min_frames

def main():
    parser = argparse.ArgumentParser(description='Compare Ground Truth and MediaPipe 2D poses with comprehensive metrics (MPJPE, PCK, AUC)')
    parser.add_argument('--sequence', type=str, default='TS1', 
                       help='Sequence to compare (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--all', action='store_true',
                       help='Process all sequences and compute overall averages (no visualization)')
    parser.add_argument('--num-frames', type=int, default=50,
                       help='Number of frames to visualize (ignored with --all)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save comparison as GIF (ignored with --all)')
    parser.add_argument('--output-dir', type=str, default='comparison_output',
                       help='Directory to save outputs')
    args = parser.parse_args()
    
    print("Ground Truth vs MediaPipe 2D Pose Comparison with Comprehensive Metrics")
    print("Metrics: MPJPE, PCK (Percentage of Correct Keypoints), AUC (Area Under Curve)")
    print("=" * 80)
    
    # Load datasets
    gt_data, mp_data = load_datasets()
    if gt_data is None or mp_data is None:
        return
    
    if args.all:
        # Process all sequences
        print("Processing ALL sequences (no visualization)")
        results = process_all_sequences(gt_data, mp_data)
        return
    
    # Single sequence processing (original functionality)
    print(f"Sequence: {args.sequence}")
    print(f"Frames to visualize: {args.num_frames}")
    
    # Check if sequence exists
    if args.sequence not in gt_data:
        print(f"ERROR: Sequence {args.sequence} not found in ground truth data")
        print(f"Available sequences: {list(gt_data.keys())}")
        return
    
    if args.sequence not in mp_data:
        print(f"ERROR: Sequence {args.sequence} not found in MediaPipe data")
        print(f"Available sequences: {list(mp_data.keys())}")
        return
    
    # Extract poses
    gt_poses_2d_norm = gt_data[args.sequence]['data_2d']
    mp_poses_2d_norm = mp_data[args.sequence]['data_2d']
    
    print(f"\n✓ Loaded sequence {args.sequence}")
    print(f"GT frames: {len(gt_poses_2d_norm)}, MP frames: {len(mp_poses_2d_norm)}")
    
    print(f"\nDenormalizing poses to pixel coordinates...")
    gt_poses_2d_pixel = denormalize_poses_2d_correct(gt_poses_2d_norm, args.sequence)
    mp_poses_2d_pixel = denormalize_poses_2d_correct(mp_poses_2d_norm, args.sequence)
    
    print(f"\nMaking both datasets root-relative in pixel domain...")
    gt_poses_2d_root_rel = make_root_relative_2d_pixel(gt_poses_2d_pixel, root_joint_idx=14)
    mp_poses_2d_root_rel = make_root_relative_2d_pixel(mp_poses_2d_pixel, root_joint_idx=14)
    
    analyze_coordinate_ranges(gt_poses_2d_root_rel, mp_poses_2d_root_rel, args.sequence)
    
    print(f"\nComputing comprehensive metrics (MPJPE, PCK, AUC) in pixel domain...")
    metrics = compute_mpjpe_2d(gt_poses_2d_root_rel, mp_poses_2d_root_rel)
    
    if metrics:
        print(f"\n" + "="*60)
        print(f"COMPREHENSIVE 2D POSE EVALUATION RESULTS")
        print(f"="*60)
        print(f"Sequence: {args.sequence}")
        print(f"Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
        print(f"\nMPJPE (Mean Per Joint Position Error):")
        print(f"  Average MPJPE: {metrics['avg_mpjpe']:.2f} pixels")
        
        print(f"\nPCK (Percentage of Correct Keypoints):")
        for key, value in metrics['pck_results'].items():
            print(f"  {key}: {value*100:.2f}%")
        
        print(f"\nAUC (Area Under Curve):")
        print(f"  AUC: {metrics['auc']:.4f}")
        
        print(f"\nJoint-wise errors (top 5 worst):")
        joint_error_pairs = [(i, metrics['joint_errors'][i], JOINT_NAMES[i]) for i in range(17)]
        joint_error_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (joint_idx, error, name) in enumerate(joint_error_pairs[:5]):
            print(f"  {name} (joint {joint_idx}): {error:.2f} pixels")
        
        print(f"="*60)
    
    # Create visualization
    print(f"\nCreating visualization...")
    try:
        result = create_comparison_visualization(
            gt_poses_2d_root_rel, mp_poses_2d_root_rel, args.sequence, metrics, args)
        
        if result[0] is None:
            return
            
        update_func, fig, min_frames = result
        
        if args.save_video:
            print("Creating animation...")
            ani = FuncAnimation(fig, update_func, frames=min_frames, 
                              interval=300, repeat=True, blit=False)
            
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, 
                                     f'{args.sequence}_gt_vs_mediapipe_2d_comprehensive.gif')
            
            ani.save(output_path, writer='pillow', fps=3, dpi=100)
            print(f"✓ Animation saved to: {output_path}")
            
            update_func(0)
            static_path = os.path.join(args.output_dir, 
                                     f'{args.sequence}_gt_vs_mediapipe_2d_comprehensive.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"✓ Static image saved to: {static_path}")
            
            plt.close(fig)
        else:
            print("Showing interactive visualization...")
            update_func(0)
            plt.show()
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()