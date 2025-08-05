"""
Calculate 2D metrics (MPJPE, PCK, AUC) for MediaPipe vs Ground Truth 2D poses
This denormalizes the poses back to pixel coordinates for meaningful metrics

Usage:
python calculate_metrics_mediapipe.py --sequence TS1 --save-video
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from tqdm import tqdm
import cv2
import glob

# Navigate to project root
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

# MPI-INF-3DHP skeleton connections for 2D visualization
connections_2d = [
    (1, 2), (2, 3), (3, 4),  # Right arm
    (1, 5), (5, 6), (6, 7),  # Left arm
    (14, 8), (8, 9), (9, 10),  # Right leg
    (14, 11), (11, 12), (12, 13),  # Left leg
    (0, 16), (16, 1), (1, 15), (15, 14)  # Spine and head
]

# Joint names for reference
JOINT_NAMES = [
    'Head_Top', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

def load_datasets():
    """Load both ground truth and MediaPipe datasets"""
    gt_path = '/nas-ctm01/homes/mfbrandao/TCPFormerForked/data/motion3d/data_test_3dhp.npz'
    mp_path = '/nas-ctm01/homes/mfbrandao/TCPFormerForked/data/motion3d/data_test_3dhp_mediapipe.npz'
    
    print("Loading datasets...")
    
    # Load ground truth
    if not os.path.exists(gt_path):
        print(f"ERROR: Ground truth dataset not found: {gt_path}")
        return None, None
    
    gt_data = np.load(gt_path, allow_pickle=True)['data'].item()
    print(f"✓ Ground truth loaded: {list(gt_data.keys())}")
    
    # Load MediaPipe
    if not os.path.exists(mp_path):
        print(f"ERROR: MediaPipe dataset not found: {mp_path}")
        return None, None
    
    mp_data = np.load(mp_path, allow_pickle=True)['data'].item()
    print(f"✓ MediaPipe loaded: {list(mp_data.keys())}")
    
    return gt_data, mp_data

def denormalize_2d_poses(poses_2d_norm, seq_name):
    """
    Denormalize 2D poses from [-1, 1] back to pixel coordinates
    This reverses the normalization done in the Fusion dataset
    """
    # Get image dimensions for this sequence (same as MPI3DHP.normalize_poses())
    if seq_name in ['TS5', 'TS6']:
        width, height = 1920, 1080
    else:
        width, height = 2048, 2048
    
    print(f"Denormalizing {seq_name} from [-1,1] to pixel coordinates ({width}x{height})")
    
    # Reverse the normalization: normalized = (pixel / width) * 2 - [1, height/width]
    # So: pixel = (normalized + [1, height/width]) * width / 2
    
    poses_pixel = poses_2d_norm.copy()
    
    # For X coordinates: pixel_x = (norm_x + 1) * width / 2
    poses_pixel[:, :, 0] = (poses_2d_norm[:, :, 0] + 1.0) * width / 2.0
    
    # For Y coordinates: pixel_y = (norm_y + height/width) * width / 2
    poses_pixel[:, :, 1] = (poses_2d_norm[:, :, 1] + height/width) * width / 2.0
    
    print(f"Denormalized pixel range:")
    print(f"  X: [{np.min(poses_pixel[:, :, 0]):.1f}, {np.max(poses_pixel[:, :, 0]):.1f}] (expected: [0, {width}])")
    print(f"  Y: [{np.min(poses_pixel[:, :, 1]):.1f}, {np.max(poses_pixel[:, :, 1]):.1f}] (expected: [0, {height}])")
    
    return poses_pixel

def compute_2d_mpjpe(gt_poses_pixel, mp_poses_pixel):
    """Compute 2D MPJPE in pixel coordinates"""
    # Ensure same number of frames
    min_frames = min(len(gt_poses_pixel), len(mp_poses_pixel))
    gt_poses = gt_poses_pixel[:min_frames]
    mp_poses = mp_poses_pixel[:min_frames]
    
    frame_errors = []
    joint_errors = np.zeros(17)
    valid_frame_count = 0
    
    for frame_idx in range(min_frames):
        gt_frame = gt_poses[frame_idx]  # (17, 2)
        mp_frame = mp_poses[frame_idx]  # (17, 2)
        
        # Check if both frames have valid data (not all zeros)
        gt_valid = not np.all(gt_frame == 0)
        mp_valid = not np.all(mp_frame == 0)
        
        if gt_valid and mp_valid:
            # Compute L2 distance per joint in pixels
            joint_diffs = np.linalg.norm(gt_frame - mp_frame, axis=1)
            frame_error = np.mean(joint_diffs)
            frame_errors.append(frame_error)
            joint_errors += joint_diffs
            valid_frame_count += 1
    
    if valid_frame_count > 0:
        avg_mpjpe = np.mean(frame_errors)
        joint_errors /= valid_frame_count
        
        print(f"\n2D MPJPE Results (pixel coordinates):")
        print(f"Valid frames: {valid_frame_count}/{min_frames}")
        print(f"Average 2D MPJPE: {avg_mpjpe:.2f} pixels")
        print(f"Joint errors (worst 5):")
        
        # Sort joints by error
        joint_error_pairs = [(i, joint_errors[i], JOINT_NAMES[i]) for i in range(17)]
        joint_error_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (joint_idx, error, name) in enumerate(joint_error_pairs[:5]):
            print(f"  {name} (joint {joint_idx}): {error:.2f} pixels")
        
        return {
            'mpjpe_2d': float(avg_mpjpe),
            'joint_errors': [float(x) for x in joint_errors],
            'valid_frames': int(valid_frame_count),
            'total_frames': int(min_frames),
            'frame_errors': frame_errors  # Keep individual frame errors for visualization
        }
    else:
        print("No valid frames found for comparison!")
        return None

def compute_2d_pck(gt_poses_pixel, mp_poses_pixel, thresholds=[5, 10, 20, 30, 50]):
    """Compute 2D PCK at different pixel thresholds"""
    min_frames = min(len(gt_poses_pixel), len(mp_poses_pixel))
    gt_poses = gt_poses_pixel[:min_frames]
    mp_poses = mp_poses_pixel[:min_frames]
    
    pck_results = {}
    valid_frame_count = 0
    all_joint_errors = []
    
    for frame_idx in range(min_frames):
        gt_frame = gt_poses[frame_idx]  # (17, 2)
        mp_frame = mp_poses[frame_idx]  # (17, 2)
        
        # Check if both frames have valid data
        gt_valid = not np.all(gt_frame == 0)
        mp_valid = not np.all(mp_frame == 0)
        
        if gt_valid and mp_valid:
            # Compute L2 distance per joint
            joint_diffs = np.linalg.norm(gt_frame - mp_frame, axis=1)  # (17,)
            all_joint_errors.append(joint_diffs)
            valid_frame_count += 1
    
    if valid_frame_count > 0:
        all_joint_errors = np.stack(all_joint_errors, axis=0)  # (valid_frames, 17)
        
        print(f"\n2D PCK Results (pixel thresholds):")
        for threshold in thresholds:
            correct = all_joint_errors < threshold  # (valid_frames, 17)
            pck = np.mean(correct) * 100  # Overall percentage
            pck_results[f'PCK@{threshold}px'] = pck
            print(f"  PCK@{threshold}px: {pck:.2f}%")
        
        return pck_results
    else:
        return {}

def compute_2d_auc(gt_poses_pixel, mp_poses_pixel, max_threshold=100.0, num_thresholds=51):
    """Compute 2D AUC metric"""
    min_frames = min(len(gt_poses_pixel), len(mp_poses_pixel))
    gt_poses = gt_poses_pixel[:min_frames]
    mp_poses = mp_poses_pixel[:min_frames]
    
    all_joint_errors = []
    valid_frame_count = 0
    
    for frame_idx in range(min_frames):
        gt_frame = gt_poses[frame_idx]
        mp_frame = mp_poses[frame_idx]
        
        gt_valid = not np.all(gt_frame == 0)
        mp_valid = not np.all(mp_frame == 0)
        
        if gt_valid and mp_valid:
            joint_diffs = np.linalg.norm(gt_frame - mp_frame, axis=1)
            all_joint_errors.extend(joint_diffs)
            valid_frame_count += 1
    
    if len(all_joint_errors) > 0:
        all_joint_errors = np.array(all_joint_errors)
        thresholds = np.linspace(0, max_threshold, num_thresholds)
        
        pck_values = []
        for threshold in thresholds:
            pck = np.mean(all_joint_errors < threshold)
            pck_values.append(pck)
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(pck_values, thresholds) / max_threshold
        
        print(f"\n2D AUC Result:")
        print(f"  AUC (0-{max_threshold}px): {auc:.4f}")
        return auc
    else:
        return 0.0

def load_video_frames_for_visualization(sequence_name, num_frames=50):
    """Load video frames for visualization"""
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    
    if not os.path.exists(video_path):
        print(f"Video frames not found at: {video_path}")
        return None
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    
    image_files.sort()
    
    frames = []
    print(f"Loading {min(num_frames, len(image_files))} frames for visualization...")
    
    for i, img_path in enumerate(image_files[:num_frames]):
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
    
    print(f"✓ Loaded {len(frames)} frames")
    return frames


def create_comparison_visualization(gt_poses_2d, mp_poses_2d, seq_name, args):
    """Create side-by-side comparison visualization"""
    min_frames = min(len(gt_poses_2d), len(mp_poses_2d), args.num_frames)
    
    if min_frames == 0:
        print("No frames to visualize!")
        return None, None, 0
    
    # Limit frames
    gt_poses = gt_poses_2d[:min_frames]
    mp_poses = mp_poses_2d[:min_frames]
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'2D Pose Comparison: Ground Truth vs MediaPipe - {seq_name} (Root-Relative)', fontsize=16)
    
    # Calculate dynamic limits based on the data (excluding root which is at 0,0)
    all_gt_non_root = gt_poses[:, :, :].reshape(-1, 2)
    all_mp_non_root = mp_poses[:, :, :].reshape(-1, 2)
    
    # Remove zero points and root positions for limit calculation
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
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.5, 0.5
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # Set limits and labels
        for ax in [ax1, ax2]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # Flip Y axis for image coordinates
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (root-relative, normalized)')
            ax.set_ylabel('Y (root-relative, normalized)')
            
            # Add origin markers for root joint
            ax.scatter(0, 0, c='green', s=100, marker='*', alpha=1.0, 
                      edgecolors='darkgreen', linewidth=2, zorder=10)
            ax.text(0.02, 0.02, 'Root(14)', fontsize=10, ha='left', va='bottom', 
                   color='green', weight='bold')
        
        # Plot Ground Truth
        ax1.set_title(f'Ground Truth\nFrame {frame_idx+1}/{min_frames}', fontsize=14)
        gt_frame = gt_poses[frame_idx]
        
        # Check if frame has valid data
        gt_valid = not np.all(gt_frame == 0)
        if gt_valid:
            # Draw skeleton connections
            for connection in connections_2d:
                joint1, joint2 = connection
                if joint1 < len(gt_frame) and joint2 < len(gt_frame):
                    x1, y1 = gt_frame[joint1]
                    x2, y2 = gt_frame[joint2]
                    # Don't skip connections just because one joint is at origin
                    ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
            
            # Draw joints (excluding root which is already marked)
            for joint_idx, (x, y) in enumerate(gt_frame):
                if joint_idx != 14:  # Skip root joint
                    ax1.scatter(x, y, c='blue', s=60, alpha=0.9, edgecolors='darkblue', linewidth=1)
                    # Changed text color to black for better visibility
                    ax1.text(x+0.02, y+0.02, str(joint_idx), fontsize=10, ha='left', va='bottom', 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        else:
            ax1.text(0, 0.1, 'No GT Data', ha='center', va='center', fontsize=16, color='red')
        
        # Plot MediaPipe
        ax2.set_title(f'MediaPipe Estimation\nFrame {frame_idx+1}/{min_frames}', fontsize=14)
        mp_frame = mp_poses[frame_idx]
        
        # Check if frame has valid data
        mp_valid = not np.all(mp_frame == 0)
        if mp_valid:
            # Draw skeleton connections
            for connection in connections_2d:
                joint1, joint2 = connection
                if joint1 < len(mp_frame) and joint2 < len(mp_frame):
                    x1, y1 = mp_frame[joint1]
                    x2, y2 = mp_frame[joint2]
                    # Don't skip connections just because one joint is at origin
                    ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
            
            # Draw joints (excluding root which is already marked)
            for joint_idx, (x, y) in enumerate(mp_frame):
                if joint_idx != 14:  # Skip root joint
                    ax2.scatter(x, y, c='red', s=60, alpha=0.9, edgecolors='darkred', linewidth=1)
                    # Changed text color to black for better visibility
                    ax2.text(x+0.02, y+0.02, str(joint_idx), fontsize=10, ha='left', va='bottom', 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        else:
            ax2.text(0, 0.1, 'No MediaPipe Data', ha='center', va='center', fontsize=16, color='red')
        
        # Compute and display frame error
        if gt_valid and mp_valid:
            frame_error = np.mean(np.linalg.norm(gt_frame - mp_frame, axis=1))
            error_text = f'Frame Error: {frame_error:.4f} (root-relative)'
        else:
            error_text = 'Frame Error: N/A'
        
        fig.suptitle(f'2D Pose Comparison: Ground Truth vs MediaPipe - {seq_name} (Root-Relative)\n{error_text}', 
                    fontsize=16)
        
        plt.tight_layout()
        return [ax1, ax2]
    
    return update, fig, min_frames



def main():
    parser = argparse.ArgumentParser(description='Calculate 2D metrics for Ground Truth vs MediaPipe poses')
    parser.add_argument('--sequence', type=str, default='TS1', 
                       help='Sequence to analyze (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--num-frames', type=int, default=50,
                       help='Number of frames to visualize')
    parser.add_argument('--save-video', action='store_true',
                       help='Save comparison as GIF')
    parser.add_argument('--output-dir', type=str, default='2d_metrics_output',
                       help='Directory to save outputs')
    args = parser.parse_args()
    
    print("2D Pose Metrics Calculator: Ground Truth vs MediaPipe")
    print("=" * 60)
    print(f"Sequence: {args.sequence}")
    print(f"Frames to analyze: {args.num_frames}")
    
    # Load datasets
    gt_data, mp_data = load_datasets()
    if gt_data is None or mp_data is None:
        return
    
    # Check if sequence exists
    if args.sequence not in gt_data:
        print(f"ERROR: Sequence {args.sequence} not found in ground truth data")
        print(f"Available sequences: {list(gt_data.keys())}")
        return
    
    if args.sequence not in mp_data:
        print(f"ERROR: Sequence {args.sequence} not found in MediaPipe data")
        print(f"Available sequences: {list(mp_data.keys())}")
        return
    
    # Extract 2D poses (they should already be in normalized [-1,1] format)
    gt_poses_2d_norm = gt_data[args.sequence]['data_2d']  # (num_frames, 17, 2)
    mp_poses_2d_norm = mp_data[args.sequence]['data_2d']  # (num_frames, 17, 2)
    
    print(f"\n✓ Loaded sequence {args.sequence}")
    print(f"GT frames: {len(gt_poses_2d_norm)}, MP frames: {len(mp_poses_2d_norm)}")
    
    # Verify normalized coordinate ranges
    print(f"\nNormalized coordinate ranges:")
    print(f"GT: X[{np.min(gt_poses_2d_norm[:, :, 0]):.3f}, {np.max(gt_poses_2d_norm[:, :, 0]):.3f}], "
          f"Y[{np.min(gt_poses_2d_norm[:, :, 1]):.3f}, {np.max(gt_poses_2d_norm[:, :, 1]):.3f}]")
    print(f"MP: X[{np.min(mp_poses_2d_norm[:, :, 0]):.3f}, {np.max(mp_poses_2d_norm[:, :, 0]):.3f}], "
          f"Y[{np.min(mp_poses_2d_norm[:, :, 1]):.3f}, {np.max(mp_poses_2d_norm[:, :, 1]):.3f}]")
    
    # Denormalize both datasets back to pixel coordinates
    print(f"\nDenormalizing poses to pixel coordinates...")
    gt_poses_pixel = denormalize_2d_poses(gt_poses_2d_norm, args.sequence)
    mp_poses_pixel = denormalize_2d_poses(mp_poses_2d_norm, args.sequence)
    
    # Calculate metrics in pixel coordinates
    print(f"\nCalculating 2D metrics in pixel coordinates...")
    
    # MPJPE
    mpjpe_metrics = compute_2d_mpjpe(gt_poses_pixel, mp_poses_pixel)
    if mpjpe_metrics is None:
        print("Failed to compute MPJPE!")
        return
    
    # PCK at various thresholds
    pck_results = compute_2d_pck(gt_poses_pixel, mp_poses_pixel, thresholds=[5, 10, 20, 30, 50])
    
    # AUC
    auc_result = compute_2d_auc(gt_poses_pixel, mp_poses_pixel, max_threshold=100.0)
    
    # Combine all metrics
    metrics = {
        **mpjpe_metrics,
        'pck_results': pck_results,
        'auc_2d': auc_result
    }
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f'{args.sequence}_2d_metrics.json')
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL 2D METRICS SUMMARY FOR {args.sequence}")
    print(f"{'='*60}")
    print(f"2D MPJPE: {metrics['mpjpe_2d']:.2f} pixels")
    print(f"2D AUC:   {metrics['auc_2d']:.4f}")
    print(f"2D PCK Results:")
    for threshold, pck in pck_results.items():
        print(f"  {threshold}: {pck:.2f}%")
    print(f"Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
    print(f"{'='*60}")
    
    # Create visualization if requested
    if args.save_video:
        print(f"\nCreating visualization...")
        
        # Load video frames for background
        frames = load_video_frames_for_visualization(args.sequence, args.num_frames)
        
        try:
            result = create_comparison_visualization(
                gt_poses_pixel, mp_poses_pixel, frames, args.sequence, args, metrics)
            
            if result[0] is None:
                return
                
            update_func, fig, min_frames = result
            
            print("Creating animation...")
            ani = FuncAnimation(fig, update_func, frames=min_frames, 
                              interval=500, repeat=True, blit=False)
            
            output_path = os.path.join(args.output_dir, f'{args.sequence}_2d_metrics_comparison.gif')
            
            ani.save(output_path, writer='pillow', fps=2, dpi=100)
            print(f"✓ Animation saved to: {output_path}")
            
            # Save static comparison
            update_func(0)
            static_path = os.path.join(args.output_dir, f'{args.sequence}_2d_metrics_comparison_static.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"✓ Static image saved to: {static_path}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()

    