"""
Compare Ground Truth 2D poses with MediaPipe 2D estimations
This visualizes both side-by-side to verify the conversion is correct

Usage:
python compare_gt_mediapipe_2d.py --sequence TS1 --num-frames 10

cd ~/TCPFormerForked/data/preprocess/MediaPipeTest/2d

# Interactive comparison
python compare_gt_mediapipe_2d.py --sequence TS1 --num-frames 20

# Save as GIF with analysis
python compare_gt_mediapipe_2d.py --sequence TS1 --num-frames 50 --save-video --save-analysis

# Compare different sequence
python compare_gt_mediapipe_2d.py --sequence TS2 --num-frames 30 --save-video
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from tqdm import tqdm

# Navigate to project root
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

# MPI-INF-3DHP skeleton connections for 2D visualization
connections_2d = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
    (0, 5), (5, 6), (6, 7),          # Left arm
    (0, 14), (14, 8), (8, 9), (9, 10),  # Right leg
    (14, 11), (11, 12), (12, 13),    # Left leg
    (0, 15), (15, 16), (0, 16)       # Spine and head
]

# Joint names for reference
JOINT_NAMES = [
    'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
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

def analyze_coordinate_ranges(gt_poses_2d, mp_poses_2d, seq_name):
    """Analyze coordinate ranges for both datasets"""
    print(f"\nCoordinate analysis for {seq_name}:")
    print(f"Ground Truth 2D:")
    print(f"  Shape: {gt_poses_2d.shape}")
    print(f"  X range: [{np.min(gt_poses_2d[:, :, 0]):.3f}, {np.max(gt_poses_2d[:, :, 0]):.3f}]")
    print(f"  Y range: [{np.min(gt_poses_2d[:, :, 1]):.3f}, {np.max(gt_poses_2d[:, :, 1]):.3f}]")
    
    print(f"MediaPipe 2D:")
    print(f"  Shape: {mp_poses_2d.shape}")
    print(f"  X range: [{np.min(mp_poses_2d[:, :, 0]):.3f}, {np.max(mp_poses_2d[:, :, 0]):.3f}]")
    print(f"  Y range: [{np.min(mp_poses_2d[:, :, 1]):.3f}, {np.max(mp_poses_2d[:, :, 1]):.3f}]")
    
    # Check for zero poses
    gt_zeros = np.sum(np.all(gt_poses_2d == 0, axis=(1, 2)))
    mp_zeros = np.sum(np.all(mp_poses_2d == 0, axis=(1, 2)))
    print(f"Zero poses: GT={gt_zeros}/{len(gt_poses_2d)}, MP={mp_zeros}/{len(mp_poses_2d)}")

def compute_comparison_metrics(gt_poses_2d, mp_poses_2d):
    """Compute comparison metrics between GT and MediaPipe poses"""
    # Ensure same number of frames
    min_frames = min(len(gt_poses_2d), len(mp_poses_2d))
    gt_poses = gt_poses_2d[:min_frames]
    mp_poses = mp_poses_2d[:min_frames]
    
    # Compute frame-wise differences
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
            # Compute L2 distance per joint
            joint_diffs = np.linalg.norm(gt_frame - mp_frame, axis=1)
            frame_error = np.mean(joint_diffs)
            frame_errors.append(frame_error)
            joint_errors += joint_diffs
            valid_frame_count += 1
    
    if valid_frame_count > 0:
        avg_frame_error = np.mean(frame_errors)
        joint_errors /= valid_frame_count
        
        print(f"\nComparison Metrics:")
        print(f"Valid frames: {valid_frame_count}/{min_frames}")
        print(f"Average frame error: {avg_frame_error:.4f}")
        print(f"Joint errors (top 5):")
        
        # Sort joints by error
        joint_error_pairs = [(i, joint_errors[i], JOINT_NAMES[i]) for i in range(17)]
        joint_error_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (joint_idx, error, name) in enumerate(joint_error_pairs[:5]):
            print(f"  {name} (joint {joint_idx}): {error:.4f}")
        
        return {
            'avg_frame_error': avg_frame_error,
            'joint_errors': joint_errors,
            'valid_frames': valid_frame_count,
            'total_frames': min_frames
        }
    else:
        print("No valid frames found for comparison!")
        return None

def create_comparison_visualization(gt_poses_2d, mp_poses_2d, seq_name, args):
    """Create side-by-side comparison visualization"""
    min_frames = min(len(gt_poses_2d), len(mp_poses_2d), args.num_frames)
    
    if min_frames == 0:
        print("No frames to visualize!")
        return
    
    # Limit frames
    gt_poses = gt_poses_2d[:min_frames]
    mp_poses = mp_poses_2d[:min_frames]
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'2D Pose Comparison: Ground Truth vs MediaPipe - {seq_name}', fontsize=16)
    
    # Calculate plot limits based on both datasets
    all_gt = gt_poses.reshape(-1, 2)
    all_mp = mp_poses.reshape(-1, 2)
    
    # Remove zero points for limit calculation
    valid_gt = all_gt[~np.all(all_gt == 0, axis=1)]
    valid_mp = all_mp[~np.all(all_mp == 0, axis=1)]
    
    if len(valid_gt) > 0 and len(valid_mp) > 0:
        all_points = np.vstack([valid_gt, valid_mp])
        x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        
        # Add padding
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
    else:
        x_min, x_max = -1.2, 1.2
        y_min, y_max = -1.2, 1.2
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # Set limits and labels
        for ax in [ax1, ax2]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # Flip Y axis for image coordinates
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
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
                    if not (x1 == 0 and y1 == 0) and not (x2 == 0 and y2 == 0):
                        ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
            
            # Draw joints
            for joint_idx, (x, y) in enumerate(gt_frame):
                if not (x == 0 and y == 0):
                    ax1.scatter(x, y, c='blue', s=60, alpha=0.9, edgecolors='darkblue', linewidth=1)
                    ax1.text(x, y, str(joint_idx), fontsize=8, ha='center', va='bottom', color='white', weight='bold')
        else:
            ax1.text(0, 0, 'No GT Data', ha='center', va='center', fontsize=16, color='red')
        
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
                    if not (x1 == 0 and y1 == 0) and not (x2 == 0 and y2 == 0):
                        ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
            
            # Draw joints
            for joint_idx, (x, y) in enumerate(mp_frame):
                if not (x == 0 and y == 0):
                    ax2.scatter(x, y, c='red', s=60, alpha=0.9, edgecolors='darkred', linewidth=1)
                    ax2.text(x, y, str(joint_idx), fontsize=8, ha='center', va='bottom', color='white', weight='bold')
        else:
            ax2.text(0, 0, 'No MediaPipe Data', ha='center', va='center', fontsize=16, color='red')
        
        # Compute and display frame error
        if gt_valid and mp_valid:
            frame_error = np.mean(np.linalg.norm(gt_frame - mp_frame, axis=1))
            error_text = f'Frame Error: {frame_error:.4f}'
        else:
            error_text = 'Frame Error: N/A'
        
        fig.suptitle(f'2D Pose Comparison: Ground Truth vs MediaPipe - {seq_name}\n{error_text}', fontsize=16)
        
        plt.tight_layout()
        return [ax1, ax2]
    
    return update, fig, min_frames

def save_comparison_analysis(gt_data, mp_data, seq_name, metrics, output_dir):
    """Save detailed comparison analysis to JSON"""
    analysis = {
        'sequence_name': seq_name,
        'gt_shape': gt_data[seq_name]['data_2d'].shape,
        'mp_shape': mp_data[seq_name]['data_2d'].shape,
        'gt_coordinate_range': {
            'x_min': float(np.min(gt_data[seq_name]['data_2d'][:, :, 0])),
            'x_max': float(np.max(gt_data[seq_name]['data_2d'][:, :, 0])),
            'y_min': float(np.min(gt_data[seq_name]['data_2d'][:, :, 1])),
            'y_max': float(np.max(gt_data[seq_name]['data_2d'][:, :, 1]))
        },
        'mp_coordinate_range': {
            'x_min': float(np.min(mp_data[seq_name]['data_2d'][:, :, 0])),
            'x_max': float(np.max(mp_data[seq_name]['data_2d'][:, :, 0])),
            'y_min': float(np.min(mp_data[seq_name]['data_2d'][:, :, 1])),
            'y_max': float(np.max(mp_data[seq_name]['data_2d'][:, :, 1]))
        },
        'comparison_metrics': metrics,
        'joint_names': JOINT_NAMES
    }
    
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, f'{seq_name}_comparison_analysis.json')
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✓ Analysis saved to: {analysis_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare Ground Truth and MediaPipe 2D poses')
    parser.add_argument('--sequence', type=str, default='TS1', 
                       help='Sequence to compare (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--num-frames', type=int, default=50,
                       help='Number of frames to visualize')
    parser.add_argument('--save-video', action='store_true',
                       help='Save comparison as GIF')
    parser.add_argument('--save-analysis', action='store_true',
                       help='Save detailed analysis to JSON')
    parser.add_argument('--output-dir', type=str, default='comparison_output',
                       help='Directory to save outputs')
    args = parser.parse_args()
    
    print("Ground Truth vs MediaPipe 2D Pose Comparison")
    print("=" * 60)
    print(f"Sequence: {args.sequence}")
    print(f"Frames to visualize: {args.num_frames}")
    
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
    
    # Extract 2D poses
    gt_poses_2d = gt_data[args.sequence]['data_2d']  # (num_frames, 17, 2)
    mp_poses_2d = mp_data[args.sequence]['data_2d']  # (num_frames, 17, 2)
    
    print(f"\n✓ Loaded sequence {args.sequence}")
    print(f"GT frames: {len(gt_poses_2d)}, MP frames: {len(mp_poses_2d)}")
    
    # Analyze coordinate ranges
    analyze_coordinate_ranges(gt_poses_2d, mp_poses_2d, args.sequence)
    
    # Compute comparison metrics
    metrics = compute_comparison_metrics(gt_poses_2d, mp_poses_2d)
    
    # Save analysis if requested
    if args.save_analysis and metrics:
        save_comparison_analysis(gt_data, mp_data, args.sequence, metrics, args.output_dir)
    
    # Create visualization
    print(f"\nCreating visualization...")
    try:
        update_func, fig, min_frames = create_comparison_visualization(
            gt_poses_2d, mp_poses_2d, args.sequence, args)
        
        if args.save_video:
            print("Creating animation...")
            ani = FuncAnimation(fig, update_func, frames=min_frames, 
                              interval=300, repeat=True, blit=False)
            
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f'{args.sequence}_gt_vs_mediapipe_2d.gif')
            
            ani.save(output_path, writer='pillow', fps=3, dpi=100)
            print(f"✓ Animation saved to: {output_path}")
            
            # Save static comparison
            update_func(0)
            static_path = os.path.join(args.output_dir, f'{args.sequence}_gt_vs_mediapipe_2d_static.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"✓ Static image saved to: {static_path}")
        else:
            # Interactive visualization
            print("Showing interactive visualization...")
            update_func(0)
            plt.show()
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()