'''
# Single sequence with visualization
python compare_gt_mediapipe_3d.py --sequence TS1 --save-video

# Process all sequences (comprehensive metrics)
python compare_gt_mediapipe_3d.py --all

# Interactive visualization with 100 frames
python compare_gt_mediapipe_3d.py --sequence TS2 --num-frames 100
'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from tqdm import tqdm
import torch
from mpl_toolkits.mplot3d import Axes3D

# Navigate to project root
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

# Import utility functions from your workspace
from utils.utils_3dhp import AccumLoss, calculate_torso_diameter, compute_pck, compute_auc, mpjpe_cal

# MPI-INF-3DHP skeleton connections for 3D visualization
connections_3d = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

# Joint names for reference
JOINT_NAMES = [
    'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

def load_datasets():
    """Load both ground truth and MediaPipe 3D datasets"""
    gt_path = project_root + '/data/preprocess/MediaPipeTest/data1/motion3d/data_test_3dhp.npz'
    mp_path = project_root + '/data/preprocess/MediaPipeTest/data1/motion3d/data_test_3dhp_mediapipe_3d.npz'

    print("Loading datasets...")
    
    if not os.path.exists(gt_path):
        print(f"ERROR: Ground truth dataset not found: {gt_path}")
        return None, None
    
    gt_data = np.load(gt_path, allow_pickle=True)['data'].item()
    print(f"✓ Ground truth loaded: {list(gt_data.keys())}")
    
    if not os.path.exists(mp_path):
        print(f"ERROR: MediaPipe 3D dataset not found: {mp_path}")
        return None, None
    
    mp_data = np.load(mp_path, allow_pickle=True)['data'].item()
    print(f"✓ MediaPipe 3D loaded: {list(mp_data.keys())}")
    
    return gt_data, mp_data

def apply_camera_transformation(poses_3d):
    """
    Apply camera transformation to align MediaPipe coordinates with MPI-INF-3DHP
    
    MediaPipe world coordinates: X=right, Y=down, Z=forward (camera view)
    MPI-INF-3DHP coordinates: X=right, Y=up, Z=backward (world view)
    
    Transformation matrix converts from camera to world coordinates
    """
    # Camera to world transformation matrix
    # This flips Y and Z axes to match MPI-INF-3DHP coordinate system
    cam2world = np.array([
        [1,  0,  0],  # X stays the same (right)
        [0, -1,  0],  # Y flips (down -> up)
        [0,  0, -1]   # Z flips (forward -> backward)
    ], dtype=np.float32)
    
    # Apply transformation to all poses
    transformed_poses = poses_3d @ cam2world.T
    
    return transformed_poses

def apply_upright_correction(poses_3d):
    """
    Apply additional rotation to make human figures stand upright
    
    This function rotates the poses around the X-axis by 90 degrees 
    to correct the orientation where heads are pointing down
    """
    # Rotation matrix: 90 degrees around X-axis (counter-clockwise)
    # This will rotate Y->Z and Z->-Y, making figures stand upright
    rotation_x_90 = np.array([
        [1,  0,  0],   # X stays the same
        [0,  0, 1],   # Y becomes Z 
        [0,  -1,  0]    # Z becomes -Y 
    ], dtype=np.float32)
    
    # Apply rotation to all poses
    upright_poses = poses_3d @ rotation_x_90.T
    
    return upright_poses

def apply_complete_camera_correction(poses_3d, is_mediapipe=False):
    """
    Apply complete camera correction for proper human pose orientation
    
    Args:
        poses_3d: Input 3D poses
        is_mediapipe: Whether these are MediaPipe poses (needs initial transformation)
    
    Returns:
        Corrected 3D poses with humans standing upright
    """
    corrected_poses = poses_3d.copy()
    
    if is_mediapipe:
        # Step 1: Apply MediaPipe to MPI-INF-3DHP coordinate transformation
        corrected_poses = apply_camera_transformation(corrected_poses)
        print("  Applied MediaPipe -> MPI-INF-3DHP transformation")
    
    # Step 2: Apply upright correction for both datasets
    corrected_poses = apply_upright_correction(corrected_poses)
    print(f"  Applied upright correction (90° X-axis rotation)")
    
    return corrected_poses

def make_root_relative_3d(poses_3d, root_joint_idx=14):
    """
    Make poses root-relative by subtracting root joint position from all joints
    """
    root_relative_poses = poses_3d.copy()
    
    for frame_idx in range(poses_3d.shape[0]):
        frame = poses_3d[frame_idx]
        root_pos = frame[root_joint_idx]
        
        if not np.all(frame == 0):
            root_relative_poses[frame_idx] = frame - root_pos[np.newaxis, :]
            root_relative_poses[frame_idx, root_joint_idx] = [0.0, 0.0, 0.0]
    
    return root_relative_poses

def compute_mpjpe_3d(gt_poses_3d, mp_poses_3d):
    """
    Compute comprehensive metrics including MPJPE, PCK, and AUC for 3D poses
    """
    min_frames = min(len(gt_poses_3d), len(mp_poses_3d))
    gt_poses = gt_poses_3d[:min_frames]
    mp_poses = mp_poses_3d[:min_frames]
    
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
    
    # Convert to numpy arrays first
    valid_gt = np.array(valid_gt_list)  # (V, 17, 3)
    valid_mp = np.array(valid_mp_list)  # (V, 17, 3)
    
    # Convert to PyTorch tensors for the utility functions
    gt_tensor = torch.from_numpy(valid_gt).float()
    mp_tensor = torch.from_numpy(valid_mp).float()
    
    # Calculate MPJPE using the same function as train_3dhp.py
    avg_mpjpe = mpjpe_cal(mp_tensor, gt_tensor).item()
    
    # Calculate joint-wise errors
    joint_errors = np.mean(np.linalg.norm(valid_gt - valid_mp, axis=2), axis=0)  # (17,)
    
    # Calculate torso diameters for PCK
    torso_diameters = calculate_torso_diameter(gt_tensor)
    
    # Compute PCK metrics
    pck_results = compute_pck(mp_tensor, gt_tensor, torso_diameters, fixed_threshold=150.0)
    
    # Compute AUC
    auc = compute_auc(mp_tensor, gt_tensor)
    
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
    # Extract 3D poses
    gt_poses_3d = gt_data[seq_name]['data_3d']
    mp_poses_3d = mp_data[seq_name]['data_3d']
    
    # Apply complete camera correction to both datasets
    print(f"Applying camera corrections for {seq_name}...")
    print("Ground Truth corrections:")
    gt_poses_3d_corrected = apply_complete_camera_correction(gt_poses_3d, is_mediapipe=False)
    
    print("MediaPipe corrections:")
    mp_poses_3d_corrected = apply_complete_camera_correction(mp_poses_3d, is_mediapipe=True)
    
    # Then make both datasets root-relative
    gt_poses_3d_root_rel = make_root_relative_3d(gt_poses_3d_corrected, root_joint_idx=14)
    mp_poses_3d_root_rel = make_root_relative_3d(mp_poses_3d_corrected, root_joint_idx=14)
    
    # Compute comprehensive metrics
    metrics = compute_mpjpe_3d(gt_poses_3d_root_rel, mp_poses_3d_root_rel)
    
    return metrics

def process_all_sequences(gt_data, mp_data):
    """Process all sequences and compute overall average metrics including PCK and AUC"""
    print("\n" + "="*70)
    print("PROCESSING ALL SEQUENCES - COMPUTING COMPREHENSIVE 3D METRICS")
    print("="*70)
    
    all_sequences = sorted(gt_data.keys())
    sequence_results = {}
    
    # Overall statistics
    total_valid_frames = 0
    total_frame_errors = []
    total_joint_errors = np.zeros(17)
    
    # For PCK and AUC averaging
    total_pck_sums = {
        'PCK@90%_torso': 0.0, 'PCK@80%_torso': 0.0, 'PCK@70%_torso': 0.0,
        'PCK@90%_150mm': 0.0, 'PCK@80%_150mm': 0.0, 'PCK@70%_150mm': 0.0
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
                
                print(f"  ✓ {seq_name}: {metrics['avg_mpjpe']:.1f} mm, "
                      f"AUC: {metrics['auc']:.4f}, "
                      f"PCK@80%_150mm: {metrics['pck_results']['PCK@80%_150mm']*100:.1f}% "
                      f"({metrics['valid_frames']}/{metrics['total_frames']} frames)")
            else:
                print(f"  ✗ {seq_name}: No valid data")
        else:
            print(f"  ✗ {seq_name}: Not found in MediaPipe 3D data")
    
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
        print("OVERALL 3D RESULTS ACROSS ALL SEQUENCES")
        print("="*70)
        print(f"Total valid frames: {total_valid_frames:,}")
        print(f"Total sequences processed: {valid_sequences}")
        print(f"Overall Average MPJPE: {overall_avg_mpjpe:.1f} mm")
        print(f"Overall AUC: {overall_auc:.4f}")
        
        print(f"\nOverall PCK Results:")
        for key, value in overall_pck_results.items():
            print(f"  {key}: {value*100:.2f}%")
        
        print(f"\nOverall Joint Errors (top 5):")
        joint_error_pairs = [(i, overall_joint_errors[i], JOINT_NAMES[i]) for i in range(17)]
        joint_error_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (joint_idx, error, name) in enumerate(joint_error_pairs[:5]):
            print(f"  {name} (joint {joint_idx}): {error:.1f} mm")
        
        print(f"\nPer-sequence breakdown:")
        for seq_name in sorted(sequence_results.keys()):
            metrics = sequence_results[seq_name]
            print(f"  {seq_name}: MPJPE={metrics['avg_mpjpe']:.1f}mm, "
                  f"AUC={metrics['auc']:.3f}, "
                  f"PCK@80%_150mm={metrics['pck_results']['PCK@80%_150mm']*100:.1f}% "
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

def analyze_coordinate_ranges(gt_poses_3d, mp_poses_3d, seq_name):
    """Analyze coordinate ranges for both 3D datasets"""
    print(f"\nCoordinate analysis for {seq_name}:")
    print(f"Ground Truth 3D (after complete correction):")
    print(f"  Shape: {gt_poses_3d.shape}")
    
    # Filter out zero poses for analysis
    gt_nonzero = gt_poses_3d[~np.all(gt_poses_3d == 0, axis=(1, 2))]
    if len(gt_nonzero) > 0:
        print(f"  X range: [{np.min(gt_nonzero[:, :, 0]):.1f}, {np.max(gt_nonzero[:, :, 0]):.1f}] mm")
        print(f"  Y range: [{np.min(gt_nonzero[:, :, 1]):.1f}, {np.max(gt_nonzero[:, :, 1]):.1f}] mm")
        print(f"  Z range: [{np.min(gt_nonzero[:, :, 2]):.1f}, {np.max(gt_nonzero[:, :, 2]):.1f}] mm")
    
    print(f"MediaPipe 3D (after complete correction):")
    print(f"  Shape: {mp_poses_3d.shape}")
    
    # Filter out zero poses for analysis
    mp_nonzero = mp_poses_3d[~np.all(mp_poses_3d == 0, axis=(1, 2))]
    if len(mp_nonzero) > 0:
        print(f"  X range: [{np.min(mp_nonzero[:, :, 0]):.1f}, {np.max(mp_nonzero[:, :, 0]):.1f}] mm")
        print(f"  Y range: [{np.min(mp_nonzero[:, :, 1]):.1f}, {np.max(mp_nonzero[:, :, 1]):.1f}] mm")
        print(f"  Z range: [{np.min(mp_nonzero[:, :, 2]):.1f}, {np.max(mp_nonzero[:, :, 2]):.1f}] mm")
    
    gt_zeros = np.sum(np.all(gt_poses_3d == 0, axis=(1, 2)))
    mp_zeros = np.sum(np.all(mp_poses_3d == 0, axis=(1, 2)))
    print(f"Zero poses: GT={gt_zeros}/{len(gt_poses_3d)}, MP={mp_zeros}/{len(mp_poses_3d)}")

def create_comparison_visualization(gt_poses_3d, mp_poses_3d, seq_name, metrics, args):
    """Create side-by-side comparison visualization with comprehensive metrics for 3D poses"""
    min_frames = min(len(gt_poses_3d), len(mp_poses_3d), args.num_frames)
    
    if min_frames == 0:
        print("No frames to visualize!")
        return None, None, 0
    
    gt_poses = gt_poses_3d[:min_frames]
    mp_poses = mp_poses_3d[:min_frames]
    frame_mpjpe = metrics['frame_mpjpe'][:min_frames] if metrics else [np.nan] * min_frames
    
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Enhanced title with all metrics
    if metrics:
        title = (f'3D Pose Comparison: GT vs MediaPipe - {seq_name} (Upright Corrected)\n'
                f'Avg MPJPE: {metrics["avg_mpjpe"]:.1f}mm | '
                f'AUC: {metrics["auc"]:.4f} | '
                f'PCK@80%_150mm: {metrics["pck_results"]["PCK@80%_150mm"]*100:.1f}%')
    else:
        title = f'3D Pose Comparison: GT vs MediaPipe - {seq_name} (Upright Corrected)'
    
    fig.suptitle(title, fontsize=16)
    
    # Calculate visualization bounds
    all_gt_non_root = gt_poses[:, :, :].reshape(-1, 3)
    all_mp_non_root = mp_poses[:, :, :].reshape(-1, 3)
    
    valid_gt = all_gt_non_root[~np.all(all_gt_non_root == 0, axis=1)]
    valid_mp = all_mp_non_root[~np.all(all_mp_non_root == 0, axis=1)]
    
    if len(valid_gt) > 0 and len(valid_mp) > 0:
        all_points = np.vstack([valid_gt, valid_mp])
        x_range = np.max(np.abs(all_points[:, 0]))
        y_range = np.max(np.abs(all_points[:, 1]))
        z_range = np.max(np.abs(all_points[:, 2]))
        max_range = max(x_range, y_range, z_range) * 1.2
        coord_min, coord_max = -max_range, max_range
    else:
        coord_min, coord_max = -500, 500
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        for ax in [ax1, ax2]:
            ax.set_xlim3d([coord_min, coord_max])
            ax.set_ylim3d([coord_min, coord_max])
            ax.set_zlim3d([coord_min, coord_max])
            ax.set_xlabel('X (mm, right)', fontsize=12)
            ax.set_ylabel('Y (mm, forward)', fontsize=12)
            ax.set_zlabel('Z (mm, up)', fontsize=12)
            
            # Mark root joint at origin with better visibility
            ax.scatter(0, 0, 0, c='green', s=200, marker='*', alpha=1.0, 
                      edgecolors='darkgreen', linewidth=3, label='Root (Hip Center)')
            
            # Add coordinate system reference with corrected labels
            ax.plot([0, 100], [0, 0], [0, 0], 'r-', linewidth=2, alpha=0.7)  # X-axis (right)
            ax.plot([0, 0], [0, 100], [0, 0], 'g-', linewidth=2, alpha=0.7)  # Y-axis (forward)
            ax.plot([0, 0], [0, 0], [0, 100], 'b-', linewidth=2, alpha=0.7)  # Z-axis (up)
            ax.text(100, 0, 0, 'X(R)', fontsize=10, color='red')
            ax.text(0, 100, 0, 'Y(F)', fontsize=10, color='green')
            ax.text(0, 0, 100, 'Z(U)', fontsize=10, color='blue')
        
        ax1.set_title(f'Ground Truth (Upright)\nFrame {frame_idx+1}/{min_frames}', fontsize=16, pad=20)
        gt_frame = gt_poses[frame_idx]
        
        gt_valid = not np.all(gt_frame == 0)
        if gt_valid:
            # Draw skeleton connections
            for connection in connections_3d:
                joint1, joint2 = connection
                if joint1 < len(gt_frame) and joint2 < len(gt_frame):
                    x1, y1, z1 = gt_frame[joint1]
                    x2, y2, z2 = gt_frame[joint2]
                    ax1.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=3, alpha=0.8)
            
            # Draw joint points with better visibility
            for joint_idx, (x, y, z) in enumerate(gt_frame):
                if joint_idx != 14:  # Skip root joint (already marked)
                    ax1.scatter(x, y, z, c='blue', s=80, alpha=0.9, 
                               edgecolors='darkblue', linewidth=2)
                    ax1.text(x+30, y+30, z+30, str(joint_idx), fontsize=11, 
                            color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.7))
        else:
            ax1.text(0, 0, 100, 'No GT Data', ha='center', va='center', 
                    fontsize=18, color='red', weight='bold')
        
        # Enhanced frame title with frame-specific MPJPE
        frame_mpjpe_val = frame_mpjpe[frame_idx] if not np.isnan(frame_mpjpe[frame_idx]) else 0
        ax2.set_title(f'MediaPipe (Upright Corrected)\nFrame {frame_idx+1}/{min_frames} | '
                     f'Frame MPJPE: {frame_mpjpe_val:.1f}mm', fontsize=16, pad=20)
        mp_frame = mp_poses[frame_idx]
        
        mp_valid = not np.all(mp_frame == 0)
        if mp_valid:
            # Draw skeleton connections
            for connection in connections_3d:
                joint1, joint2 = connection
                if joint1 < len(mp_frame) and joint2 < len(mp_frame):
                    x1, y1, z1 = mp_frame[joint1]
                    x2, y2, z2 = mp_frame[joint2]
                    ax2.plot([x1, x2], [y1, y2], [z1, z2], 'r-', linewidth=3, alpha=0.8)
            
            # Draw joint points with better visibility
            for joint_idx, (x, y, z) in enumerate(mp_frame):
                if joint_idx != 14:  # Skip root joint (already marked)
                    ax2.scatter(x, y, z, c='red', s=80, alpha=0.9, 
                               edgecolors='darkred', linewidth=2)
                    ax2.text(x+30, y+30, z+30, str(joint_idx), fontsize=11, 
                            color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7))
        else:
            ax2.text(0, 0, 100, 'No MediaPipe Data', ha='center', va='center', 
                    fontsize=18, color='red', weight='bold')
        
        # Set better viewing angles for upright human pose
        for ax in [ax1, ax2]:
            ax.view_init(elev=15, azim=45)  # Slightly elevated view for upright figures
        
        plt.tight_layout()
        return [ax1, ax2]
    
    return update, fig, min_frames

def main():
    parser = argparse.ArgumentParser(description='Compare Ground Truth and MediaPipe 3D poses with comprehensive metrics (MPJPE, PCK, AUC)')
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
    
    print("Ground Truth vs MediaPipe 3D Pose Comparison with Complete Camera Correction")
    print("Metrics: MPJPE, PCK (Percentage of Correct Keypoints), AUC (Area Under Curve)")
    print("Camera Transformation: Both datasets corrected for upright human pose visualization")
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
    
    # Single sequence processing
    print(f"Sequence: {args.sequence}")
    print(f"Frames to visualize: {args.num_frames}")
    
    # Check if sequence exists
    if args.sequence not in gt_data:
        print(f"ERROR: Sequence {args.sequence} not found in ground truth data")
        print(f"Available sequences: {list(gt_data.keys())}")
        return
    
    if args.sequence not in mp_data:
        print(f"ERROR: Sequence {args.sequence} not found in MediaPipe 3D data")
        print(f"Available sequences: {list(mp_data.keys())}")
        return
    
    # Extract poses
    gt_poses_3d = gt_data[args.sequence]['data_3d']
    mp_poses_3d = mp_data[args.sequence]['data_3d']
    
    print(f"\n✓ Loaded sequence {args.sequence}")
    print(f"GT frames: {len(gt_poses_3d)}, MP frames: {len(mp_poses_3d)}")
    
    # Apply complete camera correction to both datasets
    print(f"\nApplying complete camera corrections...")
    print("Ground Truth corrections:")
    gt_poses_3d_corrected = apply_complete_camera_correction(gt_poses_3d, is_mediapipe=False)
    
    print("MediaPipe corrections:")
    mp_poses_3d_corrected = apply_complete_camera_correction(mp_poses_3d, is_mediapipe=True)
    
    # Make both datasets root-relative
    print(f"\nMaking both datasets root-relative...")
    gt_poses_3d_root_rel = make_root_relative_3d(gt_poses_3d_corrected, root_joint_idx=14)
    mp_poses_3d_root_rel = make_root_relative_3d(mp_poses_3d_corrected, root_joint_idx=14)
    
    analyze_coordinate_ranges(gt_poses_3d_root_rel, mp_poses_3d_root_rel, args.sequence)
    
    print(f"\nComputing comprehensive metrics (MPJPE, PCK, AUC)...")
    metrics = compute_mpjpe_3d(gt_poses_3d_root_rel, mp_poses_3d_root_rel)
    
    if metrics:
        print(f"\n" + "="*60)
        print(f"COMPREHENSIVE 3D POSE EVALUATION RESULTS")
        print(f"="*60)
        print(f"Sequence: {args.sequence}")
        print(f"Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
        print(f"\nMPJPE (Mean Per Joint Position Error):")
        print(f"  Average MPJPE: {metrics['avg_mpjpe']:.2f} mm")
        
        print(f"\nPCK (Percentage of Correct Keypoints):")
        for key, value in metrics['pck_results'].items():
            print(f"  {key}: {value*100:.2f}%")
        
        print(f"\nAUC (Area Under Curve):")
        print(f"  AUC: {metrics['auc']:.4f}")
        
        print(f"\nJoint-wise errors (top 5 worst):")
        joint_error_pairs = [(i, metrics['joint_errors'][i], JOINT_NAMES[i]) for i in range(17)]
        joint_error_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (joint_idx, error, name) in enumerate(joint_error_pairs[:5]):
            print(f"  {name} (joint {joint_idx}): {error:.2f} mm")
        
        print(f"="*60)
    
    # Create visualization
    print(f"\nCreating 3D visualization with complete camera correction...")
    try:
        result = create_comparison_visualization(
            gt_poses_3d_root_rel, mp_poses_3d_root_rel, args.sequence, metrics, args)
        
        if result[0] is None:
            return
            
        update_func, fig, min_frames = result
        
        if args.save_video:
            print("Creating 3D animation...")
            ani = FuncAnimation(fig, update_func, frames=min_frames, 
                              interval=400, repeat=True, blit=False)
            
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, 
                                     f'{args.sequence}_gt_vs_mediapipe_3d_upright.gif')
            
            ani.save(output_path, writer='pillow', fps=2.5, dpi=120)
            print(f"✓ Animation saved to: {output_path}")
            
            update_func(0)
            static_path = os.path.join(args.output_dir, 
                                     f'{args.sequence}_gt_vs_mediapipe_3d_upright.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"✓ Static image saved to: {static_path}")
            
            plt.close(fig)
        else:
            print("Showing interactive 3D visualization...")
            update_func(0)
            plt.show()
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()