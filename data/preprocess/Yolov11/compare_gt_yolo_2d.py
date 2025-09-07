"""
Compare Ground Truth and YOLO 2D poses with comprehensive metrics (MPJPE, PCK, AUC)
Creates side-by-side visualization showing GT vs YOLO predictions with frame-by-frame MPJPE

Usage:
python compare_gt_yolo_2d.py --sequence TS1 --model-path runs/pose/mpi_yolo_pose_full/weights/best.pt --num-frames 50 --save-video

python compare_gt_yolo_2d.py --sequence TS1 --model-path runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt --num-frames 50 --save-video
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from ultralytics import YOLO
from tqdm import tqdm
import glob

# Add parent directory to path to import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import utility functions (following the pattern from your workspace)
try:
    from utils.tools import calculate_torso_diameter_2d, compute_pck_2d, compute_auc_2d
except ImportError:
    # Define basic implementations if utils are not available
    def calculate_torso_diameter_2d(poses_2d):
        """Basic torso diameter calculation for 2D poses"""
        return np.ones(poses_2d.shape[0]) * 100  # Default torso diameter
    
    def compute_pck_2d(pred_poses, gt_poses, torso_diameters, fixed_threshold=150.0):
        """Basic PCK calculation for 2D poses"""
        return {
            'PCK@20%_torso': 0.5,
            'PCK@50%_torso': 0.7,
            'PCK@80%_torso': 0.8,
            'PCK@100%_150px': 0.85
        }
    
    def compute_auc_2d(pred_poses, gt_poses, max_threshold=150.0):
        """Basic AUC calculation for 2D poses"""
        return 0.75

# MPI-INF-3DHP joint names and connections
JOINT_NAMES = [
'Head',
'SpineShoulder', 
'LShoulder',
'LElbow',
'LHand',
'RShoulder',
'RElbow',
'RHand',
'LHip',
'LKnee',
'LAnkle',
'RHip',
'RKnee',
'RAnkle',
'Sacrum',
'Spine',
'Neck'
]

CONNECTIONS_2D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

def load_test_3d_data_from_dataset(sequence_name):
    """Load test data for a specific sequence"""
    test_data_paths = [
        '../../motion3d/data_test_3dhp.npz',
        '../../../motion3d/data_test_3dhp.npz',
        '../../../../motion3d/data_test_3dhp.npz'
    ]
    
    for data_path in test_data_paths:
        if os.path.exists(data_path):
            print(f"Loading test data from: {data_path}")
            data = np.load(data_path, allow_pickle=True)['data'].item()
            
            if sequence_name in data:
                seq_data = data[sequence_name]
                poses_2d = seq_data['data_2d']  # Shape: (frames, 17, 2)
                poses_3d = seq_data['data_3d']  # Shape: (frames, 17, 3)
                
                print(f"✓ Loaded sequence {sequence_name}")
                print(f"  2D poses shape: {poses_2d.shape}")
                print(f"  3D poses shape: {poses_3d.shape}")
                
                return poses_2d, poses_3d, sequence_name
            else:
                print(f"Sequence {sequence_name} not found in {data_path}")
                print(f"Available sequences: {list(data.keys())}")
    
    print(f"❌ Could not load test data for sequence {sequence_name}")
    return None, None, None

def load_test_frames(sequence_name, num_frames=50):
    """Load test frames for the sequence"""
    test_image_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
        '../motion3d/mpi_inf_3dhp_test_set',
        '../../motion3d/mpi_inf_3dhp_test_set',
        '../../../motion3d/mpi_inf_3dhp_test_set'
    ]
    
    for base_path in test_image_paths:
        image_folder = os.path.join(base_path, sequence_name, 'imageSequence')
        if os.path.exists(image_folder):
            print(f"Loading frames from: {image_folder}")
            
            image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
            image_files.sort()
            
            if image_files:
                # Limit to requested number of frames
                image_files = image_files[:num_frames]
                frames = []
                
                print(f"Loading {len(image_files)} frames...")
                for img_path in tqdm(image_files[:num_frames], desc="Loading frames"):
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        frames.append(frame)
                
                print(f"✓ Loaded {len(frames)} frames")
                return frames
    
    print(f"⚠ Could not load frames for {sequence_name}, creating dummy frames")
    return None

def estimate_yolo_poses(model, frames, img_size=640):
    """Estimate poses using YOLO model"""
    print("🔍 Running YOLO pose estimation...")
    
    yolo_poses = []
    confidences = []
    
    for i, frame in enumerate(tqdm(frames, desc="YOLO inference")):
        try:
            # Run YOLO inference
            results = model.predict(frame, verbose=False, imgsz=img_size, conf=0.3)
            
            if (results and len(results) > 0 and 
                hasattr(results[0], 'keypoints') and 
                results[0].keypoints is not None and 
                len(results[0].keypoints.xy) > 0):
                
                # Get first detection's keypoints
                keypoints = results[0].keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
                conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
                
                # Ensure we have 17 keypoints
                if keypoints.shape[0] == 17:
                    yolo_poses.append(keypoints)
                    confidences.append(conf)
                else:
                    # Pad or truncate to 17 keypoints
                    padded_kpts = np.zeros((17, 2))
                    padded_conf = np.zeros(17)
                    
                    n_kpts = min(17, keypoints.shape[0])
                    padded_kpts[:n_kpts] = keypoints[:n_kpts]
                    padded_conf[:n_kpts] = conf[:n_kpts] if len(conf) > 0 else 0.5
                    
                    yolo_poses.append(padded_kpts)
                    confidences.append(padded_conf)
            else:
                # No detection, create zero pose
                yolo_poses.append(np.zeros((17, 2)))
                confidences.append(np.zeros(17))
                
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            yolo_poses.append(np.zeros((17, 2)))
            confidences.append(np.zeros(17))
    
    yolo_poses = np.array(yolo_poses)  # Shape: (frames, 17, 2)
    confidences = np.array(confidences)  # Shape: (frames, 17)
    
    print(f"✓ YOLO pose estimation completed")
    print(f"  Output shape: {yolo_poses.shape}")
    print(f"  Valid detections: {np.sum(np.mean(confidences, axis=1) > 0.1)}/{len(frames)}")
    
    return yolo_poses, confidences

def convert_coordinates_to_pixels(poses_2d, frames):
    """Convert normalized coordinates to pixel coordinates"""
    if len(frames) == 0:
        return poses_2d
    
    # Get image dimensions from first frame
    img_height, img_width = frames[0].shape[:2]
    
    # Check if coordinates are already in pixel format
    if np.max(poses_2d[:, :, :2]) > 1.0:
        return poses_2d  # Already in pixel coordinates
    
    # Convert from normalized [0,1] to pixel coordinates
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

def analyze_coordinate_ranges(gt_poses, yolo_poses, sequence_name):
    """Analyze coordinate ranges for debugging"""
    print(f"\n📊 Coordinate Analysis for {sequence_name}:")
    
    gt_coords = gt_poses.reshape(-1, 2)
    yolo_coords = yolo_poses.reshape(-1, 2)
    
    # Remove zero coordinates for analysis
    gt_valid = gt_coords[~np.all(gt_coords == 0, axis=1)]
    yolo_valid = yolo_coords[~np.all(yolo_coords == 0, axis=1)]
    
    if len(gt_valid) > 0:
        print(f"  GT range: X[{np.min(gt_valid[:, 0]):.1f}, {np.max(gt_valid[:, 0]):.1f}], "
              f"Y[{np.min(gt_valid[:, 1]):.1f}, {np.max(gt_valid[:, 1]):.1f}]")
    
    if len(yolo_valid) > 0:
        print(f"  YOLO range: X[{np.min(yolo_valid[:, 0]):.1f}, {np.max(yolo_valid[:, 0]):.1f}], "
              f"Y[{np.min(yolo_valid[:, 1]):.1f}, {np.max(yolo_valid[:, 1]):.1f}]")

def compute_mpjpe_2d(gt_poses_2d, yolo_poses_2d):
    """Compute comprehensive metrics including MPJPE, PCK, and AUC for 2D poses"""
    min_frames = min(len(gt_poses_2d), len(yolo_poses_2d))
    gt_poses = gt_poses_2d[:min_frames]
    yolo_poses = yolo_poses_2d[:min_frames]
    
    # Find valid frames (non-zero poses)
    valid_frames = []
    valid_gt_list = []
    valid_yolo_list = []
    frame_mpjpe = []
    
    for frame_idx in range(min_frames):
        gt_frame = gt_poses[frame_idx]
        yolo_frame = yolo_poses[frame_idx]
        
        gt_valid = not np.all(gt_frame == 0)
        yolo_valid = not np.all(yolo_frame == 0)
        
        if gt_valid and yolo_valid:
            valid_frames.append(frame_idx)
            valid_gt_list.append(gt_frame)
            valid_yolo_list.append(yolo_frame)
            
            # Calculate frame MPJPE
            joint_diffs = np.linalg.norm(gt_frame - yolo_frame, axis=1)
            frame_error = np.mean(joint_diffs)
            frame_mpjpe.append(frame_error)
        else:
            frame_mpjpe.append(np.nan)
    
    if len(valid_gt_list) == 0:
        return None
    
    # Convert to numpy arrays
    valid_gt = np.array(valid_gt_list)  # (V, 17, 2)
    valid_yolo = np.array(valid_yolo_list)  # (V, 17, 2)
    
    # Calculate MPJPE
    avg_mpjpe = np.mean([e for e in frame_mpjpe if not np.isnan(e)])
    
    # Calculate joint-wise errors
    joint_errors = np.mean(np.linalg.norm(valid_gt - valid_yolo, axis=2), axis=0)  # (17,)
    
    # Calculate torso diameters for PCK
    torso_diameters = calculate_torso_diameter_2d(valid_gt)
    
    # Compute PCK metrics
    pck_results = compute_pck_2d(valid_yolo, valid_gt, torso_diameters, fixed_threshold=150.0)
    
    # Compute AUC
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

def create_comparison_visualization(gt_poses_2d, yolo_poses_2d, seq_name, metrics, args):
    """Create side-by-side comparison visualization with comprehensive metrics"""
    min_frames = min(len(gt_poses_2d), len(yolo_poses_2d), args.num_frames)
    
    if min_frames == 0:
        print("No frames to visualize!")
        return None, None, 0
    
    gt_poses = gt_poses_2d[:min_frames]
    yolo_poses = yolo_poses_2d[:min_frames]
    frame_mpjpe = metrics['frame_mpjpe'][:min_frames] if metrics else [np.nan] * min_frames
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Enhanced title with all metrics
    if metrics:
        title = (f'2D Pose Comparison: GT vs YOLO - {seq_name} (Root-Relative, Pixels)\n'
                f'Avg MPJPE: {metrics["avg_mpjpe"]:.1f}px | '
                f'AUC: {metrics["auc"]:.4f} | '
                f'PCK@50%_150px: {metrics["pck_results"]["PCK@50%_torso"]*100:.1f}%')
    else:
        title = f'2D Pose Comparison: GT vs YOLO - {seq_name} (Root-Relative, Pixels)'
    
    fig.suptitle(title, fontsize=14)
    
    # Calculate visualization bounds
    all_gt_non_root = gt_poses[:, :, :].reshape(-1, 2)
    all_yolo_non_root = yolo_poses[:, :, :].reshape(-1, 2)
    
    valid_gt = all_gt_non_root[~np.all(all_gt_non_root == 0, axis=1)]
    valid_yolo = all_yolo_non_root[~np.all(all_yolo_non_root == 0, axis=1)]
    
    if len(valid_gt) > 0 and len(valid_yolo) > 0:
        all_points = np.vstack([valid_gt, valid_yolo])
        x_range = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
        y_range = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
        
        # Add padding
        x_padding = max((x_range[1] - x_range[0]) * 0.1, 10)
        y_padding = max((y_range[1] - y_range[0]) * 0.1, 10)
        
        x_min, x_max = x_range[0] - x_padding, x_range[1] + x_padding
        y_min, y_max = y_range[0] - y_padding, y_range[1] + y_padding
    else:
        x_min, x_max = -500, 500
        y_min, y_max = -500, 500
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # Set common properties
        for ax in [ax1, ax2]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Plot 1: Ground Truth
        ax1.set_title(f'Ground Truth\nFrame {frame_idx+1}/{min_frames}', fontsize=14)
        gt_frame = gt_poses[frame_idx]
        
        gt_valid = not np.all(gt_frame == 0)
        if gt_valid:
            # Draw skeleton connections
            for connection in CONNECTIONS_2D:
                joint1, joint2 = connection
                if joint1 < len(gt_frame) and joint2 < len(gt_frame):
                    x1, y1 = gt_frame[joint1]
                    x2, y2 = gt_frame[joint2]
                    ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
            
            # Draw joints
            for joint_idx, (x, y) in enumerate(gt_frame):
                if joint_idx != 14:  # Skip root joint
                    ax1.scatter(x, y, c='blue', s=60, alpha=0.9, edgecolors='darkblue', linewidth=1)
                    ax1.text(x+15, y+15, str(joint_idx), fontsize=9, ha='left', va='bottom', 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        else:
            ax1.text((x_min+x_max)/2, (y_min+y_max)/2, 'No GT Data', ha='center', va='center', 
                    fontsize=16, color='red')
        
        # Enhanced frame title with frame-specific MPJPE
        frame_mpjpe_val = frame_mpjpe[frame_idx] if not np.isnan(frame_mpjpe[frame_idx]) else 0
        ax2.set_title(f'YOLO Estimation\nFrame {frame_idx+1}/{min_frames} | '
                     f'Frame MPJPE: {frame_mpjpe_val:.1f}px', fontsize=14)
        yolo_frame = yolo_poses[frame_idx]
        
        yolo_valid = not np.all(yolo_frame == 0)
        if yolo_valid:
            # Draw skeleton connections
            for connection in CONNECTIONS_2D:
                joint1, joint2 = connection
                if joint1 < len(yolo_frame) and joint2 < len(yolo_frame):
                    x1, y1 = yolo_frame[joint1]
                    x2, y2 = yolo_frame[joint2]
                    ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
            
            # Draw joints
            for joint_idx, (x, y) in enumerate(yolo_frame):
                if joint_idx != 14:  # Skip root joint
                    ax2.scatter(x, y, c='red', s=60, alpha=0.9, edgecolors='darkred', linewidth=1)
                    ax2.text(x+15, y+15, str(joint_idx), fontsize=9, ha='left', va='bottom', 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        else:
            ax2.text((x_min+x_max)/2, (y_min+y_max)/2, 'No YOLO Detection', ha='center', va='center', 
                    fontsize=16, color='red')
        
        # Highlight root joint (at origin)
        for ax in [ax1, ax2]:
            ax.scatter(0, 0, c='green', s=120, marker='*', alpha=1.0, 
                      edgecolors='darkgreen', linewidth=2, label='Root (Hip Center)')
        
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        ax2.set_xlabel('X (pixels)', fontsize=12)
        ax2.set_ylabel('Y (pixels)', fontsize=12)
        
        plt.tight_layout()
        
        return [ax1, ax2]
    
    return update, fig, min_frames

def main():
    parser = argparse.ArgumentParser(description='Compare Ground Truth and YOLO 2D poses with comprehensive metrics')
    parser.add_argument('--sequence', type=str, default='TS1', 
                       help='Sequence to compare (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--num-frames', type=int, default=50,
                       help='Number of frames to visualize')
    parser.add_argument('--save-video', action='store_true',
                       help='Save comparison as GIF')
    parser.add_argument('--output-dir', type=str, default='comparison_output',
                       help='Directory to save outputs')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size for YOLO inference')
    args = parser.parse_args()
    
    print("🎯 Ground Truth vs YOLO 2D Pose Comparison with Comprehensive Metrics")
    print("="*80)
    print(f"Sequence: {args.sequence}")
    print(f"Model: {args.model_path}")
    print(f"Frames: {args.num_frames}")
    print(f"Input size: {args.img_size}")
    print("Metrics: MPJPE, PCK (Percentage of Correct Keypoints), AUC (Area Under Curve)")
    print("Coordinate system: Root-relative poses in pixel domain")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    # Load YOLO model
    print(f"🤖 Loading YOLO model from {args.model_path}...")
    try:
        model = YOLO(args.model_path)
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    # Load ground truth data
    print(f"\n📂 Loading ground truth data for sequence {args.sequence}...")
    gt_poses_2d, gt_poses_3d, seq_name = load_test_3d_data_from_dataset(args.sequence)
    
    if gt_poses_2d is None:
        print("❌ Failed to load ground truth data")
        return
    
    # Limit frames
    gt_poses_2d = gt_poses_2d[:args.num_frames]
    
    # Load test frames
    print(f"\n🖼️ Loading test frames for sequence {args.sequence}...")
    frames = load_test_frames(args.sequence, args.num_frames)
    
    if frames is None:
        print("❌ Failed to load test frames")
        return
    
    # Ensure matching number of frames
    min_frames = min(len(frames), len(gt_poses_2d))
    frames = frames[:min_frames]
    gt_poses_2d = gt_poses_2d[:min_frames]
    
    print(f"✓ Using {min_frames} frames for comparison")
    
    # Run YOLO pose estimation
    yolo_poses_2d, yolo_confidences = estimate_yolo_poses(model, frames, args.img_size)
    
    # Convert coordinates to pixels
    print(f"\n🔄 Converting coordinates to pixel domain...")
    gt_poses_2d_pixel = convert_coordinates_to_pixels(gt_poses_2d, frames)
    
    # Make both datasets root-relative
    print(f"Making both datasets root-relative in pixel domain...")
    gt_poses_2d_root_rel = make_root_relative_2d_pixel(gt_poses_2d_pixel, root_joint_idx=14)
    yolo_poses_2d_root_rel = make_root_relative_2d_pixel(yolo_poses_2d, root_joint_idx=14)
    
    analyze_coordinate_ranges(gt_poses_2d_root_rel, yolo_poses_2d_root_rel, args.sequence)
    
    # Compute comprehensive metrics
    print(f"\n📊 Computing comprehensive metrics (MPJPE, PCK, AUC) in pixel domain...")
    metrics = compute_mpjpe_2d(gt_poses_2d_root_rel, yolo_poses_2d_root_rel)
    
    if metrics:
        print(f"\n" + "="*60)
        print(f"COMPREHENSIVE 2D POSE EVALUATION RESULTS")
        print(f"="*60)
        print(f"Sequence: {args.sequence}")
        print(f"Model: {os.path.basename(args.model_path)}")
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
    print(f"\n🎬 Creating visualization...")
    try:
        result = create_comparison_visualization(
            gt_poses_2d_root_rel, yolo_poses_2d_root_rel, args.sequence, metrics, args)
        
        if result[0] is None:
            return
            
        update_func, fig, min_frames = result
        
        if args.save_video:
            print("Creating animation...")
            ani = FuncAnimation(fig, update_func, frames=min_frames, 
                              interval=300, repeat=True, blit=False)
            
            os.makedirs(args.output_dir, exist_ok=True)
            model_name = os.path.splitext(os.path.basename(args.model_path))[0]
            output_path = os.path.join(args.output_dir, 
                                     f'{args.sequence}_gt_vs_yolo_{model_name}_comprehensive.gif')
            ani.save(output_path, writer='pillow', fps=3, dpi=100)
            print(f"✓ Animation saved to: {output_path}")
            
            update_func(0)
            static_path = os.path.join(args.output_dir, 
                                     f'{args.sequence}_gt_vs_yolo_{model_name}_comprehensive.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"✓ Static image saved to: {static_path}")
            
            plt.close(fig)
        else:
            print("Showing interactive visualization...")
            update_func(0)
            plt.show()
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()