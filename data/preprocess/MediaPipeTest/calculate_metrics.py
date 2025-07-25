"""
Calculate metrics for MediaPipe 3D pose estimation on MPI-INF-3DHP test set
Usage: python calculate_metrics.py --sequence-name TS1 --save-video
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mediapipe as mp
from dataclasses import dataclass
import torch
import glob
import gc
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from data.reader.motion_dataset import MPI3DHP, Fusion
from data.const import H36M_TO_MPI
from utils.utils_3dhp import mpjpe_cal

# Ground truth skeleton connections
connections = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

class AccumLoss:
    """Accumulator for loss values - same as train_3dhp.py"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MediaPipe3DPoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_to_mpi_mapping = {
            11: 5, 12: 2, 13: 6, 14: 3, 15: 7, 16: 4,
            23: 11, 24: 8, 25: 12, 26: 9, 27: 13, 28: 10
        }
        self.missing_joints_estimation = {
            14: [11, 8], 1: [5, 2], 15: [14, 1]
        }

    def estimate_3d_pose_from_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        pose_3d = np.zeros((17, 3), dtype=np.float32)
        visibility = np.zeros(17, dtype=np.float32)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            for mp_idx, gt_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    pose_3d[gt_idx] = [landmark.x * 1000, landmark.y * 1000, landmark.z * 1000]
                    visibility[gt_idx] = landmark.visibility

            if len(landmarks) > 10:
                left_eyebrow_inner = landmarks[2]
                right_eyebrow_inner = landmarks[5]
                mouth_left = landmarks[9]
                mouth_right = landmarks[10]

                pose_3d[0] = [
                    (left_eyebrow_inner.x + right_eyebrow_inner.x) / 2.0 * 1000,
                    (left_eyebrow_inner.y + right_eyebrow_inner.y) / 2.0 * 1000,
                    (left_eyebrow_inner.z + right_eyebrow_inner.z) / 2.0 * 1000
                ]
                visibility[0] = (left_eyebrow_inner.visibility + right_eyebrow_inner.visibility) / 2.0

                pose_3d[16] = [
                    (mouth_left.x + mouth_right.x) / 2.0 * 1000,
                    (mouth_left.y + mouth_right.y) / 2.0 * 1000,
                    (mouth_left.z + mouth_right.z) / 2.0 * 1000
                ]
                visibility[16] = (mouth_left.visibility + mouth_right.visibility) / 2.0

            for missing_joint, source_joints in self.missing_joints_estimation.items():
                valid_sources = [j for j in source_joints if visibility[j] > 0.1]
                if valid_sources:
                    pose_3d[missing_joint] = np.mean([pose_3d[j] for j in valid_sources], axis=0)
                    visibility[missing_joint] = np.mean([visibility[j] for j in valid_sources])

            if visibility[14] > 0.1:
                pose_3d -= pose_3d[14]

            cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
            pose_3d = pose_3d @ cam2real

        return pose_3d, visibility

    def close(self):
        self.pose.close()

def load_mpi_test_frames(sequence_name, num_frames=1000):
    """Load video frames from MPI-INF-3DHP test set"""
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    
    if not os.path.exists(video_path):
        print(f"Video frames not found at: {video_path}")
        return None, []
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    
    image_files.sort()
    frames = []
    frame_indices = []
    
    print(f"Loading {min(num_frames, len(image_files))} frames...")
    
    for i, img_path in enumerate(image_files[:num_frames]):
        if i % 100 == 0:
            print(f"  Loaded {i}/{min(num_frames, len(image_files))} frames")
        
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
            try:
                frame_idx = int(os.path.basename(img_path).split('_')[-1].split('.')[0])
                frame_indices.append(frame_idx)
            except:
                frame_indices.append(len(frame_indices))
    
    print(f"✓ Loaded {len(frames)} frames")
    return frames, frame_indices

def load_test_3d_data_from_dataset(args):
    """Load test data from MPI-INF-3DHP dataset - same as train_3dhp.py"""
    @dataclass
    class DatasetArgs:
        data_root: str
        n_frames: int
        stride: int
        flip: bool
        test_augmentation: bool
        data_augmentation: bool
        reverse_augmentation: bool
        out_all: int
        test_batch_size: int

    dataset_args = DatasetArgs(
        data_root='../motion3d/',
        n_frames=27,
        stride=9,
        flip=False,
        test_augmentation=False,
        data_augmentation=False,
        reverse_augmentation=False,
        out_all=1,
        test_batch_size=1
    )
    
    dataset = Fusion(dataset_args, train=False)
    target_seq_name = args.sequence_name or ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6'][args.sequence_number % 6]
    
    print(f"Looking for sequence: {target_seq_name}")
    print(f"Dataset has {len(dataset)} samples")
    
    # Collect all samples for the target sequence
    sequence_samples = []
    sequence_info = []
    
    for i in range(len(dataset)):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            current_seq_name = seq[0] if isinstance(seq, (list, tuple)) else str(seq)
            
            if current_seq_name != target_seq_name:
                continue
            
            if i % 50 == 0:
                print(f"Processing sample {i}: {current_seq_name}")
            
            if isinstance(gt_3D, torch.Tensor):
                gt_3D = gt_3D.clone()
            else:
                gt_3D = torch.tensor(gt_3D)
                
            gt_3D = gt_3D.view(1, -1, 17, 3)
            gt_3D[:, :, 14] = 0
            
            # Extract center frame (same as train_3dhp.py evaluation)
            center_frame_idx = gt_3D.shape[1] // 2
            center_frame = gt_3D[0, center_frame_idx]
            center_frame = center_frame - center_frame[14:15, :]
            
            if hasattr(center_frame, 'cpu'):
                center_frame = center_frame.cpu().numpy()
            
            sequence_samples.append(center_frame)
            sequence_info.append({
                'sample_idx': i,
                'center_frame_idx': center_frame_idx,
                'seq_name': current_seq_name
            })
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if not sequence_samples:
        return None, None, None
    
    # Stack all center frames
    sequence_3d = np.stack(sequence_samples, axis=0).transpose(1, 0, 2)  # (17, T, 3)
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    print(f"Loaded {sequence_3d.shape[1]} frames for sequence: {target_seq_name}")
    print(f"GT sequence shape: {sequence_3d.shape}")
    
    return sequence_3d, target_seq_name, sequence_info

def calculate_torso_diameter(poses_3d):
    """Calculate torso diameter for PCK metric - same as train_3dhp.py"""
    # Torso joints: left shoulder (5), right shoulder (2), left hip (11), right hip (8)
    left_shoulder = poses_3d[:, 5, :]   # (N, 3)
    right_shoulder = poses_3d[:, 2, :]  # (N, 3)
    left_hip = poses_3d[:, 11, :]       # (N, 3)
    right_hip = poses_3d[:, 8, :]       # (N, 3)
    
    # Calculate distances
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1)  # (N,)
    hip_dist = np.linalg.norm(left_hip - right_hip, axis=1)                 # (N,)
    
    # Torso diameter is the average of shoulder and hip distances
    torso_diameter = (shoulder_dist + hip_dist) / 2.0
    return torso_diameter

def compute_pck(pred, gt, torso_diameters, fixed_threshold=150.0):
    """Compute PCK metrics - same as train_3dhp.py"""
    # pred, gt shape: (N, 17, 3)
    # torso_diameters shape: (N,)
    
    joint_errors = np.linalg.norm(pred - gt, axis=2)  # (N, 17)
    
    pck_results = {}
    
    # Torso-based thresholds
    for percentage in [70, 80, 90]:
        threshold = torso_diameters[:, None] * (percentage / 100.0)  # (N, 1)
        correct = joint_errors < threshold  # (N, 17)
        pck = np.mean(correct, axis=0)  # (17,) - per joint
        pck_results[f'PCK@{percentage}%_torso'] = np.mean(pck)  # Overall average
    
    # Fixed threshold (150mm)
    for percentage in [70, 80, 90]:
        threshold = fixed_threshold * (percentage / 100.0)
        correct = joint_errors < threshold
        pck = np.mean(correct, axis=0)
        pck_results[f'PCK@{percentage}%_150mm'] = np.mean(pck)
    
    return pck_results

def compute_auc(pred, gt, max_threshold=150.0, num_thresholds=31):
    """Compute AUC metric - same as train_3dhp.py"""
    # pred, gt shape: (N, 17, 3)
    
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

def process_frame_batch(estimator, frames, batch_size=10):
    """Process frames in batches"""
    pred_poses_3d = []
    visibilities = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        for frame in batch_frames:
            pose_3d, visibility = estimator.estimate_3d_pose_from_image(frame)
            pred_poses_3d.append(pose_3d)
            visibilities.append(visibility)
        gc.collect()
        
        if len(frames) > 100:  # Only show progress for large datasets
            print(f"Processed batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}")
    
    return np.stack(pred_poses_3d, axis=1), np.stack(visibilities, axis=1)

def evaluate_mediapipe(pred_poses_3d, gt_poses_3d, visibilities):
    """Evaluate MediaPipe predictions using same metrics as train_3dhp.py"""
    
    # pred_poses_3d: (17, T, 3)
    # gt_poses_3d: (17, T, 3)
    # visibilities: (17, T)
    
    num_frames = pred_poses_3d.shape[1]
    
    # Convert to format expected by metrics: (T, 17, 3)
    pred_poses = pred_poses_3d.transpose(1, 0, 2)  # (T, 17, 3)
    gt_poses = gt_poses_3d.transpose(1, 0, 2)      # (T, 17, 3)
    
    # Filter frames with sufficient valid joints
    valid_frames = []
    valid_pred = []
    valid_gt = []
    
    for t in range(num_frames):
        valid_joints = visibilities[:, t] > 0.1
        if np.sum(valid_joints) >= 8:  # At least 8 valid joints
            valid_frames.append(t)
            valid_pred.append(pred_poses[t])
            valid_gt.append(gt_poses[t])
    
    if len(valid_frames) == 0:
        print("No frames with sufficient valid joints found!")
        return None
    
    valid_pred = np.stack(valid_pred, axis=0)  # (V, 17, 3)
    valid_gt = np.stack(valid_gt, axis=0)      # (V, 17, 3)
    
    print(f"Evaluating on {len(valid_frames)}/{num_frames} frames with sufficient valid joints")
    
    # Initialize metrics
    error_sum = AccumLoss()
    pck_results = {
        'PCK@90%_torso': 0.0, 'PCK@80%_torso': 0.0, 'PCK@70%_torso': 0.0,
        'PCK@90%_150mm': 0.0, 'PCK@80%_150mm': 0.0, 'PCK@70%_150mm': 0.0
    }
    auc_sum = 0.0
    valid_samples = len(valid_frames)
    
    # Convert to torch tensors for MPJPE calculation
    pred_tensor = torch.from_numpy(valid_pred).float()  # (V, 17, 3)
    gt_tensor = torch.from_numpy(valid_gt).float()      # (V, 17, 3)
    
    # Calculate MPJPE
    mpjpe_error = mpjpe_cal(pred_tensor, gt_tensor).item()
    error_sum.update(mpjpe_error * valid_samples, valid_samples)
    
    # Calculate torso diameters
    torso_diameters = calculate_torso_diameter(valid_gt)
    
    # Compute PCK
    batch_pck = compute_pck(valid_pred, valid_gt, torso_diameters, fixed_threshold=150.0)
    for key in pck_results:
        pck_results[key] = batch_pck[key]
    
    # Compute AUC
    auc = compute_auc(valid_pred, valid_gt)
    auc_sum = auc
    
    # Print results (same format as train_3dhp.py)
    print(f'\n{"="*50}')
    print(f'MediaPipe 3D Pose Estimation Results')
    print(f'{"="*50}')
    print(f'Protocol #1 Error (MPJPE): {error_sum.avg:.2f} mm')
    for key, value in pck_results.items():
        print(f'{key}: {value*100:.2f}%')
    print(f'AUC: {auc_sum:.4f}')
    print(f'Valid frames: {valid_samples}/{num_frames}')
    print(f'{"="*50}')
    
    return {
        'mpjpe': error_sum.avg,
        'pck_results': pck_results,
        'auc': auc_sum,
        'valid_frames': valid_samples,
        'total_frames': num_frames
    }

def create_visualization(pred_poses_3d, gt_poses_3d, visibilities, seq_name, frame_indices, metrics):
    """Create visualization with metrics"""
    
    num_frames = min(pred_poses_3d.shape[1], 50)  # Limit to 50 frames for visualization
    
    # Calculate bounds
    all_gt = gt_poses_3d[:, :num_frames, :].reshape(-1, 3)
    all_pred = pred_poses_3d[:, :num_frames, :].reshape(-1, 3)
    valid_gt = all_gt[~np.isnan(all_gt).any(axis=1) & ~np.isinf(all_gt).any(axis=1)]
    valid_pred = all_pred[~np.isnan(all_pred).any(axis=1) & ~np.isinf(all_pred).any(axis=1)]
    
    if len(valid_gt) > 0 and len(valid_pred) > 0:
        all_poses = np.vstack([valid_gt, valid_pred])
    elif len(valid_gt) > 0:
        all_poses = valid_gt
    else:
        all_poses = valid_pred
        
    min_value = np.min(all_poses, axis=0)
    max_value = np.max(all_poses, axis=0)
    padding = (max_value - min_value) * 0.1
    min_value -= padding
    max_value += padding
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        for ax in [ax1, ax2]:
            ax.set_xlim3d([min_value[0], max_value[0]])
            ax.set_ylim3d([min_value[1], max_value[1]])
            ax.set_zlim3d([min_value[2], max_value[2]])
            ax.set_xlabel('X (mm)', fontsize=12)
            ax.set_ylabel('Y (mm)', fontsize=12)
            ax.set_zlabel('Z (mm)', fontsize=12)
        
        # Plot ground truth
        ax1.set_title(f'Ground Truth\n(Frame {frame_indices[frame_idx]})', fontsize=14, pad=20)
        x_gt = gt_poses_3d[:, frame_idx, 0]
        y_gt = gt_poses_3d[:, frame_idx, 1]
        z_gt = gt_poses_3d[:, frame_idx, 2]
        
        # Draw GT skeleton
        for connection in connections:
            start = gt_poses_3d[connection[0], frame_idx, :]
            end = gt_poses_3d[connection[1], frame_idx, :]
            ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                     'b-', linewidth=3, alpha=0.8)
        
        ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=100, alpha=0.9, edgecolors='darkblue', linewidth=2)
        ax1.scatter(x_gt[14], y_gt[14], z_gt[14], c='green', s=200, marker='*', 
                   alpha=1.0, edgecolors='darkgreen', linewidth=3)
        
        # Plot MediaPipe prediction
        ax2.set_title(f'MediaPipe Prediction\n(Frame {frame_indices[frame_idx]})', fontsize=14, pad=20)
        valid = visibilities[:, frame_idx] > 0.1
        
        if np.any(valid):
            # Draw MediaPipe skeleton
            for connection in connections:
                if valid[connection[0]] and valid[connection[1]]:
                    start = pred_poses_3d[connection[0], frame_idx, :]
                    end = pred_poses_3d[connection[1], frame_idx, :]
                    ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                             'r-', linewidth=3, alpha=0.8)
            
            # Draw joints
            for joint_idx in range(17):
                x_pos = pred_poses_3d[joint_idx, frame_idx, 0]
                y_pos = pred_poses_3d[joint_idx, frame_idx, 1]
                z_pos = pred_poses_3d[joint_idx, frame_idx, 2]
                
                if valid[joint_idx]:
                    ax2.scatter(x_pos, y_pos, z_pos, c='red', s=100, alpha=0.9, 
                               edgecolors='darkred', linewidth=2)
                else:
                    ax2.scatter(x_pos, y_pos, z_pos, c='gray', s=50, alpha=0.5, 
                               edgecolors='black', linewidth=1)
            
            if valid[14]:
                ax2.scatter(pred_poses_3d[14, frame_idx, 0], pred_poses_3d[14, frame_idx, 1], 
                           pred_poses_3d[14, frame_idx, 2], c='green', s=200, marker='*', 
                           alpha=1.0, edgecolors='darkgreen', linewidth=3)
        
        # Update title with metrics
        fig.suptitle(f'MediaPipe vs Ground Truth - {seq_name}\n'
                    f'Frame {frame_idx+1}/{num_frames} | '
                    f'MPJPE: {metrics["mpjpe"]:.1f}mm | '
                    f'AUC: {metrics["auc"]:.3f} | '
                    f'Valid Joints: {np.sum(valid)}/17', 
                    fontsize=16, y=0.95)
        
        return ax1, ax2
    
    return fig, update, num_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, help='Sequence name (TS1, TS2, etc.)')
    parser.add_argument('--save-video', action='store_true', help='Save animation as GIF')
    parser.add_argument('--batch-size', type=int, default=20, help='Process frames in batches')
    parser.add_argument('--max-frames', type=int, default=500, help='Maximum frames to process')
    args = parser.parse_args()
    
    print("MediaPipe 3D Pose Estimation Metrics Calculator")
    print("=" * 50)
    
    estimator = MediaPipe3DPoseEstimator()
    
    try:
        # Load GT data
        print("Loading ground truth data...")
        gt_poses_3d, seq_name, sequence_info = load_test_3d_data_from_dataset(args)
        if gt_poses_3d is None:
            print("Failed to load ground truth data.")
            return
        
        # Load video frames
        print(f"Loading video frames for sequence {seq_name}...")
        all_video_frames, all_video_frame_indices = load_mpi_test_frames(seq_name, args.max_frames * 2)
        if not all_video_frames:
            print("No video frames loaded.")
            return
        
        # Sample frames to match GT data
        num_frames = min(gt_poses_3d.shape[1], args.max_frames, len(all_video_frames) // 9)
        
        stride = 9
        center_offset = 13
        sampled_video_frames = []
        sampled_frame_indices = []
        
        for i in range(num_frames):
            video_idx = i * stride + center_offset
            if video_idx < len(all_video_frames):
                sampled_video_frames.append(all_video_frames[video_idx])
                sampled_frame_indices.append(all_video_frame_indices[video_idx] if video_idx < len(all_video_frame_indices) else video_idx)
        
        final_gt_poses = gt_poses_3d[:, :len(sampled_video_frames), :]
        final_frame_indices = sampled_frame_indices
        
        print(f"Processing {len(sampled_video_frames)} frames...")
        
        # Process MediaPipe poses
        print("Computing MediaPipe 3D poses...")
        pred_poses_3d, visibilities = process_frame_batch(estimator, sampled_video_frames, batch_size=args.batch_size)
        
        print("✓ MediaPipe processing completed")
        
        # Evaluate metrics
        print("Calculating metrics...")
        metrics = evaluate_mediapipe(pred_poses_3d, final_gt_poses, visibilities)
        
        if metrics is None:
            print("Failed to calculate metrics.")
            return
        
        # Create visualization if requested
        if args.save_video:
            print("Creating visualization...")
            fig, update_func, vis_frames = create_visualization(
                pred_poses_3d, final_gt_poses, visibilities, seq_name, final_frame_indices, metrics
            )
            
            # Create animation
            ani = FuncAnimation(fig, update_func, frames=range(vis_frames), 
                              interval=200, repeat=True, blit=False)
            
            output_path = f'../mediapipe_metrics_{seq_name.lower()}_mpjpe_{metrics["mpjpe"]:.1f}mm.gif'
            print(f"Saving animation to: {output_path}")
            ani.save(output_path, writer='pillow', fps=5, dpi=100)
            print(f"✓ Animation saved to: {output_path}")
            
            # Save static image
            update_func(0)
            plt.tight_layout()
            static_path = output_path.replace('.gif', '.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"✓ Static image saved to: {static_path}")
            
            plt.close(fig)
        
        # Print final summary
        print(f'\n{"="*60}')
        print(f'FINAL RESULTS FOR SEQUENCE {seq_name}')
        print(f'{"="*60}')
        print(f'MPJPE: {metrics["mpjpe"]:.2f} mm')
        print(f'AUC: {metrics["auc"]:.4f}')
        print(f'PCK@150mm (70%): {metrics["pck_results"]["PCK@70%_150mm"]*100:.2f}%')
        print(f'PCK@150mm (80%): {metrics["pck_results"]["PCK@80%_150mm"]*100:.2f}%')
        print(f'PCK@150mm (90%): {metrics["pck_results"]["PCK@90%_150mm"]*100:.2f}%')
        print(f'Valid frames: {metrics["valid_frames"]}/{metrics["total_frames"]}')
        print(f'{"="*60}')
        
    finally:
        estimator.close()
        gc.collect()

if __name__ == '__main__':
    main()