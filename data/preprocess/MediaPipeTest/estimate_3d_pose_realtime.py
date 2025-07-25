'''
cd preprocess

python3 MediaPipeTest/estimate_3d_pose_realtime.py --sequence-name TS1 --save-video --num-frames 20
'''


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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from data.reader.motion_dataset import MPI3DHP, Fusion
from data.const import H36M_TO_MPI

# Ground truth skeleton connections (based on GT joint order)
connections = [
    (0, 16),    # head top -> head
    (16, 1),    # head -> neck
    (1, 2),     # neck -> right arm (shoulder)
    (2, 3),     # right arm -> right forearm
    (3, 4),     # right forearm -> right hand
    (1, 5),     # neck -> left arm (shoulder)
    (5, 6),     # left arm -> left forearm
    (6, 7),     # left forearm -> left hand
    (1, 15),    # neck -> spine
    (15, 14),   # spine -> hip
    (14, 8),    # hip -> right up leg (hip)
    (8, 9),     # right up leg -> right leg
    (9, 10),    # right leg -> right foot
    (14, 11),   # hip -> left up leg (hip)
    (11, 12),   # left up leg -> left leg
    (12, 13),   # left leg -> left foot
]

class MediaPipe3DPoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe to GT joint mapping (aligned with provided GT order)
        self.mp_to_mpi_mapping = {
            # 0: 0,    # nose -> head top (REMOVED - will be calculated)
            # 16: 16,  # head will be calculated from mouth landmarks
            11: 5,   # left_shoulder -> left arm
            12: 2,   # right_shoulder -> right arm
            13: 6,   # left_elbow -> left forearm
            14: 3,   # right_elbow -> right forearm
            15: 7,   # left_wrist -> left hand
            16: 4,   # right_wrist -> right hand
            23: 11,  # left_hip -> left up leg
            24: 8,   # right_hip -> right up leg
            25: 12,  # left_knee -> left leg
            26: 9,   # right_knee -> right leg
            27: 13,  # left_ankle -> left foot
            28: 10,  # right_ankle -> right foot
        }
        
        # Estimation for missing joints
        self.missing_joints_estimation = {
            14: [11, 8],    # hip: average of left up leg (hip) and right up leg (hip)
            1: [5, 2],      # neck: average of left arm (shoulder) and right arm (shoulder)
            15: [14, 1],    # spine: average of hip and neck
        }

    def estimate_3d_pose_from_image(self, image):
        """Estimate 3D pose from a single image using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        pose_3d = np.zeros((17, 3))  # x, y, z
        
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # Map MediaPipe landmarks to GT joints
            visibility = np.zeros(17)
            for mp_idx, gt_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    pose_3d[gt_idx] = [
                        landmark.x * 1000,  # Convert meters to mm
                        landmark.y * 1000,
                        landmark.z * 1000
                    ]
                    visibility[gt_idx] = landmark.visibility
            
            # Calculate head landmarks using eyebrows and mouth
            if len(landmarks) > 10:  # Ensure we have enough landmarks
                # MediaPipe face landmarks indices
                left_eyebrow_inner = landmarks[2]   # MediaPipe left eyebrow inner
                right_eyebrow_inner = landmarks[5]  # MediaPipe right eyebrow inner
                mouth_left = landmarks[9]   # MediaPipe mouth left
                mouth_right = landmarks[10] # MediaPipe mouth right
                
                # Calculate head top (joint 0) as midpoint between eyebrows and move it up
                head_top_x = (left_eyebrow_inner.x + right_eyebrow_inner.x) / 2.0
                head_top_y = (left_eyebrow_inner.y + right_eyebrow_inner.y) / 2.0
                head_top_z = (left_eyebrow_inner.z + right_eyebrow_inner.z) / 2.0
                
                pose_3d[0] = [
                    head_top_x * 1000,  # Convert to mm
                    head_top_y * 1000,
                    head_top_z * 1000
                ]
                
                # Visibility for head top is average of eyebrow visibilities
                visibility[0] = (left_eyebrow_inner.visibility + right_eyebrow_inner.visibility) / 2.0
                
                # Calculate head (joint 16) as midpoint between mouth landmarks
                head_x = (mouth_left.x + mouth_right.x) / 2.0
                head_y = (mouth_left.y + mouth_right.y) / 2.0
                head_z = (mouth_left.z + mouth_right.z) / 2.0
                
                pose_3d[16] = [
                    head_x * 1000,  # Convert to mm
                    head_y * 1000,
                    head_z * 1000
                ]
                
                # Visibility for head is average of mouth visibilities
                visibility[16] = (mouth_left.visibility + mouth_right.visibility) / 2.0
                
                print(f"Head Top (joint 0) calculated from eyebrows: ({head_top_x:.3f}, {head_top_y:.3f}, {head_top_z:.3f})")
                print(f"Head (joint 16) calculated from mouth: ({head_x:.3f}, {head_y:.3f}, {head_z:.3f})")
            
            # Estimate other missing joints
            for missing_joint, source_joints in self.missing_joints_estimation.items():
                valid_sources = [j for j in source_joints if visibility[j] > 0.1]
                if valid_sources:
                    pose_3d[missing_joint] = np.mean([pose_3d[j] for j in valid_sources], axis=0)
                    visibility[missing_joint] = np.mean([visibility[j] for j in valid_sources])
            
            # Make root-relative (GT joint 14: hip)
            if visibility[14] > 0.1:
                root_pos = pose_3d[14].copy()
                pose_3d -= root_pos
            
            # Apply camera transformation to match MPI-INF-3DHP
            cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
            pose_3d = pose_3d @ cam2real
            
            return pose_3d, visibility
        return pose_3d, np.zeros(17)

    def close(self):
        self.pose.close()

def load_mpi_test_frames(sequence_name, num_frames=50):
    """Load video frames from MPI-INF-3DHP test set"""
    mpi_roots = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp',
        '../motion3d/MPI-INF-3DHP',
        '../motion3d'
    ]
    
    possible_paths = []
    for root in mpi_roots:
        possible_paths.extend([
            f'{root}/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence',
            f'{root}/mpi_inf_3dhp_test_set/{sequence_name}/imageFrames',
            f'{root}/test/{sequence_name}/imageSequence',
        ])
    
    video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            break
    
    if video_path is None:
        print("Video frames not found.")
        return None, []
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    
    image_files.sort()
    frames = []
    frame_indices = []
    
    for img_path in image_files[:num_frames]:
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
            try:
                # Extract frame number from filename
                frame_idx = int(os.path.basename(img_path).split('_')[-1].split('.')[0])
                frame_indices.append(frame_idx)
            except:
                frame_indices.append(len(frame_indices))
    
    return frames, frame_indices

def load_test_3d_data_from_dataset_multiple_samples(args):
    """Load multiple samples from the same sequence to get multiple frames - FIXED"""
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

    # Use same parameters as train_3dhp.py
    dataset_args = DatasetArgs(
        data_root='../motion3d/',
        n_frames=27,  # Same as your model
        stride=9,     # Same as your model
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
    
    # Collect multiple samples from the same sequence
    sequence_samples = []
    sample_info = []  # Store starting frame info for each sample
    
    for i in range(len(dataset)):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            
            # Extract sequence name
            current_seq_name = seq[0] if isinstance(seq, (list, tuple)) else str(seq)
            if current_seq_name != target_seq_name:
                continue
            
            print(f"Found matching sequence: {current_seq_name} (sample {i})")
            
            # Process exactly like train_3dhp.py evaluation
            if isinstance(gt_3D, torch.Tensor):
                gt_3D = gt_3D.clone()
            else:
                gt_3D = torch.tensor(gt_3D)
                
            gt_3D = gt_3D.view(1, -1, 17, 3)  # (1, T, 17, 3)
            gt_3D[:, :, 14] = 0  # Set root joint (hip) to 0
            
            # Extract center frame from this sample (same as evaluation)
            center_frame_idx = gt_3D.shape[1] // 2  # This is frame 13 (0-indexed) for 27 frames
            center_frame = gt_3D[0, center_frame_idx]  # (17, 3)
            
            # Make root-relative (same as evaluation)
            center_frame = center_frame - center_frame[14:15, :]
            
            # Convert to numpy
            if hasattr(center_frame, 'cpu'):
                center_frame = center_frame.cpu().numpy()
            
            sequence_samples.append(center_frame)
            
            # Store the center frame index information
            # Each sample starts at: sample_index * stride + center_offset
            sample_start_frame = i * dataset_args.stride  # This is just a guess for relative positioning
            center_frame_absolute = sample_start_frame + center_frame_idx
            sample_info.append(center_frame_absolute)
            
            # Stop when we have enough frames
            if len(sequence_samples) >= args.num_frames:
                break
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if not sequence_samples:
        return None, None, None
    
    # Stack all center frames: (T, 17, 3) -> (17, T, 3)
    sequence_3d = np.stack(sequence_samples, axis=0).transpose(1, 0, 2)
    
    # Apply camera transformation
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    print(f"Loaded {sequence_3d.shape[1]} center frames from {len(sequence_samples)} samples for sequence: {target_seq_name}")
    print(f"GT sequence shape: {sequence_3d.shape}")
    print(f"Center frame is index {27//2} = {27//2} from each 27-frame sample")
    
    return sequence_3d, target_seq_name, sample_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, help='Sequence name (TS1, TS2, etc.)')
    parser.add_argument('--num-frames', type=int, default=20, help='Number of frames to process')
    parser.add_argument('--save-video', action='store_true', help='Save animation as GIF')
    parser.add_argument('--frame-start', type=int, default=0, help='Starting frame for comparison')
    args = parser.parse_args()
    
    # Define GT joint names for clarity
    GT_JOINT_NAMES = {
        0: "Head Top", 1: "Neck", 2: "Right Arm", 3: "Right Forearm", 4: "Right Hand",
        5: "Left Arm", 6: "Left Forearm", 7: "Left Hand", 8: "Right Up Leg", 9: "Right Leg",
        10: "Right Foot", 11: "Left Up Leg", 12: "Left Leg", 13: "Left Foot", 14: "Hip",
        15: "Spine", 16: "Head"
    }
    
    print("Ground Truth Joint Mappings:")
    for idx, name in GT_JOINT_NAMES.items():
        print(f"Joint {idx}: {name}")
    
    print("\n⚠️  Updated: Head Top (joint 0) now calculated as midpoint between left and right ears")
    
    estimator = MediaPipe3DPoseEstimator()
    
    try:
        # Load ground truth data - now multiple samples to get multiple frames
        gt_poses_3d, seq_name, sample_info = load_test_3d_data_from_dataset_multiple_samples(args)
        if gt_poses_3d is None:
            print("Failed to load ground truth data.")
            return
        
        # Load video frames
        all_video_frames, all_video_frame_indices = load_mpi_test_frames(seq_name, args.num_frames * 20)
        if not all_video_frames:
            print("No video frames loaded. Exiting.")
            return
        
        print(f"Loaded {len(all_video_frames)} video frames")
        
        # **FIXED**: Sample video frames to match the center frames from GT samples
        # Each GT sample uses stride=9 and takes center frame (index 13)
        # So video frame pattern should be: 0*9+13, 1*9+13, 2*9+13, etc.
        stride = 9
        center_offset = 13  # Center frame index for 27-frame samples
        
        sampled_video_frames = []
        sampled_frame_indices = []
        
        for i in range(gt_poses_3d.shape[1]):
            # Calculate the video frame index that corresponds to this GT center frame
            video_idx = i * stride + center_offset
            if video_idx < len(all_video_frames):
                sampled_video_frames.append(all_video_frames[video_idx])
                sampled_frame_indices.append(all_video_frame_indices[video_idx] if video_idx < len(all_video_frame_indices) else video_idx)
        
        # Use the minimum between available frames
        num_frames = min(len(sampled_video_frames), gt_poses_3d.shape[1], args.num_frames)
        
        final_video_frames = sampled_video_frames[:num_frames]
        final_gt_poses = gt_poses_3d[:, :num_frames, :]
        final_frame_indices = sampled_frame_indices[:num_frames]
        
        print(f"Using {num_frames} synchronized frames")
        print(f"Video frame pattern: stride={stride}, center_offset={center_offset}")
        print(f"Video frame indices: {final_frame_indices[:10]}...")
        print(f"GT shape: {final_gt_poses.shape}")
        
        if num_frames == 0:
            print("No frames available. Exiting.")
            return
        
        # Pre-compute ALL MediaPipe poses to avoid processing delays
        print("Pre-computing MediaPipe 3D poses for all frames...")
        pred_poses_3d = []
        visibilities = []
        
        for i, frame in enumerate(final_video_frames):
            pose_3d, visibility = estimator.estimate_3d_pose_from_image(frame)
            pred_poses_3d.append(pose_3d)
            visibilities.append(visibility)
            
            if (i + 1) % 5 == 0:
                print(f"Pre-computed {i + 1}/{num_frames} frames")
        
        pred_poses_3d = np.stack(pred_poses_3d, axis=1)  # (17, T, 3)
        visibilities = np.stack(visibilities, axis=1)  # (17, T)
        
        print("✓ All MediaPipe poses pre-computed - no processing delays in animation")
        
        # Calculate MPJPE
        valid_joints = visibilities > 0.1
        mpjpe = np.zeros(num_frames)
        valid_frame_count = 0
        
        for t in range(num_frames):
            valid = valid_joints[:, t]
            if np.sum(valid) > 5:  # Need at least 5 valid joints
                mpjpe[t] = np.mean(np.linalg.norm(
                    final_gt_poses[valid, t, :] - pred_poses_3d[valid, t, :], axis=1))
                valid_frame_count += 1
            else:
                mpjpe[t] = np.nan
        
        valid_mpjpe = mpjpe[~np.isnan(mpjpe)]
        if len(valid_mpjpe) > 0:
            overall_mpjpe = np.mean(valid_mpjpe)
            print(f"Overall MPJPE: {overall_mpjpe:.2f} mm (computed on {valid_frame_count}/{num_frames} frames)")
        else:
            overall_mpjpe = float('inf')
            print("Could not compute MPJPE - insufficient valid joints")
        
        # Set up visualization
        all_gt = final_gt_poses.reshape(-1, 3)
        all_pred = pred_poses_3d.reshape(-1, 3)
        
        # Remove invalid values
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
            ax1.set_title(f'Ground Truth\n(Video Frame {final_frame_indices[frame_idx]})', fontsize=14, pad=20)
            x_gt = final_gt_poses[:, frame_idx, 0]
            y_gt = final_gt_poses[:, frame_idx, 1]
            z_gt = final_gt_poses[:, frame_idx, 2]
            
            # Draw GT skeleton
            for connection in connections:
                start = final_gt_poses[connection[0], frame_idx, :]
                end = final_gt_poses[connection[1], frame_idx, :]
                ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                         'b-', linewidth=3, alpha=0.8)
            
            # Draw GT joints with larger markers
            ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=100, alpha=0.9, edgecolors='darkblue', linewidth=2)
            
            # Add joint indices as text labels for GT with enhanced visibility
            for joint_idx in range(17):
                # Add text with black outline for better visibility
                text_obj = ax1.text(x_gt[joint_idx], y_gt[joint_idx], z_gt[joint_idx], 
                        str(joint_idx), fontsize=24, color='yellow', weight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7, edgecolor='white'))
            
            # Highlight hip (joint 14) for GT
            ax1.scatter(x_gt[14], y_gt[14], z_gt[14], c='green', s=200, marker='*', 
                       alpha=1.0, edgecolors='darkgreen', linewidth=3)
            
            # Plot MediaPipe prediction
            ax2.set_title(f'MediaPipe Prediction\n(Video Frame {final_frame_indices[frame_idx]})', fontsize=14, pad=20)
            valid = visibilities[:, frame_idx] > 0.1
            
            if np.any(valid):
                # Draw MediaPipe skeleton first
                for connection in connections:
                    if valid[connection[0]] and valid[connection[1]]:
                        start = pred_poses_3d[connection[0], frame_idx, :]
                        end = pred_poses_3d[connection[1], frame_idx, :]
                        ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                                 'r-', linewidth=3, alpha=0.8)
                
                # Draw all joints for MediaPipe with enhanced visibility
                for joint_idx in range(17):
                    x_pos = pred_poses_3d[joint_idx, frame_idx, 0]
                    y_pos = pred_poses_3d[joint_idx, frame_idx, 1]
                    z_pos = pred_poses_3d[joint_idx, frame_idx, 2]
                    
                    if valid[joint_idx]:
                        # Valid joints - red with larger markers
                        ax2.scatter(x_pos, y_pos, z_pos, 
                                   c='red', s=100, alpha=0.9, 
                                   edgecolors='darkred', linewidth=2)
                        # Enhanced text labels for valid joints
                        text_obj = ax2.text(x_pos, y_pos, z_pos, 
                                str(joint_idx), fontsize=24, color='yellow', weight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7, edgecolor='white'))
                    else:
                        # Invalid joints - gray with smaller markers
                        ax2.scatter(x_pos, y_pos, z_pos, 
                                   c='gray', s=50, alpha=0.5, 
                                   edgecolors='black', linewidth=1)
                        # Text labels for invalid joints
                        text_obj = ax2.text(x_pos, y_pos, z_pos, 
                                str(joint_idx), fontsize=10, color='white', weight='normal',
                                ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='gray', alpha=0.6, edgecolor='black'))
                
                # Highlight hip (joint 14) for MediaPipe if valid
                if valid[14]:
                    ax2.scatter(pred_poses_3d[14, frame_idx, 0], pred_poses_3d[14, frame_idx, 1], 
                               pred_poses_3d[14, frame_idx, 2], c='green', s=200, marker='*', 
                               alpha=1.0, edgecolors='darkgreen', linewidth=3)
            
            # Update main title with MPJPE
            frame_error = mpjpe[frame_idx] if not np.isnan(mpjpe[frame_idx]) else 0
            fig.suptitle(f'Ground Truth vs MediaPipe - {seq_name}\n'
                        f'Frame {frame_idx+1}/{num_frames} '
                        f'(Video Frame {final_frame_indices[frame_idx]}) | '
                        f'MPJPE: {frame_error:.1f}mm | '
                        f'Valid Joints: {np.sum(valid)}/17', 
                        fontsize=16, y=0.95)
            
            return ax1, ax2
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=range(num_frames), interval=300, repeat=True, blit=False)
        
        if args.save_video:
            output_path = f'../mpi_mediapipe_comparison_{seq_name.lower()}_center_aligned_with_indices_fixed_head.gif'
            print(f"Saving animation to: {output_path}")
            ani.save(output_path, writer='pillow', fps=3, dpi=120)
            print(f"Comparison GIF saved to: {output_path}")
            
            # Save static image
            update(0)
            plt.tight_layout()
            static_path = output_path.replace('.gif', '.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"Static image saved to: {static_path}")
        
        plt.show()
        
    finally:
        estimator.close()

if __name__ == '__main__':
    main()