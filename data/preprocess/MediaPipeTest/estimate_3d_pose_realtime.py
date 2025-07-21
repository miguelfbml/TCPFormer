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
from scipy.interpolate import interp1d

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from data.reader.motion_dataset import MPI3DHP

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
            0: 0,    # nose -> head top
            7: 16,   # left_ear -> head
            8: 16,   # right_ear -> head
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
            
            # Estimate missing joints
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
            cam2real = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)  # Identity to test tilt
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
                # Extract frame index from filename (e.g., 'frame_0001.jpg' -> 1)
                frame_idx = int(os.path.basename(img_path).split('_')[-1].split('.')[0])
                frame_indices.append(frame_idx)
            except:
                frame_idx = len(frame_indices) + 1
                frame_indices.append(frame_idx)
    
    print(f"Video frame filenames: {[os.path.basename(f) for f in image_files[:10]]}")
    return frames, frame_indices

def interpolate_poses(valid_poses, valid_indices, num_frames):
    """Interpolate GT poses for all frames based on valid frames"""
    if len(valid_poses) < 2:
        print("Not enough valid poses for interpolation")
        return None, None
    
    valid_indices = np.array(valid_indices)
    valid_poses = np.array(valid_poses)  # Shape: (T_valid, 17, 3)
    
    # Create interpolation functions for each joint and coordinate
    interpolated_poses = np.zeros((num_frames, 17, 3))
    target_indices = np.arange(1, num_frames + 1)
    
    for joint in range(17):
        for coord in range(3):
            f = interp1d(valid_indices, valid_poses[:, joint, coord], kind='linear', fill_value='extrapolate')
            interpolated_poses[:, joint, coord] = f(target_indices)
    
    return interpolated_poses.transpose(1, 0, 2), target_indices.tolist()  # (17, T, 3)

def load_test_3d_data_from_dataset(args, num_frames, video_frame_indices):
    """Load 3D ground truth data from MPI-INF-3DHP dataset using MPI3DHP class"""
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
        subject_id: int

    dataset_args = DatasetArgs(
        data_root='../motion3d/',
        n_frames=num_frames,
        stride=1,
        flip=False,
        test_augmentation=False,
        data_augmentation=False,
        reverse_augmentation=False,
        out_all=1,
        test_batch_size=1,
        subject_id=args.subject_id if hasattr(args, 'subject_id') else None
    )
    
    dataset = MPI3DHP(dataset_args, train=False)
    
    target_seq_name = args.sequence_name or ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6'][args.sequence_number % 6]
    
    # Inspect dataset structure
    try:
        data = np.load('../motion3d/data_test_3dhp.npz', allow_pickle=True)['data'].item()
        print(f"Dataset keys for {target_seq_name}: {data[target_seq_name].keys()}")
        print(f"GT data shape: {data[target_seq_name]['data_3d'].shape}")
        print(f"Valid frames shape: {data[target_seq_name]['valid'].shape}")
        print(f"Number of valid frames: {np.sum(data[target_seq_name]['valid'])}")
        print(f"First 20 valid frame flags: {data[target_seq_name]['valid'][:20]}")
        if 'frame_indices' in data[target_seq_name]:
            print(f"Dataset frame indices: {data[target_seq_name]['frame_indices'][:20]}")
        if 'subject_id' in data[target_seq_name]:
            print(f"Subject IDs: {data[target_seq_name]['subject_id']}")
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
    
    best_sequence = None
    best_matched_frames = 0
    best_sequence_data = []
    best_frame_indices = []
    
    for i in range(len(dataset)):
        try:
            pose_2d, pose_3d_normalized, pose_3d, valid_frame, seq_name = dataset[i]
            
            if seq_name != target_seq_name:
                continue
            
            # Check subject ID if specified
            if hasattr(args, 'subject_id') and args.subject_id is not None:
                try:
                    data = np.load('../motion3d/data_test_3dhp.npz', allow_pickle=True)['data'].item()
                    if 'subject_id' in data[target_seq_name] and data[target_seq_name]['subject_id'] != args.subject_id:
                        continue
                except:
                    pass
            
            # pose_3d: (T, 17, 3)
            total_frames = pose_3d.shape[0]
            print(f"Ground truth sequence {seq_name} (index {i}) has {total_frames} frames, selecting up to {num_frames} frames")
            
            # Debugging: Print raw pose for hip joint
            print(f"Raw GT pose (first frame, hip joint 14): {pose_3d[0, 14, :]}")
            
            # Collect valid GT poses
            sequence_data = []
            frame_indices = []
            matched_frames = 0
            for frame_idx in range(total_frames):
                if valid_frame[frame_idx].item():  # Only use valid frames
                    if (frame_idx + 1) in video_frame_indices:  # Ensure frame index matches video
                        pose = pose_3d[frame_idx].numpy()  # (17, 3)
                        # Make root-relative to hip (joint 14)
                        pose = pose - pose[14:15, :]
                        sequence_data.append(pose)
                        frame_indices.append(frame_idx + 1)  # 1-based indexing
                        matched_frames += 1
                        print(f"Matched GT frame {frame_idx + 1} to video frame {frame_idx + 1}")
                    else:
                        print(f"Frame {frame_idx + 1} not in video_frame_indices, skipping")
                else:
                    print(f"Frame {frame_idx + 1} is invalid, skipping")
                
                if matched_frames >= num_frames:
                    break
            
            # Keep track of the sequence with the most matched frames
            if matched_frames > best_matched_frames:
                best_matched_frames = matched_frames
                best_sequence = i
                best_sequence_data = sequence_data
                best_frame_indices = frame_indices
            
            # Stop if we have enough frames
            if matched_frames >= num_frames:
                break
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if not best_sequence_data:
        print(f"No valid ground truth data found for sequence {target_seq_name}: {best_matched_frames}/{num_frames} frames matched")
        return None, None, None
    
    # Interpolate poses if fewer than num_frames are valid
    if best_matched_frames < num_frames:
        print(f"Interpolating GT poses: {best_matched_frames} valid frames found, interpolating to {num_frames}")
        sequence_3d, frame_indices = interpolate_poses(best_sequence_data, best_frame_indices, num_frames)
        if sequence_3d is None:
            print("Interpolation failed due to insufficient valid poses")
            return None, None, None
    else:
        sequence_3d = np.stack(best_sequence_data, axis=1)  # (17, T, 3)
    
    # Apply camera transformation to align with MediaPipe
    cam2real = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)  # Identity to test tilt
    sequence_3d = sequence_3d @ cam2real
    
    # Debugging: Print pose ranges for GT after transformations
    print(f"GT pose range after transformation (min, max): X={sequence_3d[:, :, 0].min():.2f}, {sequence_3d[:, :, 0].max():.2f}; "
          f"Y={sequence_3d[:, :, 1].min():.2f}, {sequence_3d[:, :, 1].max():.2f}; "
          f"Z={sequence_3d[:, :, 2].min():.2f}, {sequence_3d[:, :, 2].max():.2f}")
    
    # Debugging: Print matched frame indices
    print(f"Matched GT frame indices (sequence {best_sequence}): {frame_indices}")
    
    # Debugging: Print sample poses for all frames
    for t, frame_idx in enumerate(frame_indices):
        print(f"Sample GT pose (frame {frame_idx}, hip joint 14): {sequence_3d[14, t, :]}")
        print(f"Sample GT pose (frame {frame_idx}, head joint 0): {sequence_3d[0, t, :]}")
    
    return sequence_3d, target_seq_name, frame_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, help='Sequence name (TS1, TS2, etc.)')
    parser.add_argument('--num-frames', type=int, default=50, help='Number of frames to process')
    parser.add_argument('--save-video', action='store_true', help='Save animation as GIF')
    parser.add_argument('--frame-start', type=int, default=0, help='Starting frame for comparison')
    parser.add_argument('--subject-id', type=int, default=None, help='Subject ID for multi-sequence datasets')
    args = parser.parse_args()
    
    # Define GT joint names for clarity
    GT_JOINT_NAMES = {
        0: "Head Top",
        1: "Neck",
        2: "Right Arm",
        3: "Right Forearm",
        4: "Right Hand",
        5: "Left Arm",
        6: "Left Forearm",
        7: "Left Hand",
        8: "Right Up Leg",
        9: "Right Leg",
        10: "Right Foot",
        11: "Left Up Leg",
        12: "Left Leg",
        13: "Left Foot",
        14: "Hip",
        15: "Spine",
        16: "Head"
    }
    
    # Print joint mappings for clarity
    print("Ground Truth Joint Mappings:")
    for idx, name in GT_JOINT_NAMES.items():
        print(f"Joint {idx}: {name}")
    
    # Use connections directly for ground truth
    gt_connections = connections
    
    # Define MediaPipe connections to match GT joint order
    mp_connections = [
        (0, 16),    # head top -> head
        (16, 1),    # head -> neck
        (1, 2),     # neck -> right arm
        (2, 3),     # right arm -> right forearm
        (3, 4),     # right forearm -> right hand
        (1, 5),     # neck -> left arm
        (5, 6),     # left arm -> left forearm
        (6, 7),     # left forearm -> left hand
        (1, 15),    # neck -> spine
        (15, 14),   # spine -> hip
        (14, 8),    # hip -> right up leg
        (8, 9),     # right up leg -> right leg
        (9, 10),    # right leg -> right foot
        (14, 11),   # hip -> left up leg
        (11, 12),   # left up leg -> left leg
        (12, 13),   # left leg -> left foot
    ]
    
    estimator = MediaPipe3DPoseEstimator()
    
    try:
        # Load video frames first
        frames, frame_indices = load_mpi_test_frames(args.sequence_name, args.num_frames)
        if not frames:
            print("No video frames loaded. Exiting.")
            return
        
        print(f"Loaded {len(frames)} video frames with indices: {frame_indices[:10]}...")
        
        # Load ground truth data, matching the video frame indices
        num_frames = len(frames)
        gt_poses_3d, seq_name, gt_frame_indices = load_test_3d_data_from_dataset(args, num_frames, frame_indices)
        if gt_poses_3d is None:
            print("Failed to load ground truth data.")
            return
        
        # Synchronize frames
        num_frames = min(len(frames), gt_poses_3d.shape[1], args.num_frames)
        start_frame = args.frame_start
        end_frame = min(start_frame + args.num_frames, num_frames)
        
        # Ensure frame indices match
        common_indices = sorted(set(frame_indices).intersection(set(gt_frame_indices)))
        if not common_indices:
            print("No common frame indices between GT and video frames. Exiting.")
            return
        
        # Filter frames and GT poses to common indices
        video_frame_mask = [frame_indices[i] in common_indices for i in range(len(frame_indices))]
        gt_frame_mask = [gt_frame_indices[i] in common_indices for i in range(len(gt_frame_indices))]
        
        frames = [frames[i] for i in range(len(frames)) if video_frame_mask[i]]
        gt_poses_3d = gt_poses_3d[:, gt_frame_mask, :]
        synced_frame_indices = [frame_indices[i] for i in range(len(frame_indices)) if video_frame_mask[i]]
        min_frames = len(synced_frame_indices)
        
        print(f"Synchronized {min_frames} frames with indices: {synced_frame_indices}")
        
        if min_frames == 0:
            print("No synchronized frames available. Exiting.")
            return
        
        # Estimate 3D poses
        pred_poses_3d = []
        visibilities = []
        for frame_idx, frame in zip(synced_frame_indices, frames):
            pose_3d, visibility = estimator.estimate_3d_pose_from_image(frame)
            pred_poses_3d.append(pose_3d)
            visibilities.append(visibility)
            print(f"Processed MediaPipe pose for frame {frame_idx}")
        
        pred_poses_3d = np.stack(pred_poses_3d, axis=1)  # (17, T, 3)
        visibilities = np.stack(visibilities, axis=1)  # (17, T)
        
        # Debugging: Print pose ranges for MediaPipe
        print(f"MediaPipe pose range (min, max): X={pred_poses_3d[:, :, 0].min():.2f}, {pred_poses_3d[:, :, 0].max():.2f}; "
              f"Y={pred_poses_3d[:, :, 1].min():.2f}, {pred_poses_3d[:, :, 1].max():.2f}; "
              f"Z={pred_poses_3d[:, :, 2].min():.2f}, {pred_poses_3d[:, :, 2].max():.2f}")
        
        # Debugging: Print sample poses for all frames
        for t, frame_idx in enumerate(synced_frame_indices):
            print(f"Sample poses (frame {frame_idx}, hip joint 14):")
            print(f"GT: {gt_poses_3d[14, t, :]}")
            print(f"MediaPipe: {pred_poses_3d[14, t, :]}")
            print(f"Sample poses (frame {frame_idx}, head joint 0):")
            print(f"GT: {gt_poses_3d[0, t, :]}")
            print(f"MediaPipe: {pred_poses_3d[0, t, :]}")
        
        # Calculate MPJPE
        valid_joints = visibilities > 0.1
        mpjpe = np.zeros(min_frames)
        for t in range(min_frames):
            valid = valid_joints[:, t]
            if np.any(valid):
                mpjpe[t] = np.mean(np.linalg.norm(
                    gt_poses_3d[valid, t, :] - pred_poses_3d[valid, t, :], axis=1))
        
        overall_mpjpe = np.mean(mpjpe[np.isfinite(mpjpe)])
        print(f"Overall MPJPE: {overall_mpjpe:.2f} mm")
        
        # Set up visualization
        valid_gt = gt_poses_3d[~np.isnan(gt_poses_3d) & ~np.isinf(gt_poses_3d)]
        valid_pred = pred_poses_3d[~np.isnan(pred_poses_3d) & ~np.isinf(pred_poses_3d)]
        all_poses = np.vstack([valid_gt.reshape(-1, 3), valid_pred.reshape(-1, 3)])
        min_value = np.min(all_poses, axis=0)
        max_value = np.max(all_poses, axis=0)
        padding = (max_value - min_value) * 0.1
        min_value -= padding
        max_value += padding
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': '3d'})
        
        def update(frame_idx):
            ax1.clear()
            ax2.clear()
            
            for ax in [ax1, ax2]:
                ax.set_xlim3d([min_value[0], max_value[0]])
                ax.set_ylim3d([min_value[1], max_value[1]])
                ax.set_zlim3d([min_value[2], max_value[2]])
                ax.set_xlabel('X (mm)', fontsize=10)
                ax.set_ylabel('Y (mm)', fontsize=10)
                ax.set_zlabel('Z (mm)', fontsize=10)
            
            # Plot ground truth
            ax1.set_title(f'Ground Truth\n(Frame {synced_frame_indices[frame_idx]})', fontsize=12)
            x_gt = gt_poses_3d[:, frame_idx, 0]
            y_gt = gt_poses_3d[:, frame_idx, 1]
            z_gt = gt_poses_3d[:, frame_idx, 2]
            
            for connection in gt_connections:
                start = gt_poses_3d[connection[0], frame_idx, :]
                end = gt_poses_3d[connection[1], frame_idx, :]
                ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                         'b-', linewidth=2, alpha=0.8)
            
            ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=60, alpha=0.9, edgecolors='darkblue')
            ax1.scatter(x_gt[14], y_gt[14], z_gt[14], c='green', s=120, marker='*', 
                       alpha=1.0, edgecolors='darkgreen')
            
            # Add joint indices for ground truth
            for i in range(17):
                if not np.isnan(x_gt[i]) and not np.isinf(x_gt[i]):
                    ax1.text(x_gt[i], y_gt[i], z_gt[i], str(i), color='black', fontsize=8)
            
            # Plot prediction
            ax2.set_title(f'MediaPipe Prediction\n(Frame {synced_frame_indices[frame_idx]})', fontsize=12)
            valid = visibilities[:, frame_idx] > 0.1
            x_pred = pred_poses_3d[valid, frame_idx, 0]
            y_pred = pred_poses_3d[valid, frame_idx, 1]
            z_pred = pred_poses_3d[valid, frame_idx, 2]
            
            if np.any(valid):
                ax2.scatter(x_pred, y_pred, z_pred, c='red', s=60, alpha=0.9, edgecolors='darkred')
                for connection in mp_connections:
                    if valid[connection[0]] and valid[connection[1]]:
                        start = pred_poses_3d[connection[0], frame_idx, :]
                        end = pred_poses_3d[connection[1], frame_idx, :]
                        ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                                 'r-', linewidth=2, alpha=0.8)
                
                if valid[14]:
                    ax2.scatter(pred_poses_3d[14, frame_idx, 0], pred_poses_3d[14, frame_idx, 1], 
                               pred_poses_3d[14, frame_idx, 2], c='green', s=120, marker='*', 
                               alpha=1.0, edgecolors='darkgreen')
                
                # Add joint indices for predictions
                for i in range(17):
                    if valid[i]:
                        ax2.text(pred_poses_3d[i, frame_idx, 0], pred_poses_3d[i, frame_idx, 1], 
                                 pred_poses_3d[i, frame_idx, 2], str(i), color='black', fontsize=8)
            
            # Update title with MPJPE
            fig.suptitle(f'Ground Truth vs MediaPipe - {seq_name}\n'
                        f'Frame {synced_frame_indices[frame_idx]}, MPJPE: {mpjpe[frame_idx]:.1f}mm', 
                        fontsize=14)
            
            return ax1, ax2
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=range(min_frames), interval=150, repeat=True, blit=False)
        if args.save_video:
            output_path = f'../mpi_mediapipe_comparison_{seq_name.lower()}.gif'
            ani.save(output_path, writer='pillow', fps=8, dpi=100)
            print(f"Comparison GIF saved to: {output_path}")
            
            update(0)
            plt.tight_layout()
            plt.savefig(output_path.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
            print(f"Static image saved to: {output_path.replace('.gif', '.png')}")
        
        plt.show()
        
    finally:
        estimator.close()

if __name__ == '__main__':
    main()