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
            cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
            pose_3d = pose_3d @ cam2real
            
            return pose_3d, visibility
        return pose_3d, np.zeros(17)

    def close(self):
        self.pose.close()

def load_mpi_test_frames_with_indices(sequence_name, max_frames=1000):
    """Load ALL available video frames from MPI-INF-3DHP test set with their actual indices"""
    mpi_roots = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp',
        '../motion3d/MPI-INF-3DHP',
        '../motion3d',
        '../videos_test_sequences'  # Also check for generated videos
    ]
    
    possible_paths = []
    for root in mpi_roots:
        possible_paths.extend([
            f'{root}/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence',
            f'{root}/mpi_inf_3dhp_test_set/{sequence_name}/imageFrames',
            f'{root}/test/{sequence_name}/imageSequence',
            f'{root}/{sequence_name}/imageSequence',
            f'{root}'  # For generated videos
        ])
    
    # First try to find image sequences
    video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it contains images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(path, ext)))
            if image_files:
                video_path = path
                print(f"Found image sequence: {video_path}")
                break
    
    # If no image sequence found, try to find generated videos
    if video_path is None:
        for root in mpi_roots:
            video_file = os.path.join(root, f"{sequence_name}.mp4")
            if os.path.exists(video_file):
                print(f"Found video file: {video_file}")
                return load_frames_from_video(video_file, max_frames)
    
    if video_path is None:
        print(f"No video data found for sequence {sequence_name}")
        return None, None, None
    
    # Load image sequence
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    
    image_files.sort()
    
    frames = []
    frame_indices = []
    actual_frame_numbers = []
    
    print(f"Loading frames from {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files[:max_frames]):
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
            frame_indices.append(i)  # Sequential index
            
            # Try to extract actual frame number from filename
            try:
                filename = os.path.basename(img_path)
                # Common patterns: frame_000001.jpg, img_000001.jpg, 000001.jpg
                if '_' in filename:
                    frame_num = int(filename.split('_')[-1].split('.')[0])
                else:
                    frame_num = int(filename.split('.')[0])
                actual_frame_numbers.append(frame_num)
            except:
                actual_frame_numbers.append(i + 1)  # 1-based fallback
    
    print(f"Loaded {len(frames)} frames with indices {actual_frame_numbers[:10]}...")
    return frames, frame_indices, actual_frame_numbers

def load_frames_from_video(video_path, max_frames=1000):
    """Load frames from MP4 video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_indices = []
    actual_frame_numbers = []
    
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_indices.append(frame_count)
        actual_frame_numbers.append(frame_count + 1)  # 1-based
        frame_count += 1
    
    cap.release()
    print(f"Loaded {len(frames)} frames from video")
    return frames, frame_indices, actual_frame_numbers

def load_synchronized_gt_data(args, target_frame_indices, stride=1):
    """Load ground truth data synchronized with specific frame indices"""
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

    # Use stride=1 to get consecutive frames without gaps
    dataset_args = DatasetArgs(
        data_root='../motion3d/',
        n_frames=1,  # Load single frames to have full control
        stride=1,    # No gaps
        flip=False,
        test_augmentation=False,
        data_augmentation=False,
        reverse_augmentation=False,
        out_all=1,
        test_batch_size=1
    )
    
    dataset = Fusion(dataset_args, train=False)
    
    # Find all samples for our target sequence
    target_seq_name = args.sequence_name or ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6'][args.sequence_number % 6]
    sequence_samples = []
    
    print(f"Looking for GT data for sequence: {target_seq_name}")
    
    for i in range(len(dataset)):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            
            current_seq_name = seq[0] if isinstance(seq, (list, tuple)) else str(seq)
            if current_seq_name == target_seq_name:
                sequence_samples.append((i, gt_3D, batch_cam, scale))
                
        except Exception as e:
            continue
    
    print(f"Found {len(sequence_samples)} GT samples for sequence {target_seq_name}")
    
    if not sequence_samples:
        return None, None, None
    
    # Create a mapping of available GT frames
    all_gt_poses = []
    all_gt_indices = []
    
    for sample_idx, (dataset_idx, gt_3D, batch_cam, scale) in enumerate(sequence_samples):
        if isinstance(gt_3D, torch.Tensor):
            gt_3D = gt_3D.clone()
        else:
            gt_3D = torch.tensor(gt_3D)
            
        gt_3D = gt_3D.view(1, -1, 17, 3)  # (1, T, 17, 3)
        gt_3D[:, :, 14] = 0  # Set root joint to 0
        
        # Extract all frames from this sample
        for frame_idx in range(gt_3D.shape[1]):
            pose = gt_3D[0, frame_idx]
            pose = pose - pose[14:15, :]  # Root-relative
            if hasattr(pose, 'cpu'):
                pose = pose.cpu().numpy()
            
            all_gt_poses.append(pose)
            # Map to actual frame indices (assuming sequential mapping)
            actual_frame_idx = sample_idx * gt_3D.shape[1] + frame_idx + 1
            all_gt_indices.append(actual_frame_idx)
    
    print(f"Total GT frames available: {len(all_gt_poses)} with indices {all_gt_indices[:20]}...")
    
    # Now synchronize with target frame indices
    synchronized_poses = []
    synchronized_indices = []
    
    # Create a dictionary for fast lookup
    gt_frame_dict = {idx: pose for idx, pose in zip(all_gt_indices, all_gt_poses)}
    
    for target_idx in target_frame_indices:
        if target_idx in gt_frame_dict:
            synchronized_poses.append(gt_frame_dict[target_idx])
            synchronized_indices.append(target_idx)
        else:
            # Find closest available frame
            closest_idx = min(all_gt_indices, key=lambda x: abs(x - target_idx))
            if abs(closest_idx - target_idx) <= 5:  # Allow small gaps
                synchronized_poses.append(gt_frame_dict[closest_idx])
                synchronized_indices.append(target_idx)
                print(f"Using GT frame {closest_idx} for video frame {target_idx}")
    
    if not synchronized_poses:
        print("No synchronized GT data found")
        return None, None, None
    
    # Stack and apply transformations
    sequence_3d = np.stack(synchronized_poses, axis=1)  # (17, T, 3)
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    print(f"Synchronized {sequence_3d.shape[1]} GT frames with video frames")
    return sequence_3d, target_seq_name, synchronized_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default='TS1', help='Sequence name (TS1, TS2, etc.)')
    parser.add_argument('--num-frames', type=int, default=50, help='Number of frames to process')
    parser.add_argument('--save-video', action='store_true', help='Save animation as GIF')
    parser.add_argument('--frame-start', type=int, default=0, help='Starting frame for comparison')
    parser.add_argument('--frame-stride', type=int, default=1, help='Frame sampling stride')
    args = parser.parse_args()
    
    # Define GT joint names for clarity
    GT_JOINT_NAMES = {
        0: "Head Top", 1: "Neck", 2: "Right Arm", 3: "Right Forearm", 4: "Right Hand",
        5: "Left Arm", 6: "Left Forearm", 7: "Left Hand", 8: "Right Up Leg", 9: "Right Leg",
        10: "Right Foot", 11: "Left Up Leg", 12: "Left Leg", 13: "Left Foot",
        14: "Hip", 15: "Spine", 16: "Head"
    }
    
    print("Ground Truth Joint Mappings:")
    for idx, name in GT_JOINT_NAMES.items():
        print(f"Joint {idx}: {name}")
    
    # Use connections directly for ground truth and MediaPipe
    gt_connections = connections
    mp_connections = connections  # Same connections since we map to same joint indices
    
    estimator = MediaPipe3DPoseEstimator()
    
    try:
        # Load ALL available video frames with their indices
        video_data = load_mpi_test_frames_with_indices(args.sequence_name, max_frames=2000)
        if video_data[0] is None:
            print("No video frames loaded. Exiting.")
            return
        
        all_frames, frame_indices, actual_frame_numbers = video_data
        print(f"Loaded {len(all_frames)} total video frames")
        
        # Sample frames based on arguments
        start_idx = args.frame_start
        end_idx = min(start_idx + args.num_frames * args.frame_stride, len(all_frames))
        
        # Select frames with stride
        selected_indices = list(range(start_idx, end_idx, args.frame_stride))
        selected_frames = [all_frames[i] for i in selected_indices]
        selected_frame_numbers = [actual_frame_numbers[i] for i in selected_indices]
        
        print(f"Selected {len(selected_frames)} frames with stride {args.frame_stride}")
        print(f"Frame numbers: {selected_frame_numbers[:10]}...")
        
        # Load synchronized ground truth data
        gt_poses_3d, seq_name, gt_frame_indices = load_synchronized_gt_data(
            args, selected_frame_numbers)
        
        if gt_poses_3d is None:
            print("Failed to load synchronized ground truth data.")
            return
        
        # Ensure we have matching number of frames
        min_frames = min(len(selected_frames), gt_poses_3d.shape[1])
        selected_frames = selected_frames[:min_frames]
        gt_poses_3d = gt_poses_3d[:, :min_frames, :]
        selected_frame_numbers = selected_frame_numbers[:min_frames]
        
        print(f"Final synchronized frames: {min_frames}")
        
        if min_frames == 0:
            print("No synchronized frames available. Exiting.")
            return
        
        # Estimate 3D poses for selected frames
        print("Estimating MediaPipe 3D poses...")
        pred_poses_3d = []
        visibilities = []
        
        for i, frame in enumerate(selected_frames):
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{len(selected_frames)}")
            
            pose_3d, visibility = estimator.estimate_3d_pose_from_image(frame)
            pred_poses_3d.append(pose_3d)
            visibilities.append(visibility)
        
        pred_poses_3d = np.stack(pred_poses_3d, axis=1)  # (17, T, 3)
        visibilities = np.stack(visibilities, axis=1)  # (17, T)
        
        # Calculate MPJPE
        valid_joints = visibilities > 0.1
        mpjpe = np.zeros(min_frames)
        
        for t in range(min_frames):
            valid = valid_joints[:, t]
            if np.any(valid):
                mpjpe[t] = np.mean(np.linalg.norm(
                    gt_poses_3d[valid, t, :] - pred_poses_3d[valid, t, :], axis=1))
            else:
                mpjpe[t] = np.nan
        
        overall_mpjpe = np.nanmean(mpjpe)
        print(f"Overall MPJPE: {overall_mpjpe:.2f} mm")
        
        # Set up visualization
        valid_gt = gt_poses_3d[~np.isnan(gt_poses_3d) & ~np.isinf(gt_poses_3d)]
        valid_pred = pred_poses_3d[~np.isnan(pred_poses_3d) & ~np.isinf(pred_poses_3d)]
        
        if valid_gt.size > 0 and valid_pred.size > 0:
            all_poses = np.vstack([valid_gt.reshape(-1, 3), valid_pred.reshape(-1, 3)])
        elif valid_gt.size > 0:
            all_poses = valid_gt.reshape(-1, 3)
        else:
            all_poses = valid_pred.reshape(-1, 3)
            
        min_value = np.min(all_poses, axis=0)
        max_value = np.max(all_poses, axis=0)
        padding = (max_value - min_value) * 0.1
        min_value -= padding
        max_value += padding
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
        
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
            
            current_frame_num = selected_frame_numbers[frame_idx]
            
            # Plot ground truth
            ax1.set_title(f'Ground Truth\n(Video Frame {current_frame_num})', fontsize=12)
            x_gt = gt_poses_3d[:, frame_idx, 0]
            y_gt = gt_poses_3d[:, frame_idx, 1]
            z_gt = gt_poses_3d[:, frame_idx, 2]
            
            # Draw GT skeleton
            for connection in gt_connections:
                if (not np.isnan(x_gt[connection[0]]) and not np.isnan(x_gt[connection[1]])):
                    start = gt_poses_3d[connection[0], frame_idx, :]
                    end = gt_poses_3d[connection[1], frame_idx, :]
                    ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                             'b-', linewidth=2, alpha=0.8)
            
            ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=60, alpha=0.9, edgecolors='darkblue')
            ax1.scatter(x_gt[14], y_gt[14], z_gt[14], c='green', s=120, marker='*', 
                       alpha=1.0, edgecolors='darkgreen')
            
            # Plot MediaPipe prediction
            ax2.set_title(f'MediaPipe Prediction\n(Video Frame {current_frame_num})', fontsize=12)
            valid = visibilities[:, frame_idx] > 0.1
            
            if np.any(valid):
                x_pred = pred_poses_3d[:, frame_idx, 0]
                y_pred = pred_poses_3d[:, frame_idx, 1]
                z_pred = pred_poses_3d[:, frame_idx, 2]
                
                # Draw MediaPipe skeleton
                for connection in mp_connections:
                    if valid[connection[0]] and valid[connection[1]]:
                        start = pred_poses_3d[connection[0], frame_idx, :]
                        end = pred_poses_3d[connection[1], frame_idx, :]
                        ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                                 'r-', linewidth=2, alpha=0.8)
                
                # Plot joints
                valid_x = x_pred[valid]
                valid_y = y_pred[valid]
                valid_z = z_pred[valid]
                ax2.scatter(valid_x, valid_y, valid_z, c='red', s=60, alpha=0.9, edgecolors='darkred')
                
                # Highlight root joint
                if valid[14]:
                    ax2.scatter(x_pred[14], y_pred[14], z_pred[14], c='green', s=120, marker='*', 
                               alpha=1.0, edgecolors='darkgreen')
            
            # Update title with MPJPE
            frame_mpjpe = mpjpe[frame_idx] if not np.isnan(mpjpe[frame_idx]) else 0
            fig.suptitle(f'Synchronized 3D Pose Comparison - {seq_name}\n'
                        f'Video Frame {current_frame_num} | MPJPE: {frame_mpjpe:.1f}mm | '
                        f'Overall MPJPE: {overall_mpjpe:.1f}mm', fontsize=14)
            
            return ax1, ax2
        
        # Create animation
        print("Creating visualization...")
        ani = FuncAnimation(fig, update, frames=range(min_frames), interval=200, repeat=True, blit=False)
        
        if args.save_video:
            output_path = f'../mpi_mediapipe_synchronized_{seq_name.lower()}_stride{args.frame_stride}.gif'
            print(f"Saving animation to: {output_path}")
            ani.save(output_path, writer='pillow', fps=5, dpi=100)
            print(f"✓ Animation saved: {output_path}")
            
            # Save static image
            update(0)
            plt.tight_layout()
            static_path = output_path.replace('.gif', '.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"✓ Static image saved: {static_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        estimator.close()

if __name__ == '__main__':
    main()