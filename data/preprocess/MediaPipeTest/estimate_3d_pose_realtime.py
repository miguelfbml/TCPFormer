'''
MediaPipe 3D Pose Estimation vs Ground Truth Comparison

cd data/preprocess

# Try to load real frames with 3D pose estimation
python estimate_3d_pose_realtime.py --sequence-name TS1 --num-frames 30

# Save as GIF
python estimate_3d_pose_realtime.py --sequence-name TS1 --save-video --num-frames 20
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
import time
import glob

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from data.reader.motion_dataset import MPI3DHP, Fusion
from data.const import H36M_TO_MPI

class MediaPipe3DPoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Heavy model for better 3D estimation
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe to MPI joint mapping for 3D poses
        self.mp_to_mpi_mapping = {
            # MediaPipe landmark index -> MPI joint index
            0: 9,   # nose -> nose
            11: 11, # left_shoulder -> left shoulder  
            12: 14, # right_shoulder -> right shoulder
            13: 12, # left_elbow -> left elbow
            14: 15, # right_elbow -> right elbow
            15: 13, # left_wrist -> left wrist
            16: 16, # right_wrist -> right wrist
            23: 4,  # left_hip -> left hip
            24: 1,  # right_hip -> right hip
            25: 5,  # left_knee -> left knee
            26: 2,  # right_knee -> right knee
            27: 6,  # left_ankle -> left ankle
            28: 3,  # right_ankle -> right ankle
            7: 10,  # left_ear -> head
            8: 10,  # right_ear -> head
        }
        
        # Estimate missing joints from available ones
        self.missing_joints_estimation = {
            0: [9, 10],     # root from nose and head
            7: [8, 9],      # spine from thorax and nose  
            8: [11, 14],    # thorax from shoulders
        }
        
    def estimate_3d_pose_from_image(self, image):
        """Estimate 3D pose from a single image using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        pose_3d = np.zeros((17, 4))  # x, y, z, visibility
        
        if results.pose_world_landmarks:  # Use world landmarks for 3D coordinates
            landmarks = results.pose_world_landmarks.landmark
            
            # First pass: map directly available joints
            for mp_idx, mpi_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    pose_3d[mpi_idx] = [
                        landmark.x * 1000,  # Convert to mm (MediaPipe uses meters)
                        landmark.y * 1000,  # Convert to mm
                        landmark.z * 1000,  # Convert to mm
                        landmark.visibility
                    ]
            
            # Second pass: estimate missing joints from available ones
            for missing_joint, source_joints in self.missing_joints_estimation.items():
                if pose_3d[missing_joint, 3] == 0:  # Joint not detected
                    valid_sources = [j for j in source_joints if pose_3d[j, 3] > 0.1]
                    if len(valid_sources) >= 2:
                        # Average position of source joints
                        avg_x = np.mean([pose_3d[j, 0] for j in valid_sources])
                        avg_y = np.mean([pose_3d[j, 1] for j in valid_sources])
                        avg_z = np.mean([pose_3d[j, 2] for j in valid_sources])
                        pose_3d[missing_joint] = [avg_x, avg_y, avg_z, 0.5]
            
            # Make pose root-relative (subtract root joint - index 14)
            if pose_3d[14, 3] > 0.1:  # If root joint is detected
                root_pos = pose_3d[14, :3]
                pose_3d[:, :3] = pose_3d[:, :3] - root_pos
            
        return pose_3d, results
    
    def close(self):
        self.pose.close()

def load_mpi_test_frames(sequence_name, num_frames=50):
    """Load actual video frames from MPI-INF-3DHP test set"""
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
            f'{root}/mpi_inf_3dhp_test_set/{sequence_name}/images',
            f'{root}/mpi_inf_3dhp_test_set/{sequence_name}',
            f'{root}/test/{sequence_name}/imageSequence',
            f'{root}/Test/{sequence_name}/imageSequence',
            f'{root}/MPI_INF_3DHP/test/{sequence_name}/imageSequence',
        ])
    
    print(f"Looking for video frames for sequence: {sequence_name}")
    
    video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            print(f"Found video path: {video_path}")
            break
    
    if video_path is None:
        print("Video frames not found. Checked paths:")
        for path in possible_paths[:5]:
            print(f"  - {path} (exists: {os.path.exists(path)})")
        return None
    
    print(f"Loading frames from: {video_path}")
    
    frames = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        files = glob.glob(os.path.join(video_path, ext))
        files.extend(glob.glob(os.path.join(video_path, ext.upper())))
        image_files.extend(files)
    
    if image_files:
        image_files.sort()
        print(f"Found {len(image_files)} image files")
        
        for i, img_path in enumerate(image_files[:num_frames]):
            frame = cv2.imread(img_path)
            if frame is not None:
                frames.append(frame)
                if i % 50 == 0:
                    print(f"Loaded {i+1}/{min(num_frames, len(image_files))} images...")
        
        print(f"Successfully loaded {len(frames)} frames")
    else:
        print("No image files found")
        return None
    
    return frames

def load_test_3d_data_from_dataset(args):
    """Load 3D test data from MPI-INF-3DHP dataset"""
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
        data_root='../../motion3d/', 
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
    
    print(f"Dataset length: {len(dataset)}")
    
    sequence_data = []
    target_seq_name = None
    processed_samples = 0
    
    available_sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    if args.sequence_name:
        target_seq_name = args.sequence_name
    else:
        target_seq_name = available_sequences[args.sequence_number % len(available_sequences)]
    
    print(f"Looking for sequence: {target_seq_name}")
    
    # Load 3D ground truth data
    for i in range(len(dataset)):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            
            # Extract sequence name
            if isinstance(seq, (list, tuple)):
                current_seq_name = seq[0]
            elif isinstance(seq, torch.Tensor):
                current_seq_name = seq.item() if seq.numel() == 1 else str(seq)
            else:
                current_seq_name = str(seq)
            
            if current_seq_name != target_seq_name:
                continue
            
            print(f"Found matching sequence: {current_seq_name} (sample {i})")
            
            # Process 3D ground truth exactly like train_3dhp.py
            if isinstance(gt_3D, torch.Tensor):
                gt_3D = gt_3D.clone()
            else:
                gt_3D = torch.tensor(gt_3D)
                
            gt_3D = gt_3D.view(1, -1, 17, 3)  # N=1, T, 17, 3
            gt_3D[:, :, 14] = 0  # Set root joint to 0
            
            # Extract center frame
            center_frame = gt_3D.shape[1] // 2
            center_pose = gt_3D[0, center_frame]  # (17, 3)
            
            # Make root-relative
            center_pose = center_pose - center_pose[14:15, :]
            
            # Convert to numpy
            if hasattr(center_pose, 'cpu'):
                center_pose = center_pose.cpu().numpy()
            else:
                center_pose = np.array(center_pose)
            
            sequence_data.append(center_pose)
            processed_samples += 1
            
            if processed_samples >= args.num_frames:
                break
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Processed {processed_samples} samples for sequence {target_seq_name}")
    
    if not sequence_data:
        print("No valid 3D ground truth data found")
        return None, None
    
    # Stack frames: (17, T, 3)
    sequence_3d = np.stack(sequence_data, axis=1)
    
    # Apply camera transformation (same as compare_gt_pred.py)
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    print(f"Stacked 3D sequence shape: {sequence_3d.shape}")
    
    return sequence_3d, target_seq_name

def create_3d_pose_visualization(estimator, frames, gt_poses_3d, seq_name, args):
    """Create 3D pose visualization comparing MediaPipe 3D vs Ground Truth 3D"""
    
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    fig.suptitle(f'3D Pose Comparison - {seq_name}', fontsize=16)
    
    estimated_poses_3d = []
    
    # MPI skeleton connections
    connections = [
        (10, 9), (9, 8), (8, 11), (8, 14), (14, 15), (15, 16),
        (11, 12), (12, 13), (8, 7), (7, 0), (0, 4), (0, 1),
        (1, 2), (2, 3), (4, 5), (5, 6)
    ]
    
    def update(frame_idx):
        if frame_idx >= len(frames) or frame_idx >= gt_poses_3d.shape[1]:
            return
        
        # Get frame and ground truth
        frame = frames[frame_idx]
        gt_pose_3d = gt_poses_3d[:, frame_idx, :]  # (17, 3)
        
        # Estimate MediaPipe 3D pose
        if frame_idx >= len(estimated_poses_3d):
            estimated_pose_3d, mp_results = estimator.estimate_3d_pose_from_image(frame)
            estimated_poses_3d.append(estimated_pose_3d)
        else:
            estimated_pose_3d = estimated_poses_3d[frame_idx]
        
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Ground Truth 3D Pose
        ax1.set_title(f'Ground Truth 3D Pose - {seq_name}', fontsize=14)
        
        # Plot GT 3D keypoints
        x_gt, y_gt, z_gt = gt_pose_3d[:, 0], gt_pose_3d[:, 1], gt_pose_3d[:, 2]
        ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=60, alpha=0.9, edgecolors='darkblue')
        
        # Draw GT skeleton connections
        for connection in connections:
            joint1, joint2 = connection
            if joint1 < len(gt_pose_3d) and joint2 < len(gt_pose_3d):
                ax1.plot([gt_pose_3d[joint1, 0], gt_pose_3d[joint2, 0]], 
                        [gt_pose_3d[joint1, 1], gt_pose_3d[joint2, 1]], 
                        [gt_pose_3d[joint1, 2], gt_pose_3d[joint2, 2]], 
                        'b-', linewidth=2, alpha=0.8)
        
        # Highlight root joint (should be at origin)
        ax1.scatter(x_gt[14], y_gt[14], z_gt[14], c='green', s=120, marker='*', 
                   alpha=1.0, edgecolors='darkgreen', linewidth=2)
        
        # Set equal aspect ratio for GT
        max_range_gt = np.array([x_gt.max()-x_gt.min(), y_gt.max()-y_gt.min(), 
                                z_gt.max()-z_gt.min()]).max() / 2.0
        mid_x_gt = (x_gt.max()+x_gt.min()) * 0.5
        mid_y_gt = (y_gt.max()+y_gt.min()) * 0.5
        mid_z_gt = (z_gt.max()+z_gt.min()) * 0.5
        ax1.set_xlim(mid_x_gt - max_range_gt, mid_x_gt + max_range_gt)
        ax1.set_ylim(mid_y_gt - max_range_gt, mid_y_gt + max_range_gt)
        ax1.set_zlim(mid_z_gt - max_range_gt, mid_z_gt + max_range_gt)
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        
        # Plot 2: MediaPipe 3D Predictions
        ax2.set_title(f'MediaPipe 3D Pose Estimation - {seq_name}', fontsize=14)
        
        # Filter valid MediaPipe joints
        valid_est = estimated_pose_3d[:, 3] > 0.1  # visibility > 0.1
        
        if np.any(valid_est):
            x_est = estimated_pose_3d[valid_est, 0]
            y_est = estimated_pose_3d[valid_est, 1] 
            z_est = estimated_pose_3d[valid_est, 2]
            
            ax2.scatter(x_est, y_est, z_est, c='red', s=60, alpha=0.9, edgecolors='darkred')
            
            # Draw MediaPipe skeleton connections
            for connection in connections:
                joint1, joint2 = connection
                if (joint1 < len(estimated_pose_3d) and joint2 < len(estimated_pose_3d) and
                    estimated_pose_3d[joint1, 3] > 0.1 and estimated_pose_3d[joint2, 3] > 0.1):
                    ax2.plot([estimated_pose_3d[joint1, 0], estimated_pose_3d[joint2, 0]], 
                            [estimated_pose_3d[joint1, 1], estimated_pose_3d[joint2, 1]], 
                            [estimated_pose_3d[joint1, 2], estimated_pose_3d[joint2, 2]], 
                            'r-', linewidth=2, alpha=0.8)
            
            # Highlight root joint
            if estimated_pose_3d[14, 3] > 0.1:
                ax2.scatter(estimated_pose_3d[14, 0], estimated_pose_3d[14, 1], estimated_pose_3d[14, 2], 
                           c='green', s=120, marker='*', alpha=1.0, edgecolors='darkgreen', linewidth=2)
            
            # Set equal aspect ratio for MediaPipe
            all_coords = estimated_pose_3d[valid_est, :3]
            max_range_est = np.array([all_coords[:, 0].max()-all_coords[:, 0].min(), 
                                     all_coords[:, 1].max()-all_coords[:, 1].min(),
                                     all_coords[:, 2].max()-all_coords[:, 2].min()]).max() / 2.0
            mid_x_est = (all_coords[:, 0].max()+all_coords[:, 0].min()) * 0.5
            mid_y_est = (all_coords[:, 1].max()+all_coords[:, 1].min()) * 0.5
            mid_z_est = (all_coords[:, 2].max()+all_coords[:, 2].min()) * 0.5
            ax2.set_xlim(mid_x_est - max_range_est, mid_x_est + max_range_est)
            ax2.set_ylim(mid_y_est - max_range_est, mid_y_est + max_range_est)
            ax2.set_zlim(mid_z_est - max_range_est, mid_z_est + max_range_est)
        else:
            ax2.text(0, 0, 0, 'No valid 3D pose detected', fontsize=12, ha='center')
            ax2.set_xlim(-500, 500)
            ax2.set_ylim(-500, 500)
            ax2.set_zlim(-500, 500)
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        
        # Calculate 3D MPJPE error
        detected_joints = np.sum(valid_est)
        if np.any(valid_est):
            # Compare only joints that are detected by MediaPipe
            valid_indices = np.where(valid_est)[0]
            if len(valid_indices) > 0:
                gt_valid = gt_pose_3d[valid_indices]
                est_valid = estimated_pose_3d[valid_indices, :3]
                mpjpe = np.mean(np.linalg.norm(gt_valid - est_valid, axis=1))
                
                fig.suptitle(f'3D Pose Comparison - {seq_name}\n'
                            f'Frame: {frame_idx+1}/{len(frames)} | '
                            f'MediaPipe Joints: {detected_joints}/17 | '
                            f'MPJPE: {mpjpe:.1f}mm', fontsize=16)
            else:
                fig.suptitle(f'3D Pose Comparison - {seq_name}\n'
                            f'Frame: {frame_idx+1}/{len(frames)} | '
                            f'MediaPipe Joints: {detected_joints}/17 | '
                            f'No valid comparison', fontsize=16)
        else:
            fig.suptitle(f'3D Pose Comparison - {seq_name}\n'
                        f'Frame: {frame_idx+1}/{len(frames)} | '
                        f'MediaPipe Joints: {detected_joints}/17 | '
                        f'No pose detected', fontsize=16)
        
        plt.tight_layout()
        
        return [ax1, ax2]
    
    return update, fig

def create_dummy_frames_3d(gt_poses_3d, num_frames, image_size=(640, 480)):
    """Create dummy frames for 3D pose testing"""
    frames = []
    
    for t in range(min(num_frames, gt_poses_3d.shape[1])):
        # Create a simple background
        frame = np.random.randint(50, 100, (image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Add some simple visual elements based on 3D pose
        pose_3d = gt_poses_3d[:, t, :]
        
        # Project 3D to 2D for visualization (simple orthographic projection)
        # Normalize and center the pose
        if np.any(~np.isnan(pose_3d)):
            x_2d = ((pose_3d[:, 0] - np.nanmin(pose_3d[:, 0])) / 
                    (np.nanmax(pose_3d[:, 0]) - np.nanmin(pose_3d[:, 0]) + 1e-8) * 
                    (image_size[0] * 0.6) + image_size[0] * 0.2).astype(int)
            y_2d = ((pose_3d[:, 1] - np.nanmin(pose_3d[:, 1])) / 
                    (np.nanmax(pose_3d[:, 1]) - np.nanmin(pose_3d[:, 1]) + 1e-8) * 
                    (image_size[1] * 0.6) + image_size[1] * 0.2).astype(int)
            
            # Draw simple stick figure
            for joint_idx in range(17):
                if not np.isnan(pose_3d[joint_idx, 0]):
                    cv2.circle(frame, (x_2d[joint_idx], y_2d[joint_idx]), 8, (255, 255, 255), -1)
                    cv2.circle(frame, (x_2d[joint_idx], y_2d[joint_idx]), 8, (0, 0, 0), 2)
        
        frames.append(frame)
    
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, 
                       help='Specific sequence name (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--num-frames', type=int, default=50, help='Number of frames to process')
    parser.add_argument('--save-video', action='store_true', help='Save animation as video')
    parser.add_argument('--real-time', action='store_true', help='Run in real-time mode')
    args = parser.parse_args()
    
    print(f"Initializing MediaPipe 3D pose estimator for sequence: {args.sequence_name}")
    estimator = MediaPipe3DPoseEstimator()
    
    try:
        print("Loading 3D test data from dataset...")
        gt_poses_3d, seq_name = load_test_3d_data_from_dataset(args)
        
        if gt_poses_3d is None:
            print("No 3D ground truth data loaded. Please check your dataset.")
            return
        
        print(f"✓ Loaded {gt_poses_3d.shape[1]} 3D ground truth frames for sequence {seq_name}")
        
        # Debug: Print some statistics
        print(f"3D Ground truth pose stats:")
        print(f"  Shape: {gt_poses_3d.shape}")
        print(f"  X range: {np.min(gt_poses_3d[:, :, 0]):.1f} to {np.max(gt_poses_3d[:, :, 0]):.1f} mm")
        print(f"  Y range: {np.min(gt_poses_3d[:, :, 1]):.1f} to {np.max(gt_poses_3d[:, :, 1]):.1f} mm")
        print(f"  Z range: {np.min(gt_poses_3d[:, :, 2]):.1f} to {np.max(gt_poses_3d[:, :, 2]):.1f} mm")
        
        # Load actual video frames
        print("Loading video frames...")
        frames = load_mpi_test_frames(seq_name, args.num_frames)
        
        if frames is None:
            print("Could not load video frames. Creating dummy frames...")
            frames = create_dummy_frames_3d(gt_poses_3d, args.num_frames)
        
        print(f"✓ Using {len(frames)} frames for MediaPipe 3D processing")
        
        # Ensure matching frame counts
        min_frames = min(len(frames), gt_poses_3d.shape[1])
        frames = frames[:min_frames]
        gt_poses_3d = gt_poses_3d[:, :min_frames, :]
        
        # Create 3D visualization
        update_func, fig = create_3d_pose_visualization(estimator, frames, gt_poses_3d, seq_name, args)
        
        if args.real_time:
            print("Starting real-time 3D processing...")
            for i in range(min_frames):
                update_func(i)
                plt.pause(0.2)
                if i == 0:
                    plt.show(block=False)
            
            input("Press Enter to close...")
        else:
            print("Creating 3D animation...")
            ani = FuncAnimation(fig, update_func, frames=min_frames, 
                              interval=500, repeat=True, blit=False)
            
            if args.save_video:
                output_path = f'../3d_pose_comparison_{seq_name.lower()}_frames_{min_frames}.gif'
                print(f"Saving 3D animation to: {output_path}")
                ani.save(output_path, writer='pillow', fps=2, dpi=100)
                print(f"✓ 3D Animation saved successfully to: {output_path}")
            
            plt.show()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.close()
        print("MediaPipe 3D estimator closed.")

if __name__ == '__main__':
    main()