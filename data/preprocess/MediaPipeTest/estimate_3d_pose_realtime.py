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

# MPI-INF-3DHP skeleton connections
connections = [
    (10, 9), (9, 8), (8, 11), (8, 14), (14, 15), (15, 16),
    (11, 12), (12, 13), (8, 7), (7, 0), (0, 4), (0, 1),
    (1, 2), (2, 3), (4, 5), (5, 6)
]

def convert_h36m_to_mpi_connection():
    """Convert connections to MPI-INF-3DHP joint mapping."""
    new_connections = []
    for connection in connections:
        new_connection = (H36M_TO_MPI[connection[0]], H36M_TO_MPI[connection[1]])
        new_connections.append(new_connection)
    return new_connections

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
        
        # MediaPipe to MPI joint mapping (aligned with MPI-INF-3DHP)
        self.mp_to_mpi_mapping = {
            0: 10,   # nose -> head
            7: 9,    # left_ear -> nose
            8: 9,    # right_ear -> nose
            11: 11,  # left_shoulder -> left shoulder
            12: 14,  # right_shoulder -> right shoulder
            13: 12,  # left_elbow -> left elbow
            14: 15,  # right_elbow -> right elbow
            15: 13,  # left_wrist -> left wrist
            16: 16,  # right_wrist -> right wrist
            23: 4,   # left_hip -> left hip
            24: 1,   # right_hip -> right hip
            25: 5,   # left_knee -> left knee
            26: 2,   # right_knee -> right knee
            27: 6,   # left_ankle -> left ankle
            28: 3,   # right_ankle -> right ankle
        }
        
        # Estimation for missing joints
        self.missing_joints_estimation = {
            0: [4, 1],      # root: average of hips
            7: [11, 14],    # spine: average of shoulders
            8: [11, 14],    # thorax: average of shoulders
        }

    def estimate_3d_pose_from_image(self, image):
        """Estimate 3D pose from a single image using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        pose_3d = np.zeros((17, 3))  # x, y, z
        
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # Map MediaPipe landmarks to MPI joints
            visibility = np.zeros(17)
            for mp_idx, mpi_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    pose_3d[mpi_idx] = [
                        landmark.x * 1000,  # Convert meters to mm
                        landmark.y * 1000,
                        landmark.z * 1000
                    ]
                    visibility[mpi_idx] = landmark.visibility
            
            # Estimate missing joints
            for missing_joint, source_joints in self.missing_joints_estimation.items():
                valid_sources = [j for j in source_joints if visibility[j] > 0.1]
                if valid_sources:
                    pose_3d[missing_joint] = np.mean([pose_3d[j] for j in valid_sources], axis=0)
                    visibility[missing_joint] = np.mean([visibility[j] for j in valid_sources])
            
            # Make root-relative (MPI joint 14)
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
                frame_idx = int(os.path.basename(img_path).split('_')[-1].split('.')[0])
                frame_indices.append(frame_idx)
            except:
                frame_indices.append(len(frame_indices) + 1)
    
    return frames, frame_indices

def load_test_3d_data_from_dataset(args):
    """Load 3D ground truth data from MPI-INF-3DHP dataset with frame rate alignment"""
    data_path = os.path.join('../motion3d/', 'data_test_3dhp.npz')
    data = np.load(data_path, allow_pickle=True)['data'].item()
    
    target_seq_name = args.sequence_name or ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6'][args.sequence_number % 6]
    
    if target_seq_name not in data:
        print(f"Sequence {target_seq_name} not found in test data.")
        return None, None, None
    
    # Load ground truth poses for the target sequence
    anim = data[target_seq_name]
    data_3d = anim['data_3d']  # Shape: (T, 17, 3)
    
    # Make root-relative (MPI joint 14)
    data_3d[:, :14] -= data_3d[:, 14:15]
    data_3d[:, 15:] -= data_3d[:, 14:15]
    
    # Subsample to match the number of video frames
    total_frames = data_3d.shape[0]
    print(f"Ground truth sequence {target_seq_name} has {total_frames} frames")
    stride = max(1, total_frames // args.num_frames) if total_frames > args.num_frames else 1
    print(f"Using stride {stride} to align with {args.num_frames} video frames")
    
    sequence_data = []
    frame_indices = []
    frame_counter = 1
    
    for frame_idx in range(0, total_frames, stride):
        pose = data_3d[frame_idx]
        sequence_data.append(pose)
        frame_indices.append(frame_counter)
        frame_counter += 1
        if len(sequence_data) >= args.num_frames:
            break
    
    if not sequence_data:
        print(f"No ground truth poses loaded for sequence {target_seq_name}.")
        return None, None, None
    
    sequence_3d = np.stack(sequence_data, axis=1)  # (17, T, 3)
    
    # Apply camera transformation to match MPI-INF-3DHP
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    print(f"Loaded {sequence_3d.shape[1]} ground truth frames with indices: {frame_indices[:10]}...")
    return sequence_3d, target_seq_name, frame_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, help='Sequence name (TS1, TS2, etc.)')
    parser.add_argument('--num-frames', type=int, default=50, help='Number of frames to process')
    parser.add_argument('--save-video', action='store_true', help='Save animation as GIF')
    parser.add_argument('--frame-start', type=int, default=0, help='Starting frame for comparison')
    args = parser.parse_args()
    
    # Get connections for ground truth
    gt_connections = convert_h36m_to_mpi_connection()
    
    # Define MediaPipe connections that directly correspond to ground truth
    mp_connections = [
        (10, 9),    # head -> nose
        (9, 8),     # nose -> thorax
        (8, 11),    # thorax -> left shoulder
        (8, 14),    # thorax -> right shoulder
        (11, 12),   # left shoulder -> left elbow
        (12, 13),   # left elbow -> left wrist
        (14, 15),   # right shoulder -> right elbow
        (15, 16),   # right elbow -> right wrist
        (8, 7),     # thorax -> spine
        (7, 0),     # spine -> root
        (0, 4),     # root -> left hip
        (0, 1),     # root -> right hip
        (4, 5),     # left hip -> left knee
        (5, 6),     # left knee -> left ankle
        (1, 2),     # right hip -> right knee
        (2, 3),     # right knee -> right ankle
    ]
    
    estimator = MediaPipe3DPoseEstimator()
    
    try:
        # Load ground truth data
        gt_poses_3d, seq_name, gt_frame_indices = load_test_3d_data_from_dataset(args)
        if gt_poses_3d is None:
            print("Failed to load ground truth data.")
            return
        
        # Load video frames
        frames, frame_indices = load_mpi_test_frames(seq_name, args.num_frames)
        if not frames:
            print("No video frames loaded. Exiting.")
            return
        
        print(f"Loaded {len(frames)} video frames with indices: {frame_indices[:10]}...")
        
        # Synchronize frames (mimic compare_gt_pred.py)
        num_frames = min(len(frames), gt_poses_3d.shape[1], args.num_frames)
        start_frame = args.frame_start
        end_frame = min(start_frame + args.num_frames, num_frames)
        frames = frames[start_frame:end_frame]
        gt_poses_3d = gt_poses_3d[:, start_frame:end_frame, :]
        synced_frame_indices = list(range(start_frame + 1, end_frame + 1))
        min_frames = len(synced_frame_indices)
        
        print(f"Synchronized {min_frames} frames with indices: {synced_frame_indices[:10]}...")
        
        if min_frames == 0:
            print("No synchronized frames available. Exiting.")
            return
        
        # Estimate 3D poses
        pred_poses_3d = []
        visibilities = []
        for frame in frames:
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