'''
cd data/preprocess

# Create simple 2-panel comparison
python estimate_2d_pose_realtime.py --sequence-name TS1 --num-frames 30

# Save as GIF
python estimate_2d_pose_realtime.py --sequence-name TS1 --save-video --num-frames 20

# Real-time mode
python estimate_2d_pose_realtime.py --sequence-name TS2 --real-time --num-frames 15
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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from data.reader.motion_dataset import MPI3DHP, Fusion
from data.const import H36M_TO_MPI

class MediaPipe2DPoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe to H36M/MPI mapping
        self.mp_to_h36m_mapping = {
            0: 9,   # nose -> nose
            11: 11, # left_shoulder -> left shoulder
            12: 8,  # right_shoulder -> right shoulder  
            13: 12, # left_elbow -> left elbow
            14: 9,  # right_elbow -> right elbow
            15: 13, # left_wrist -> left wrist
            16: 10, # right_wrist -> right wrist
            23: 4,  # left_hip -> left hip
            24: 1,  # right_hip -> right hip
            25: 5,  # left_knee -> left knee
            26: 2,  # right_knee -> right knee
            27: 6,  # left_ankle -> left ankle
            28: 3,  # right_ankle -> right ankle
        }
        
    def estimate_pose_from_image(self, image):
        """Estimate 2D pose from a single image using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        pose_2d = np.zeros((17, 3))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for mp_idx, h36m_idx in self.mp_to_h36m_mapping.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    pose_2d[h36m_idx] = [
                        landmark.x,
                        landmark.y,
                        landmark.visibility
                    ]
        
        return pose_2d, results
    
    def close(self):
        self.pose.close()

def load_test_data_from_dataset(args):
    """Load test data from MPI-INF-3DHP dataset"""
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
    
    sequence_data = []
    input_2d_data = []
    target_seq_name = None
    processed_samples = 0
    
    available_sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    if args.sequence_name:
        target_seq_name = args.sequence_name
    else:
        target_seq_name = available_sequences[args.sequence_number % len(available_sequences)]
    
    print(f"Looking for sequence: {target_seq_name}")
    
    for i in range(len(dataset)):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            
            if isinstance(seq, (list, tuple)):
                current_seq_name = seq[0]
            elif isinstance(seq, torch.Tensor):
                current_seq_name = seq.item() if seq.numel() == 1 else str(seq)
            else:
                current_seq_name = str(seq)
            
            if current_seq_name != target_seq_name:
                continue
            
            # Process 3D GT
            if isinstance(gt_3D, torch.Tensor):
                gt_3D = gt_3D.clone()
            else:
                gt_3D = torch.tensor(gt_3D)
                
            gt_3D = gt_3D.view(1, -1, 17, 3)
            gt_3D[:, :, 14] = 0
            
            center_frame = gt_3D.shape[1] // 2
            center_pose = gt_3D[0, center_frame]
            center_pose = center_pose - center_pose[14:15, :]
            
            if hasattr(center_pose, 'cpu'):
                center_pose = center_pose.cpu().numpy()
            elif hasattr(center_pose, 'numpy'):
                center_pose = center_pose.numpy()
            else:
                center_pose = np.array(center_pose)
            
            # Process 2D input
            if isinstance(input_2D, torch.Tensor):
                input_2D_np = input_2D.clone()
            else:
                input_2D_np = torch.tensor(input_2D)
            
            if input_2D_np.ndim == 4:
                input_center_frame = input_2D_np.shape[1] // 2
                input_2d_pose = input_2D_np[0, input_center_frame].cpu().numpy()
            elif input_2D_np.ndim == 3:
                input_center_frame = input_2D_np.shape[0] // 2
                input_2d_pose = input_2D_np[input_center_frame].cpu().numpy()
            else:
                input_2d_pose = input_2D_np.cpu().numpy()
            
            if input_2d_pose.shape[1] < 2:
                continue
            
            if input_2d_pose.shape[1] == 2:
                confidence = np.ones((input_2d_pose.shape[0], 1))
                input_2d_pose = np.hstack([input_2d_pose, confidence])
            
            sequence_data.append(center_pose)
            input_2d_data.append(input_2d_pose)
            processed_samples += 1
            
            if processed_samples >= args.num_frames:
                break
                
        except Exception as e:
            continue
    
    if not sequence_data:
        return None, None, None
    
    sequence_3d = np.stack(sequence_data, axis=1)
    sequence_2d = np.stack(input_2d_data, axis=1)
    
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    return sequence_3d, sequence_2d, target_seq_name

def create_dummy_images_from_2d_poses(poses_2d, image_size=(640, 480)):
    """Create dummy images with 2D pose keypoints drawn on them"""
    images = []
    
    for t in range(poses_2d.shape[1]):
        image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        pose_2d = poses_2d[:, t, :]
        
        # Draw keypoints
        for joint_idx in range(17):
            if pose_2d[joint_idx, 2] > 0.1:
                x = int(pose_2d[joint_idx, 0] * image_size[0])
                y = int(pose_2d[joint_idx, 1] * image_size[1])
                cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
        
        # Draw skeleton connections
        connections = [
            (10, 9), (9, 8), (8, 11), (8, 14), (14, 15), (15, 16),
            (11, 12), (12, 13), (8, 7), (7, 0), (0, 4), (0, 1),
            (1, 2), (2, 3), (4, 5), (5, 6)
        ]
        
        for connection in connections:
            joint1, joint2 = connection
            if (pose_2d[joint1, 2] > 0.1 and pose_2d[joint2, 2] > 0.1):
                x1 = int(pose_2d[joint1, 0] * image_size[0])
                y1 = int(pose_2d[joint1, 1] * image_size[1])
                x2 = int(pose_2d[joint2, 0] * image_size[0])
                y2 = int(pose_2d[joint2, 1] * image_size[1])
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        images.append(image)
    
    return images

def create_simple_visualization(estimator, images, gt_poses_2d, seq_name, args):
    """Create simple visualization with just GT and MediaPipe predictions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'2D Pose Comparison - {seq_name}', fontsize=16)
    
    estimated_poses = []
    
    def update(frame_idx):
        if frame_idx >= len(images):
            return
        
        image = images[frame_idx]
        gt_pose_2d = gt_poses_2d[:, frame_idx, :]
        
        # Get MediaPipe estimation
        estimated_pose, mp_results = estimator.estimate_pose_from_image(image)
        estimated_poses.append(estimated_pose)
        
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Ground Truth 2D Pose
        ax1.set_title('Ground Truth 2D Pose', fontsize=14)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.invert_yaxis()
        
        # Plot GT keypoints
        valid_gt = gt_pose_2d[:, 2] > 0.1
        if np.any(valid_gt):
            ax1.scatter(gt_pose_2d[valid_gt, 0], gt_pose_2d[valid_gt, 1], 
                       c='blue', s=80, alpha=0.8, edgecolors='darkblue', linewidth=2)
        
        # Draw GT skeleton connections
        connections = [
            (10, 9), (9, 8), (8, 11), (8, 14), (14, 15), (15, 16),
            (11, 12), (12, 13), (8, 7), (7, 0), (0, 4), (0, 1),
            (1, 2), (2, 3), (4, 5), (5, 6)
        ]
        
        for connection in connections:
            joint1, joint2 = connection
            if (gt_pose_2d[joint1, 2] > 0.1 and gt_pose_2d[joint2, 2] > 0.1):
                ax1.plot([gt_pose_2d[joint1, 0], gt_pose_2d[joint2, 0]], 
                        [gt_pose_2d[joint1, 1], gt_pose_2d[joint2, 1]], 
                        'b-', linewidth=3, alpha=0.7)
        
        # Add joint labels
        joint_names = ['Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
                      'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
                      'RShoulder', 'RElbow', 'RWrist']
        
        for i, (valid, name) in enumerate(zip(valid_gt, joint_names)):
            if valid:
                ax1.annotate(f'{i}', (gt_pose_2d[i, 0], gt_pose_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, color='white', weight='bold')
        
        ax1.set_xlabel('X (normalized)', fontsize=12)
        ax1.set_ylabel('Y (normalized)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MediaPipe Predictions
        ax2.set_title('MediaPipe 2D Pose Predictions', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.invert_yaxis()
        
        # Plot MediaPipe keypoints
        valid_est = estimated_pose[:, 2] > 0.1
        if np.any(valid_est):
            ax2.scatter(estimated_pose[valid_est, 0], estimated_pose[valid_est, 1], 
                       c='red', s=80, alpha=0.8, edgecolors='darkred', linewidth=2)
        
        # Draw MediaPipe skeleton connections
        for connection in connections:
            joint1, joint2 = connection
            if (estimated_pose[joint1, 2] > 0.1 and estimated_pose[joint2, 2] > 0.1):
                ax2.plot([estimated_pose[joint1, 0], estimated_pose[joint2, 0]], 
                        [estimated_pose[joint1, 1], estimated_pose[joint2, 1]], 
                        'r-', linewidth=3, alpha=0.7)
        
        # Add joint labels
        for i, (valid, name) in enumerate(zip(valid_est, joint_names)):
            if valid:
                ax2.annotate(f'{i}', (estimated_pose[i, 0], estimated_pose[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, color='white', weight='bold')
        
        ax2.set_xlabel('X (normalized)', fontsize=12)
        ax2.set_ylabel('Y (normalized)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Update frame info
        fig.suptitle(f'2D Pose Comparison - {seq_name}\nFrame: {frame_idx+1}/{len(images)}', fontsize=16)
        
        plt.tight_layout()
        
        return [ax1, ax2]
    
    return update, fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, 
                       help='Specific sequence name (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--num-frames', type=int, default=50, help='Number of frames to process')
    parser.add_argument('--save-video', action='store_true', help='Save animation as video')
    parser.add_argument('--real-time', action='store_true', help='Run in real-time mode')
    args = parser.parse_args()
    
    print("Initializing MediaPipe pose estimator...")
    estimator = MediaPipe2DPoseEstimator()
    
    try:
        print("Loading test data from dataset...")
        gt_poses_3d, gt_poses_2d, seq_name = load_test_data_from_dataset(args)
        
        if gt_poses_3d is None or gt_poses_2d is None:
            print("No data loaded. Please check your dataset.")
            return
        
        print(f"Loaded {gt_poses_2d.shape[1]} frames for sequence {seq_name}")
        
        # Create dummy images from 2D poses
        print("Creating dummy images from 2D poses...")
        images = create_dummy_images_from_2d_poses(gt_poses_2d)
        
        # Create visualization
        update_func, fig = create_simple_visualization(estimator, images, gt_poses_2d, seq_name, args)
        
        if args.real_time:
            # Real-time processing
            print("Starting real-time processing...")
            for i in range(len(images)):
                update_func(i)
                plt.pause(0.1)
                if i == 0:
                    plt.show(block=False)
            
            input("Press Enter to close...")
            
        else:
            # Create animation
            print("Creating animation...")
            ani = FuncAnimation(fig, update_func, frames=len(images), 
                              interval=300, repeat=True, blit=False)
            
            if args.save_video:
                output_path = f'../simple_2d_pose_{seq_name.lower()}.gif'
                print(f"Saving animation to: {output_path}")
                
                ani.save(output_path, writer='pillow', fps=3, dpi=100)
                print(f"Animation saved successfully to: {output_path}")
                
                # Also save as MP4
                output_mp4 = output_path.replace('.gif', '.mp4')
                try:
                    ani.save(output_mp4, writer='ffmpeg', fps=5, dpi=100)
                    print(f"MP4 version saved to: {output_mp4}")
                except:
                    print("FFmpeg not available, MP4 not saved")
            
            plt.show()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.close()
        print("MediaPipe estimator closed.")

if __name__ == '__main__':
    main()