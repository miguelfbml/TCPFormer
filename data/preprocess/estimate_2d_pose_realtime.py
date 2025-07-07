'''
cd data/preprocess

# Run with specific sequence
python estimate_2d_pose_realtime.py --sequence-name TS1 --num-frames 50

# Save as video
python estimate_2d_pose_realtime.py --sequence-name TS2 --save-video --num-frames 30

# Real-time mode
python estimate_2d_pose_realtime.py --sequence-name TS3 --real-time --num-frames 20
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
            model_complexity=2,  # Heavy model (0=lite, 1=full, 2=heavy)
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe to H36M/MPI mapping
        self.mp_to_h36m_mapping = {
            # MediaPipe landmark index -> H36M joint index
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
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_image)
        
        # Initialize pose array (17 joints, 3 coordinates: x, y, confidence)
        pose_2d = np.zeros((17, 3))
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Map MediaPipe landmarks to H36M format
            for mp_idx, h36m_idx in self.mp_to_h36m_mapping.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    pose_2d[h36m_idx] = [
                        landmark.x,  # Normalized x coordinate
                        landmark.y,  # Normalized y coordinate
                        landmark.visibility  # Confidence/visibility score
                    ]
        
        return pose_2d, results
    
    def close(self):
        self.pose.close()

def load_test_data_from_dataset(args):
    """Load test data from MPI-INF-3DHP dataset exactly like compare_gt_pred.py"""
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

    # Use same parameters as your model evaluation
    dataset_args = DatasetArgs(
        data_root='../motion3d/', 
        n_frames=27,  # Same as your model's n_frames
        stride=9,     # Same as your model's stride
        flip=False,
        test_augmentation=False,
        data_augmentation=False,
        reverse_augmentation=False,
        out_all=1,
        test_batch_size=1
    )
    
    dataset = Fusion(dataset_args, train=False)  # Use Fusion like compare_gt_pred.py
    
    print(f"Dataset length: {len(dataset)}")
    
    if args.sequence_number >= len(dataset):
        print(f"ERROR: Sequence {args.sequence_number} is out of range! Dataset has {len(dataset)} sequences.")
        return None, None, None

    # Get multiple samples to build sequences
    sequence_data = []
    input_2d_data = []
    target_seq_name = None
    processed_samples = 0
    
    # Available sequence names in MPI-INF-3DHP test set
    available_sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    # If sequence_name is provided, use it; otherwise use sequence_number
    if args.sequence_name:
        target_seq_name = args.sequence_name
    else:
        target_seq_name = available_sequences[args.sequence_number % len(available_sequences)]
    
    print(f"Looking for sequence: {target_seq_name}")
    
    for i in range(len(dataset)):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            
            # Extract sequence name properly
            if isinstance(seq, (list, tuple)):
                current_seq_name = seq[0]
            elif isinstance(seq, torch.Tensor):
                current_seq_name = seq.item() if seq.numel() == 1 else str(seq)
            else:
                current_seq_name = str(seq)
            
            # Filter by target sequence
            if current_seq_name != target_seq_name:
                continue
            
            print(f"Found matching sequence: {current_seq_name} (sample {i})")
            
            # Process exactly like compare_gt_pred.py
            if isinstance(gt_3D, torch.Tensor):
                gt_3D = gt_3D.clone()
            else:
                gt_3D = torch.tensor(gt_3D)
                
            gt_3D = gt_3D.view(1, -1, 17, 3)  # N=1, T, 17, 3
            gt_3D[:, :, 14] = 0  # Set root joint to 0
            
            # Extract center frame (same as evaluation)
            center_frame = gt_3D.shape[1] // 2
            center_pose = gt_3D[0, center_frame]  # (17, 3)
            
            # Make root-relative (same as evaluation)
            center_pose = center_pose - center_pose[14:15, :]
            
            # Convert to numpy
            if hasattr(center_pose, 'cpu'):
                center_pose = center_pose.cpu().numpy()
            elif hasattr(center_pose, 'numpy'):
                center_pose = center_pose.numpy()
            else:
                center_pose = np.array(center_pose)
            
            # Also get the 2D input data
            if isinstance(input_2D, torch.Tensor):
                input_2D_np = input_2D.clone()
            else:
                input_2D_np = torch.tensor(input_2D)
            
            # Handle different input_2D shapes
            if input_2D_np.ndim == 4:  # (1, T, 17, 3) or (1, T, 17, 2)
                input_center_frame = input_2D_np.shape[1] // 2
                input_2d_pose = input_2D_np[0, input_center_frame].cpu().numpy()  # (17, 2/3)
            elif input_2D_np.ndim == 3:  # (T, 17, 3) or (T, 17, 2)
                input_center_frame = input_2D_np.shape[0] // 2
                input_2d_pose = input_2D_np[input_center_frame].cpu().numpy()  # (17, 2/3)
            else:
                input_2d_pose = input_2D_np.cpu().numpy()
            
            # Ensure we have at least x, y coordinates
            if input_2d_pose.shape[1] < 2:
                print(f"Skipping sample {i}: insufficient 2D coordinates")
                continue
            
            # Add confidence column if not present
            if input_2d_pose.shape[1] == 2:
                confidence = np.ones((input_2d_pose.shape[0], 1))
                input_2d_pose = np.hstack([input_2d_pose, confidence])
            
            sequence_data.append(center_pose)
            input_2d_data.append(input_2d_pose)
            processed_samples += 1
            
            # Limit the number of samples for visualization
            if processed_samples >= args.num_frames:
                break
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Processed {processed_samples} samples for sequence {target_seq_name}")
    
    if not sequence_data:
        print("No valid ground truth data found")
        return None, None, None
    
    # Stack frames: (17, T, 3) for 3D, (17, T, 3) for 2D
    sequence_3d = np.stack(sequence_data, axis=1)
    sequence_2d = np.stack(input_2d_data, axis=1)  # (17, T, 3)
    
    print(f"Stacked 3D sequence shape: {sequence_3d.shape}")
    print(f"Stacked 2D sequence shape: {sequence_2d.shape}")
    
    # Apply camera transformation to 3D data
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    print(f"Ground truth sequence: {target_seq_name}, 3D shape: {sequence_3d.shape}, 2D shape: {sequence_2d.shape}")
    return sequence_3d, sequence_2d, target_seq_name

def create_dummy_images_from_2d_poses(poses_2d, image_size=(640, 480)):
    """Create dummy images with 2D pose keypoints drawn on them"""
    images = []
    
    for t in range(poses_2d.shape[1]):  # For each time frame
        # Create a blank image
        image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Get 2D pose for this frame
        pose_2d = poses_2d[:, t, :]  # (17, 3)
        
        # Draw keypoints on the image
        for joint_idx in range(17):
            if pose_2d[joint_idx, 2] > 0.5:  # If confidence > 0.5
                # Convert normalized coordinates to pixel coordinates
                x = int(pose_2d[joint_idx, 0] * image_size[0])
                y = int(pose_2d[joint_idx, 1] * image_size[1])
                
                # Draw keypoint
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                
                # Add joint index as text
                cv2.putText(image, str(joint_idx), (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw skeleton connections
        connections = [
            (10, 9), (9, 8), (8, 11), (8, 14), (14, 15), (15, 16),
            (11, 12), (12, 13), (8, 7), (7, 0), (0, 4), (0, 1),
            (1, 2), (2, 3), (4, 5), (5, 6)
        ]
        
        for connection in connections:
            joint1, joint2 = connection
            if (pose_2d[joint1, 2] > 0.5 and pose_2d[joint2, 2] > 0.5):
                x1 = int(pose_2d[joint1, 0] * image_size[0])
                y1 = int(pose_2d[joint1, 1] * image_size[1])
                x2 = int(pose_2d[joint2, 0] * image_size[0])
                y2 = int(pose_2d[joint2, 1] * image_size[1])
                
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        images.append(image)
    
    return images

def create_realtime_visualization(estimator, images, gt_poses_2d, seq_name, args):
    """Create real-time visualization comparing MediaPipe estimates with ground truth"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Real-time 2D Pose Estimation - {seq_name}', fontsize=16)
    
    # Initialize pose storage
    estimated_poses = []
    processing_times = []
    
    def update(frame_idx):
        if frame_idx >= len(images):
            return
        
        image = images[frame_idx]
        gt_pose_2d = gt_poses_2d[:, frame_idx, :]  # (17, 3)
        
        # Measure processing time
        start_time = time.time()
        estimated_pose, mp_results = estimator.estimate_pose_from_image(image)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        estimated_poses.append(estimated_pose)
        
        # Clear all axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        # Plot 1: Original image with MediaPipe overlay
        ax1.set_title('Ground Truth 2D Pose + MediaPipe Detection')
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if mp_results.pose_landmarks:
            # Draw MediaPipe pose landmarks
            annotated_image = image.copy()
            estimator.mp_drawing.draw_landmarks(
                annotated_image, mp_results.pose_landmarks, estimator.mp_pose.POSE_CONNECTIONS)
            ax1.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        ax1.axis('off')
        
        # Plot 2: 2D pose comparison (scatter plot)
        ax2.set_title('2D Pose Comparison')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.invert_yaxis()  # Invert y-axis for image coordinates
        
        # Plot ground truth (blue)
        valid_gt = gt_pose_2d[:, 2] > 0.5  # Filter by confidence
        ax2.scatter(gt_pose_2d[valid_gt, 0], gt_pose_2d[valid_gt, 1], 
                   c='blue', s=50, alpha=0.7, label='Ground Truth')
        
        # Plot MediaPipe estimates (red)
        valid_est = estimated_pose[:, 2] > 0.5
        ax2.scatter(estimated_pose[valid_est, 0], estimated_pose[valid_est, 1], 
                   c='red', s=50, alpha=0.7, label='MediaPipe')
        
        ax2.legend()
        ax2.set_xlabel('X (normalized)')
        ax2.set_ylabel('Y (normalized)')
        
        # Plot 3: Processing time graph
        ax3.set_title('Processing Time')
        if len(processing_times) > 1:
            ax3.plot(processing_times, 'g-', linewidth=2)
            ax3.axhline(y=np.mean(processing_times), color='r', linestyle='--', 
                       label=f'Avg: {np.mean(processing_times):.3f}s')
            ax3.legend()
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True)
        
        # Plot 4: Error analysis
        ax4.set_title('Joint-wise Error Analysis')
        if len(estimated_poses) > 0:
            # Calculate per-joint errors
            errors = []
            for i in range(17):  # 17 joints
                if gt_pose_2d[i, 2] > 0.5 and estimated_pose[i, 2] > 0.5:
                    error = np.linalg.norm(gt_pose_2d[i, :2] - estimated_pose[i, :2])
                    errors.append(error)
                else:
                    errors.append(0)
            
            joint_names = ['Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
                          'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
                          'RShoulder', 'RElbow', 'RWrist']
            
            bars = ax4.bar(range(17), errors, color='orange', alpha=0.7)
            ax4.set_xticks(range(17))
            ax4.set_xticklabels(joint_names, rotation=45, ha='right')
            ax4.set_ylabel('Error (normalized units)')
            
            # Add error values on bars
            for i, (bar, error) in enumerate(zip(bars, errors)):
                if error > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{error:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Update frame info
        fps = 1.0 / processing_time if processing_time > 0 else 0
        fig.suptitle(f'Real-time 2D Pose Estimation - {seq_name}\n'
                    f'Frame: {frame_idx+1}/{len(images)} | '
                    f'Processing Time: {processing_time:.3f}s | '
                    f'FPS: {fps:.1f}', fontsize=14)
        
        plt.tight_layout()
    
    return update
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
        
        # Create the figure first
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Real-time 2D Pose Estimation - {seq_name}', fontsize=16)
        
        # Create visualization function
        update_func = create_realtime_visualization_fixed(estimator, images, gt_poses_2d, seq_name, args, fig, ax1, ax2, ax3, ax4)
        
        if args.real_time:
            # Real-time processing
            print("Starting real-time processing...")
            for i in range(len(images)):
                update_func(i)
                plt.pause(0.033)  # ~30 FPS
                if i == 0:
                    plt.show(block=False)
            
            input("Press Enter to close...")
            
        else:
            # Create animation
            print("Creating animation...")
            ani = FuncAnimation(fig, update_func, frames=len(images), 
                              interval=200, repeat=True, blit=False)
            
            if args.save_video:
                output_path = f'../realtime_2d_pose_{seq_name.lower()}.gif'
                print(f"Saving animation to: {output_path}")
                
                # Use better writer settings for GIF
                writer = 'pillow'
                ani.save(output_path, writer=writer, fps=5, dpi=80, bitrate=1800)
                print(f"Animation saved successfully to: {output_path}")
                
                # Also save as MP4 for better compatibility
                output_mp4 = output_path.replace('.gif', '.mp4')
                try:
                    ani.save(output_mp4, writer='ffmpeg', fps=10, dpi=100, bitrate=1800)
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

def create_realtime_visualization_fixed(estimator, images, gt_poses_2d, seq_name, args, fig, ax1, ax2, ax3, ax4):
    """Create real-time visualization with fixed figure references"""
    
    # Initialize pose storage
    estimated_poses = []
    processing_times = []
    
    def update(frame_idx):
        if frame_idx >= len(images):
            return
        
        image = images[frame_idx]
        gt_pose_2d = gt_poses_2d[:, frame_idx, :]  # (17, 3)
        
        # Measure processing time
        start_time = time.time()
        estimated_pose, mp_results = estimator.estimate_pose_from_image(image)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        estimated_poses.append(estimated_pose)
        
        # Clear all axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        # Plot 1: Original image with MediaPipe overlay
        ax1.set_title('Ground Truth 2D Pose + MediaPipe Detection', fontsize=10)
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if mp_results.pose_landmarks:
            # Draw MediaPipe pose landmarks
            annotated_image = image.copy()
            estimator.mp_drawing.draw_landmarks(
                annotated_image, mp_results.pose_landmarks, estimator.mp_pose.POSE_CONNECTIONS)
            ax1.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        ax1.axis('off')
        
        # Plot 2: 2D pose comparison (scatter plot)
        ax2.set_title('2D Pose Comparison', fontsize=10)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.invert_yaxis()  # Invert y-axis for image coordinates
        
        # Plot ground truth (blue)
        valid_gt = gt_pose_2d[:, 2] > 0.1  # Lower threshold for better visibility
        if np.any(valid_gt):
            ax2.scatter(gt_pose_2d[valid_gt, 0], gt_pose_2d[valid_gt, 1], 
                       c='blue', s=30, alpha=0.7, label='Ground Truth')
        
        # Plot MediaPipe estimates (red)
        valid_est = estimated_pose[:, 2] > 0.1
        if np.any(valid_est):
            ax2.scatter(estimated_pose[valid_est, 0], estimated_pose[valid_est, 1], 
                       c='red', s=30, alpha=0.7, label='MediaPipe')
        
        ax2.legend(fontsize=8)
        ax2.set_xlabel('X (normalized)', fontsize=8)
        ax2.set_ylabel('Y (normalized)', fontsize=8)
        ax2.tick_params(labelsize=8)
        
        # Plot 3: Processing time graph
        ax3.set_title('Processing Time', fontsize=10)
        if len(processing_times) > 0:
            ax3.plot(processing_times, 'g-', linewidth=2)
            if len(processing_times) > 1:
                ax3.axhline(y=np.mean(processing_times), color='r', linestyle='--', 
                           label=f'Avg: {np.mean(processing_times):.3f}s')
                ax3.legend(fontsize=8)
        ax3.set_xlabel('Frame', fontsize=8)
        ax3.set_ylabel('Time (seconds)', fontsize=8)
        ax3.tick_params(labelsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error analysis
        ax4.set_title('Joint-wise Error Analysis', fontsize=10)
        if len(estimated_poses) > 0:
            # Calculate per-joint errors
            errors = []
            for i in range(17):  # 17 joints
                if gt_pose_2d[i, 2] > 0.1 and estimated_pose[i, 2] > 0.1:
                    error = np.linalg.norm(gt_pose_2d[i, :2] - estimated_pose[i, :2])
                    errors.append(error)
                else:
                    errors.append(0)
            
            joint_names = ['Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
                          'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
                          'RShoulder', 'RElbow', 'RWrist']
            
            bars = ax4.bar(range(17), errors, color='orange', alpha=0.7)
            ax4.set_xticks(range(17))
            ax4.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=7)
            ax4.set_ylabel('Error (normalized units)', fontsize=8)
            ax4.tick_params(labelsize=8)
            
            # Add error values on bars (only for non-zero errors)
            for i, (bar, error) in enumerate(zip(bars, errors)):
                if error > 0.01:  # Only show significant errors
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{error:.2f}', ha='center', va='bottom', fontsize=6)
        
        # Update frame info
        fps = 1.0 / processing_time if processing_time > 0 else 0
        fig.suptitle(f'Real-time 2D Pose Estimation - {seq_name}\n'
                    f'Frame: {frame_idx+1}/{len(images)} | '
                    f'Processing Time: {processing_time:.3f}s | '
                    f'FPS: {fps:.1f}', fontsize=12)
        
        plt.tight_layout()
        
        # Return the artists for blitting (optional)
        return [ax1, ax2, ax3, ax4]
    
    return update

if __name__ == '__main__':
    main()