'''
cd data/preprocess

# Try to load real frames
python estimate_2d_pose_realtime.py --sequence-name TS1 --num-frames 30

# Save as GIF
python estimate_2d_pose_realtime.py --sequence-name TS1 --save-video --num-frames 20
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

def load_mpi_test_frames(sequence_name, num_frames=50):
    """Load actual video frames from MPI-INF-3DHP test set"""
    # MPI-INF-3DHP test set video paths - updated for your system
    mpi_roots = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp',
        '../motion3d/MPI-INF-3DHP',
        '../motion3d'
    ]
    
    # Try different possible paths for MPI-INF-3DHP test videos
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
        print("Video frames not found. Available paths checked:")
        for path in possible_paths[:10]:  # Show first 10 paths
            print(f"  - {path} (exists: {os.path.exists(path)})")
        
        # Try to find any image files in the expected directory structure
        for root in mpi_roots:
            if os.path.exists(root):
                print(f"\nExploring {root}:")
                try:
                    # List subdirectories
                    for item in os.listdir(root):
                        item_path = os.path.join(root, item)
                        if os.path.isdir(item_path) and sequence_name in item:
                            print(f"  Found related directory: {item_path}")
                            # Look for image subdirectories
                            for subitem in os.listdir(item_path):
                                subitem_path = os.path.join(item_path, subitem)
                                if os.path.isdir(subitem_path):
                                    print(f"    Subdirectory: {subitem_path}")
                                    # Check if it contains images
                                    image_files = glob.glob(os.path.join(subitem_path, '*.jpg'))
                                    image_files.extend(glob.glob(os.path.join(subitem_path, '*.png')))
                                    if image_files:
                                        print(f"      Contains {len(image_files)} images")
                                        video_path = subitem_path
                                        break
                            if video_path:
                                break
                except Exception as e:
                    print(f"  Error exploring {root}: {e}")
                
                if video_path:
                    break
        
        if video_path is None:
            return None
    
    print(f"Loading frames from: {video_path}")
    
    # Load frames from image sequence
    frames = []
    
    # Check for different image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        files = glob.glob(os.path.join(video_path, ext))
        files.extend(glob.glob(os.path.join(video_path, ext.upper())))
        image_files.extend(files)
    
    if image_files:
        image_files.sort()  # Sort to ensure correct order
        print(f"Found {len(image_files)} image files")
        
        # Load images
        for i, img_path in enumerate(image_files[:num_frames]):
            frame = cv2.imread(img_path)
            if frame is not None:
                frames.append(frame)
                if i % 50 == 0:  # Progress indicator
                    print(f"Loaded {i+1}/{min(num_frames, len(image_files))} images...")
            else:
                print(f"Warning: Could not load image {img_path}")
                
        print(f"Successfully loaded {len(frames)} frames")
        
    else:
        print("No image files found in the directory")
        print(f"Directory contents: {os.listdir(video_path) if os.path.exists(video_path) else 'Directory does not exist'}")
        return None
    
    return frames

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
    
    input_2d_data = []
    target_seq_name = None
    processed_samples = 0
    
    available_sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    # Fix sequence selection logic
    if args.sequence_name:
        if args.sequence_name not in available_sequences:
            print(f"Warning: {args.sequence_name} not in available sequences {available_sequences}")
            target_seq_name = available_sequences[0]
        else:
            target_seq_name = args.sequence_name
    else:
        target_seq_name = available_sequences[args.sequence_number % len(available_sequences)]
    
    print(f"Looking for sequence: {target_seq_name}")
    
    # First pass: find samples from the target sequence
    found_samples = []
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
            
            if current_seq_name == target_seq_name:
                found_samples.append(i)
                print(f"Found matching sequence: {target_seq_name} (sample {i})")
                if len(found_samples) >= args.num_frames:
                    break
                    
        except Exception as e:
            continue
    
    print(f"Found {len(found_samples)} samples for sequence {target_seq_name}")
    
    if not found_samples:
        print(f"No samples found for sequence {target_seq_name}")
        return None, None
    
    # Second pass: process the found samples
    for sample_idx in found_samples:
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[sample_idx]
            
            # Process 2D input
            if isinstance(input_2D, torch.Tensor):
                input_2D_np = input_2D.clone()
            else:
                input_2D_np = torch.tensor(input_2D)
            
            # Extract center frame for consistency with evaluation
            if input_2D_np.ndim == 4:  # (1, T, 17, 3) or (1, T, 17, 2)
                input_center_frame = input_2D_np.shape[1] // 2
                input_2d_pose = input_2D_np[0, input_center_frame].cpu().numpy()
            elif input_2D_np.ndim == 3:  # (T, 17, 3) or (T, 17, 2)
                input_center_frame = input_2D_np.shape[0] // 2
                input_2d_pose = input_2D_np[input_center_frame].cpu().numpy()
            else:  # (17, 3) or (17, 2)
                input_2d_pose = input_2D_np.cpu().numpy()
            
            # Ensure we have at least x, y coordinates
            if input_2d_pose.shape[1] < 2:
                continue
            
            # Add confidence if not present
            if input_2d_pose.shape[1] == 2:
                confidence = np.ones((input_2d_pose.shape[0], 1)) * 0.9  # High confidence for GT
                input_2d_pose = np.hstack([input_2d_pose, confidence])
            
            # Normalize 2D poses to [0, 1] range if they're not already
            if np.max(input_2d_pose[:, :2]) > 2.0:  # Likely in pixel coordinates
                # Assume image size (you might need to adjust this)
                image_width, image_height = 2048, 2048  # Common MPI-INF-3DHP size
                input_2d_pose[:, 0] /= image_width   # Normalize x
                input_2d_pose[:, 1] /= image_height  # Normalize y
                input_2d_pose[:, 0] = np.clip(input_2d_pose[:, 0], 0, 1)
                input_2d_pose[:, 1] = np.clip(input_2d_pose[:, 1], 0, 1)
            
            input_2d_data.append(input_2d_pose)
            processed_samples += 1
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    print(f"Successfully processed {processed_samples} samples for sequence {target_seq_name}")
    
    if not input_2d_data:
        return None, None
    
    # Stack frames: (17, T, 3)
    sequence_2d = np.stack(input_2d_data, axis=1)
    
    return sequence_2d, target_seq_name

def create_simple_visualization(estimator, frames, gt_poses_2d, seq_name, args):
    """Create simple visualization with just GT and MediaPipe predictions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'2D Pose Comparison - {seq_name}', fontsize=16)
    
    estimated_poses = []
    
    def update(frame_idx):
        if frame_idx >= len(frames) or frame_idx >= gt_poses_2d.shape[1]:
            return
        
        # Get real frame and ground truth
        frame = frames[frame_idx]
        gt_pose_2d = gt_poses_2d[:, frame_idx, :]
        
        # Get MediaPipe estimation from real frame
        estimated_pose, mp_results = estimator.estimate_pose_from_image(frame)
        estimated_poses.append(estimated_pose)
        
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Ground Truth 2D Pose
        ax1.set_title(f'Ground Truth 2D Pose - {seq_name}', fontsize=14)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.invert_yaxis()
        
        # Plot GT keypoints with better visibility
        valid_gt = gt_pose_2d[:, 2] > 0.1
        if np.any(valid_gt):
            ax1.scatter(gt_pose_2d[valid_gt, 0], gt_pose_2d[valid_gt, 1], 
                       c='blue', s=100, alpha=0.9, edgecolors='darkblue', linewidth=2)
        
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
                        'b-', linewidth=4, alpha=0.8)
        
        # Add joint labels for ground truth
        joint_names = ['Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
                      'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
                      'RShoulder', 'RElbow', 'RWrist']
        
        for i, (valid, name) in enumerate(zip(valid_gt, joint_names)):
            if valid:
                ax1.annotate(f'{i}', (gt_pose_2d[i, 0], gt_pose_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, color='white', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
        
        ax1.set_xlabel('X (normalized)', fontsize=12)
        ax1.set_ylabel('Y (normalized)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MediaPipe Predictions
        ax2.set_title(f'MediaPipe 2D Pose Predictions - {seq_name}', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.invert_yaxis()
        
        # Plot MediaPipe keypoints
        valid_est = estimated_pose[:, 2] > 0.1
        if np.any(valid_est):
            ax2.scatter(estimated_pose[valid_est, 0], estimated_pose[valid_est, 1], 
                       c='red', s=100, alpha=0.9, edgecolors='darkred', linewidth=2)
        
        # Draw MediaPipe skeleton connections
        for connection in connections:
            joint1, joint2 = connection
            if (estimated_pose[joint1, 2] > 0.1 and estimated_pose[joint2, 2] > 0.1):
                ax2.plot([estimated_pose[joint1, 0], estimated_pose[joint2, 0]], 
                        [estimated_pose[joint1, 1], estimated_pose[joint2, 1]], 
                        'r-', linewidth=4, alpha=0.8)
        
        # Add joint labels for MediaPipe
        for i, (valid, name) in enumerate(zip(valid_est, joint_names)):
            if valid:
                ax2.annotate(f'{i}', (estimated_pose[i, 0], estimated_pose[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, color='white', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
        
        ax2.set_xlabel('X (normalized)', fontsize=12)
        ax2.set_ylabel('Y (normalized)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Update frame info with better statistics
        detected_joints = np.sum(valid_est)
        gt_joints = np.sum(valid_gt)
        
        # Calculate 2D pose similarity (simple distance metric)
        if np.any(valid_gt) and np.any(valid_est):
            # Only compare joints that are valid in both GT and estimation
            common_valid = valid_gt & valid_est
            if np.any(common_valid):
                distances = np.linalg.norm(gt_pose_2d[common_valid, :2] - estimated_pose[common_valid, :2], axis=1)
                avg_error = np.mean(distances)
                fig.suptitle(f'2D Pose Comparison - {seq_name}\n'
                            f'Frame: {frame_idx+1}/{len(frames)} | '
                            f'GT Joints: {gt_joints} | MediaPipe Joints: {detected_joints} | '
                            f'Avg Error: {avg_error:.3f}', fontsize=16)
            else:
                fig.suptitle(f'2D Pose Comparison - {seq_name}\n'
                            f'Frame: {frame_idx+1}/{len(frames)} | '
                            f'GT Joints: {gt_joints} | MediaPipe Joints: {detected_joints} | '
                            f'No common joints', fontsize=16)
        else:
            fig.suptitle(f'2D Pose Comparison - {seq_name}\n'
                        f'Frame: {frame_idx+1}/{len(frames)} | '
                        f'GT Joints: {gt_joints} | MediaPipe Joints: {detected_joints}', fontsize=16)
        
        plt.tight_layout()
        
        return [ax1, ax2]
    
    return update, fig

def create_better_dummy_frames(gt_poses_2d, num_frames, image_size=(640, 480)):
    """Create better dummy frames with human-like stick figures"""
    frames = []
    
    for t in range(min(num_frames, gt_poses_2d.shape[1])):
        # Create a more realistic background
        frame = np.random.randint(20, 50, (image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Add some texture/noise
        noise = np.random.randint(-10, 10, (image_size[1], image_size[0], 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        pose_2d = gt_poses_2d[:, t, :]
        
        # Draw a more realistic human figure
        # Draw head as a circle
        if pose_2d[9, 2] > 0.1:  # nose
            head_x = int(pose_2d[9, 0] * image_size[0])
            head_y = int(pose_2d[9, 1] * image_size[1])
            cv2.circle(frame, (head_x, head_y), 25, (200, 180, 150), -1)  # Skin color
            cv2.circle(frame, (head_x, head_y), 25, (100, 100, 100), 2)   # Outline
        
        # Draw torso as a rectangle
        if pose_2d[8, 2] > 0.1 and pose_2d[0, 2] > 0.1:  # thorax and root
            torso_top_x = int(pose_2d[8, 0] * image_size[0])
            torso_top_y = int(pose_2d[8, 1] * image_size[1])
            torso_bottom_x = int(pose_2d[0, 0] * image_size[0])
            torso_bottom_y = int(pose_2d[0, 1] * image_size[1])
            
            cv2.rectangle(frame, 
                         (torso_top_x - 30, torso_top_y), 
                         (torso_bottom_x + 30, torso_bottom_y), 
                         (100, 150, 200), -1)  # Shirt color
            cv2.rectangle(frame, 
                         (torso_top_x - 30, torso_top_y), 
                         (torso_bottom_x + 30, torso_bottom_y), 
                         (50, 50, 50), 2)      # Outline
        
        # Draw limbs as thick lines
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
                
                # Different colors for different body parts
                if joint1 in [4, 5, 6] or joint2 in [4, 5, 6]:  # Left leg
                    color = (100, 100, 150)
                elif joint1 in [1, 2, 3] or joint2 in [1, 2, 3]:  # Right leg
                    color = (100, 100, 150)
                elif joint1 in [11, 12, 13] or joint2 in [11, 12, 13]:  # Left arm
                    color = (200, 180, 150)
                elif joint1 in [14, 15, 16] or joint2 in [14, 15, 16]:  # Right arm
                    color = (200, 180, 150)
                else:  # Torso
                    color = (150, 150, 150)
                
                cv2.line(frame, (x1, y1), (x2, y2), color, 8)
        
        # Draw joints as circles
        for joint_idx in range(17):
            if pose_2d[joint_idx, 2] > 0.1:
                x = int(pose_2d[joint_idx, 0] * image_size[0])
                y = int(pose_2d[joint_idx, 1] * image_size[1])
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)
        
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
    
    print(f"Initializing MediaPipe pose estimator for sequence: {args.sequence_name}")
    estimator = MediaPipe2DPoseEstimator()
    
    try:
        print("Loading test data from dataset...")
        gt_poses_2d, seq_name = load_test_data_from_dataset(args)
        
        if gt_poses_2d is None:
            print("No ground truth data loaded. Please check your dataset.")
            return
        
        print(f"✓ Loaded {gt_poses_2d.shape[1]} ground truth frames for sequence {seq_name}")
        
        # Verify we got the right sequence
        if args.sequence_name and seq_name != args.sequence_name:
            print(f"WARNING: Requested {args.sequence_name} but loaded {seq_name}")
        
        # Debug: Print some statistics
        print(f"Ground truth pose stats:")
        print(f"  Shape: {gt_poses_2d.shape}")
        print(f"  X range: {np.min(gt_poses_2d[:, :, 0]):.3f} to {np.max(gt_poses_2d[:, :, 0]):.3f}")
        print(f"  Y range: {np.min(gt_poses_2d[:, :, 1]):.3f} to {np.max(gt_poses_2d[:, :, 1]):.3f}")
        print(f"  Confidence range: {np.min(gt_poses_2d[:, :, 2]):.3f} to {np.max(gt_poses_2d[:, :, 2]):.3f}")
        
        # Load actual video frames
        print("Loading video frames...")
        frames = load_mpi_test_frames(seq_name, args.num_frames)
        
        if frames is None:
            print("Could not load video frames. Creating fallback visualization...")
            # Create dummy frames with better human-like figures
            frames = create_better_dummy_frames(gt_poses_2d, args.num_frames)
        
        print(f"✓ Using {len(frames)} frames for MediaPipe processing")
        
        # Ensure we have matching number of frames
        min_frames = min(len(frames), gt_poses_2d.shape[1])
        frames = frames[:min_frames]
        gt_poses_2d = gt_poses_2d[:, :min_frames, :]
        
        # Create visualization
        update_func, fig = create_simple_visualization(estimator, frames, gt_poses_2d, seq_name, args)
        
        if args.real_time:
            # Real-time processing
            print("Starting real-time processing...")
            for i in range(min_frames):
                update_func(i)
                plt.pause(0.1)
                if i == 0:
                    plt.show(block=False)
            
            input("Press Enter to close...")
            
        else:
            # Create animation
            print("Creating animation...")
            ani = FuncAnimation(fig, update_func, frames=min_frames, 
                              interval=300, repeat=True, blit=False)
            
            if args.save_video:
                output_path = f'../2d_pose_comparison_{seq_name.lower()}_frames_{min_frames}.gif'
                print(f"Saving animation to: {output_path}")
                
                ani.save(output_path, writer='pillow', fps=3, dpi=100)
                print(f"✓ Animation saved successfully to: {output_path}")
            
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