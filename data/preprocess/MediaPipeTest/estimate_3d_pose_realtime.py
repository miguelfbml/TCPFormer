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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from data.reader.motion_dataset import MPI3DHP, Fusion
from data.const import H36M_TO_MPI

# Ground truth skeleton connections
connections = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

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

def load_mpi_test_frames(sequence_name, num_frames=50):
    """Load video frames from MPI-INF-3DHP test set"""
    
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    
    # Check if the path exists
    if not os.path.exists(video_path):
        print(f"Video frames not found at: {video_path}")
        return None, []
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    
    image_files.sort()
    
    # Limit frames to prevent memory issues
    if num_frames > 1000:
        print(f"⚠️  Warning: Requested {num_frames} frames, limiting to 1000 to prevent memory issues")
        num_frames = 1000
    
    # Load frames in smaller batches to manage memory
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
                # Extract frame number from filename
                frame_idx = int(os.path.basename(img_path).split('_')[-1].split('.')[0])
                frame_indices.append(frame_idx)
            except:
                frame_indices.append(len(frame_indices))
    
    print(f"✓ Loaded {len(frames)} frames")
    return frames, frame_indices

def load_test_3d_data_from_dataset_multiple_samples(args):
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
    
    sequence_samples = []
    sample_info = []
    
    for i in range(len(dataset)):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            current_seq_name = seq[0] if isinstance(seq, (list, tuple)) else str(seq)
            if current_seq_name != target_seq_name:
                continue
            
            print(f"Found matching sequence: {current_seq_name} (sample {i})")
            
            if isinstance(gt_3D, torch.Tensor):
                gt_3D = gt_3D.clone()
            else:
                gt_3D = torch.tensor(gt_3D)
                
            gt_3D = gt_3D.view(1, -1, 17, 3)
            gt_3D[:, :, 14] = 0
            
            center_frame_idx = gt_3D.shape[1] // 2
            center_frame = gt_3D[0, center_frame_idx]
            center_frame = center_frame - center_frame[14:15, :]
            
            if hasattr(center_frame, 'cpu'):
                center_frame = center_frame.cpu().numpy()
            
            sequence_samples.append(center_frame)
            sample_start_frame = i * dataset_args.stride
            center_frame_absolute = sample_start_frame + center_frame_idx
            sample_info.append(center_frame_absolute)
            
            if len(sequence_samples) >= args.num_frames:
                break
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if not sequence_samples:
        return None, None, None
    
    sequence_3d = np.stack(sequence_samples, axis=0).transpose(1, 0, 2)
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    print(f"Loaded {sequence_3d.shape[1]} center frames from {len(sequence_samples)} samples for sequence: {target_seq_name}")
    print(f"GT sequence shape: {sequence_3d.shape}")
    
    return sequence_3d, target_seq_name, sample_info

def process_frame_batch(estimator, frames, batch_size=10):
    pred_poses_3d = []
    visibilities = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        for frame in batch_frames:
            pose_3d, visibility = estimator.estimate_3d_pose_from_image(frame)
            pred_poses_3d.append(pose_3d)
            visibilities.append(visibility)
        gc.collect()
        print(f"Processed batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}")
    return np.stack(pred_poses_3d, axis=1), np.stack(visibilities, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, help='Sequence name (TS1, TS2, etc.)')
    parser.add_argument('--num-frames', type=int, default=20, help='Number of frames to process')
    parser.add_argument('--save-video', action='store_true', help='Save animation as GIF')
    parser.add_argument('--frame-start', type=int, default=0, help='Starting frame for comparison')
    parser.add_argument('--batch-size', type=int, default=50, help='Process frames in batches to manage memory')
    parser.add_argument('--resize-frames', action='store_true', help='Resize frames to 640x480 to save memory')
    args = parser.parse_args()
    
    # Memory management warnings
    if args.num_frames > 200:
        print(f"⚠️  Warning: Processing {args.num_frames} frames may use significant memory")
        print("Consider using --batch-size or --resize-frames to reduce memory usage")
    
    estimator = MediaPipe3DPoseEstimator()
    
    try:
        # Load GT data
        gt_poses_3d, seq_name, sample_info = load_test_3d_data_from_dataset_multiple_samples(args)
        if gt_poses_3d is None:
            print("Failed to load ground truth data.")
            return
        
        # Load video frames only if saving video
        if args.save_video:
            print("Loading video frames for visualization...")
            all_video_frames, all_video_frame_indices = load_mpi_test_frames(seq_name, args.num_frames * 20)
            if not all_video_frames:
                print("No video frames loaded. Exiting.")
                return
            
            print(f"Loaded {len(all_video_frames)} video frames")
            
            # Sample video frames with stride pattern
            stride = 9
            center_offset = 13
            sampled_video_frames = []
            sampled_frame_indices = []
            
            for i in range(gt_poses_3d.shape[1]):
                video_idx = i * stride + center_offset
                if video_idx < len(all_video_frames):
                    frame = all_video_frames[video_idx]
                    
                    # Resize frames if requested to save memory
                    if args.resize_frames:
                        frame = cv2.resize(frame, (640, 480))
                    
                    sampled_video_frames.append(frame)
                    sampled_frame_indices.append(all_video_frame_indices[video_idx] if video_idx < len(all_video_frame_indices) else video_idx)
            
            num_frames = min(len(sampled_video_frames), gt_poses_3d.shape[1], args.num_frames)
            final_video_frames = sampled_video_frames[:num_frames]
            final_frame_indices = sampled_frame_indices[:num_frames]
        else:
            # For MPJPE calculation only - create dummy frames
            num_frames = min(gt_poses_3d.shape[1], args.num_frames)
            print(f"Creating {num_frames} dummy frames for MPJPE calculation...")
            # Create minimal dummy frames (just black images)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            final_video_frames = [dummy_frame.copy() for _ in range(num_frames)]
            final_frame_indices = list(range(num_frames))
        
        final_gt_poses = gt_poses_3d[:, :num_frames, :]
        
        print(f"Using {num_frames} frames")
        print(f"GT shape: {final_gt_poses.shape}")
        
        if num_frames == 0:
            print("No frames available. Exiting.")
            return
        
        # Process frames in batches to manage memory
        batch_size = min(args.batch_size, num_frames)
        print(f"Processing MediaPipe poses in batches of {batch_size}...")
        
        pred_poses_3d = []
        visibilities = []
        
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            print(f"Processing batch {batch_start//batch_size + 1}/{(num_frames-1)//batch_size + 1}: frames {batch_start}-{batch_end-1}")
            
            # Process batch
            batch_poses = []
            batch_vis = []
            
            for i in range(batch_start, batch_end):
                pose_3d, visibility = estimator.estimate_3d_pose_from_image(final_video_frames[i])
                batch_poses.append(pose_3d)
                batch_vis.append(visibility)
            
            # Add to main arrays
            pred_poses_3d.extend(batch_poses)
            visibilities.extend(batch_vis)
            
            # Clear batch data to free memory
            del batch_poses, batch_vis
            gc.collect()
        
        # Convert to numpy arrays
        pred_poses_3d = np.stack(pred_poses_3d, axis=1)  # (17, T, 3)
        visibilities = np.stack(visibilities, axis=1)  # (17, T)
        
        print("✓ All MediaPipe poses computed")
        
        # Calculate MPJPE
        print("Computing MPJPE...")
        valid_joints = visibilities > 0.1
        mpjpe = np.zeros(num_frames, dtype=np.float32)
        valid_frame_count = 0
        
        for t in range(num_frames):
            if t % 100 == 0:
                print(f"  MPJPE progress: {t}/{num_frames}")
            
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
            print(f"\nOverall MPJPE: {overall_mpjpe:.2f} mm (computed on {valid_frame_count}/{num_frames} frames)")
            
            # Print per-frame MPJPE statistics
            print(f"MPJPE Statistics:")
            print(f"  Mean: {overall_mpjpe:.2f} mm")
            print(f"  Min:  {np.min(valid_mpjpe):.2f} mm")
            print(f"  Max:  {np.max(valid_mpjpe):.2f} mm")
            print(f"  Std:  {np.std(valid_mpjpe):.2f} mm")
        else:
            overall_mpjpe = float('inf')
            print("Could not compute MPJPE - insufficient valid joints")
        
        # Only create visualization if --save-video flag is set
        if args.save_video:
            print("\nCreating visualization...")
            
            # For very large datasets, ask user if they want visualization
            if num_frames > 100:
                response = input(f"Create animation with {num_frames} frames? This may take time and memory (y/n): ")
                if response.lower() != 'y':
                    print("Skipping visualization. Results saved.")
                    return
            
            # Only print joint mappings when creating video
            GT_JOINT_NAMES = {
                0: "Head Top", 1: "Neck", 2: "Right Arm", 3: "Right Forearm", 4: "Right Hand",
                5: "Left Arm", 6: "Left Forearm", 7: "Left Hand", 8: "Right Up Leg", 9: "Right Leg",
                10: "Right Foot", 11: "Left Up Leg", 12: "Left Leg", 13: "Left Foot", 14: "Hip",
                15: "Spine", 16: "Head"
            }
            
            print("Ground Truth Joint Mappings:")
            for idx, name in GT_JOINT_NAMES.items():
                print(f"Joint {idx}: {name}")
            
            # Set up visualization bounds
            all_gt = final_gt_poses.reshape(-1, 3)
            all_pred = pred_poses_3d.reshape(-1, 3)
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
            
            # Create matplotlib figure
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
                    ax1.text(x_gt[joint_idx], y_gt[joint_idx], z_gt[joint_idx], 
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
                            ax2.scatter(x_pos, y_pos, z_pos, 
                                       c='red', s=100, alpha=0.9, 
                                       edgecolors='darkred', linewidth=2)
                            ax2.text(x_pos, y_pos, z_pos, 
                                    str(joint_idx), fontsize=24, color='yellow', weight='bold',
                                    ha='center', va='center',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7, edgecolor='white'))
                        else:
                            ax2.scatter(x_pos, y_pos, z_pos, 
                                       c='gray', s=50, alpha=0.5, 
                                       edgecolors='black', linewidth=1)
                            ax2.text(x_pos, y_pos, z_pos, 
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
            
            # Create animation with reduced interval for large datasets
            interval = 300 if num_frames <= 50 else 100
            ani = FuncAnimation(fig, update, frames=range(num_frames), interval=interval, repeat=True, blit=False)
            
            output_path = f'../mpi_mediapipe_comparison_{seq_name.lower()}_mpjpe_{overall_mpjpe:.1f}mm.gif'
            print(f"Saving animation to: {output_path}")
            
            # Reduce quality for large animations
            dpi = 120 if num_frames <= 100 else 80
            fps = 3 if num_frames <= 100 else 5
            
            ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
            print(f"Comparison GIF saved to: {output_path}")
            
            # Save static image
            update(0)
            plt.tight_layout()
            static_path = output_path.replace('.gif', '.png')
            plt.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"Static image saved to: {static_path}")
            
            # Clean up matplotlib objects
            plt.close(fig)
            del ani, fig, ax1, ax2
            
            # Clean up visualization data
            del all_gt, all_pred, valid_gt, valid_pred, all_poses
        
        else:
            # No visualization mode
            print(f"\nMPJPE calculation completed for sequence {seq_name}")
            print("Use --save-video flag to create visualization")
        
        # Clean up pose data
        del pred_poses_3d, visibilities, final_gt_poses
        if args.save_video:
            del final_video_frames
        
    finally:
        estimator.close()
        gc.collect()

if __name__ == '__main__':
    main()