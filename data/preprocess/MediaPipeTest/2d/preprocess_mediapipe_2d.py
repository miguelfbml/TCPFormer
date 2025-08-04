"""
Preprocess MediaPipe 2D poses for MPI-INF-3DHP test sequences and save to files
This creates MediaPipe equivalent of ground truth 2D poses for efficient comparison

Usage:
python preprocess_mediapipe_2d.py --sequence TS1 --output-dir mediapipe_2d_poses
python preprocess_mediapipe_2d.py --all-sequences --output-dir mediapipe_2d_poses
"""

import argparse
import os
import cv2
import numpy as np
import glob
import json
import time
from tqdm import tqdm
import gc

# Navigate to project root
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from data.reader.motion_dataset import Fusion
from utils.tools import get_config

class MediaPipe2DPoseEstimator:
    def __init__(self, resize_resolution=(640, 480)):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.resize_resolution = resize_resolution
        
        # MediaPipe to MPI-INF-3DHP joint mapping
        self.mp_to_mpi_mapping = {
            0: 16,   # nose -> head
            11: 5,   # left_shoulder -> left shoulder
            12: 2,   # right_shoulder -> right shoulder
            13: 6,   # left_elbow -> left elbow
            14: 3,   # right_elbow -> right elbow
            15: 7,   # left_wrist -> left wrist
            16: 4,   # right_wrist -> right wrist
            23: 11,  # left_hip -> left hip
            24: 8,   # right_hip -> right hip
            25: 12,  # left_knee -> left knee
            26: 9,   # right_knee -> right knee
            27: 13,  # left_ankle -> left ankle
            28: 10,  # right_ankle -> right ankle
        }
        
        # Missing joint estimation
        self.missing_joints_estimation = {
            0: [11, 8],  # root from hips
            1: [5, 2],   # neck from shoulders
            14: [11, 8], # hip from left/right hips
            15: [14, 1], # spine from hip and neck
        }

    def estimate_2d_pose_from_image(self, image):
        """Estimate 2D pose from image, return normalized coordinates [0,1] with confidence."""
        if image is None:
            return np.zeros((17, 3), dtype=np.float32)
        
        # Resize for faster processing
        if self.resize_resolution:
            image = cv2.resize(image, self.resize_resolution, interpolation=cv2.INTER_AREA)
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        pose_2d = np.zeros((17, 3), dtype=np.float32)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            confidence_threshold = 0.3

            # Map MediaPipe landmarks to MPI joints
            for mp_idx, mpi_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks) and landmarks[mp_idx].visibility > confidence_threshold:
                    pose_2d[mpi_idx] = [landmarks[mp_idx].x, landmarks[mp_idx].y, landmarks[mp_idx].visibility]

            # Estimate missing joints
            for missing_joint, source_joints in self.missing_joints_estimation.items():
                valid_sources = [j for j in source_joints if pose_2d[j, 2] > confidence_threshold]
                if valid_sources:
                    pose_2d[missing_joint, :2] = np.mean([pose_2d[j, :2] for j in valid_sources], axis=0)
                    pose_2d[missing_joint, 2] = np.mean([pose_2d[j, 2] for j in valid_sources]) * 0.9

            # Root joint from hips
            if pose_2d[11, 2] > confidence_threshold and pose_2d[8, 2] > confidence_threshold:
                pose_2d[0, :2] = (pose_2d[11, :2] + pose_2d[8, :2]) / 2.0
                pose_2d[0, 1] -= 0.05  # Upward offset
                pose_2d[0, 2] = min(pose_2d[11, 2], pose_2d[8, 2])

        return pose_2d

    def close(self):
        self.pose.close()

def load_sequence_images(sequence_name):
    """Load all images for a sequence."""
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video path not found: {video_path}")
        return None, None
    
    # Load image file paths
    image_files = []
    for ext in ['*.jpg']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    
    image_files.sort()  # Sort numerically: 000001.jpg, 000002.jpg, etc.
    
    if not image_files:
        print(f"ERROR: No image files found in {video_path}")
        return None, None
    
    print(f"Found {len(image_files)} images for sequence {sequence_name}")
    return image_files, video_path

def process_sequence_mediapipe(sequence_name, estimator, args):
    """Process all images in a sequence with MediaPipe."""
    print(f"\nProcessing sequence: {sequence_name}")
    
    # Load image paths
    image_files, video_path = load_sequence_images(sequence_name)
    if image_files is None:
        return None
    
    # Process images in batches
    all_poses_2d = []
    batch_size = 50  # Process in batches to manage memory
    
    print(f"Processing {len(image_files)} images in batches of {batch_size}...")
    
    for batch_start in tqdm(range(0, len(image_files), batch_size), desc=f"Processing {sequence_name}"):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_poses = []
        
        for i in range(batch_start, batch_end):
            image_path = image_files[i]
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load {image_path}")
                # Add zero pose for missing image
                batch_poses.append(np.zeros((17, 3), dtype=np.float32))
                continue
            
            # Get MediaPipe 2D pose
            pose_2d = estimator.estimate_2d_pose_from_image(image)
            batch_poses.append(pose_2d)
        
        all_poses_2d.extend(batch_poses)
        
        # Clear memory
        del batch_poses
        gc.collect()
        
        # Progress update
        if batch_start % (batch_size * 10) == 0:
            valid_poses = sum(1 for p in all_poses_2d if np.sum(p[:, 2] > 0.1) > 5)
            print(f"  Processed {len(all_poses_2d)}/{len(image_files)} images, {valid_poses} with valid poses")
    
    # Convert to numpy array: (num_frames, 17, 3)
    sequence_poses = np.stack(all_poses_2d, axis=0)
    
    # Statistics
    valid_frames = np.sum([np.sum(pose[:, 2] > 0.1) > 5 for pose in sequence_poses])
    avg_confidence = np.mean(sequence_poses[:, :, 2])
    
    print(f"✓ Sequence {sequence_name} completed:")
    print(f"  Total frames: {len(sequence_poses)}")
    print(f"  Valid frames (>5 joints): {valid_frames}/{len(sequence_poses)} ({100*valid_frames/len(sequence_poses):.1f}%)")
    print(f"  Average confidence: {avg_confidence:.3f}")
    
    return sequence_poses

def save_mediapipe_poses(sequence_name, poses_2d, output_dir, args):
    """Save MediaPipe poses in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy file (efficient for Python)
    np_path = os.path.join(output_dir, f'{sequence_name}_mediapipe_2d.npz')
    np.savez_compressed(np_path, 
                       poses_2d=poses_2d,
                       sequence_name=sequence_name,
                       num_frames=len(poses_2d))
    
    # Save metadata as JSON
    metadata = {
        'sequence_name': sequence_name,
        'num_frames': len(poses_2d),
        'pose_shape': list(poses_2d.shape),
        'valid_frames': int(np.sum([np.sum(pose[:, 2] > 0.1) > 5 for pose in poses_2d])),
        'avg_confidence': float(np.mean(poses_2d[:, :, 2])),
        'coordinate_system': 'normalized_[0,1]',
        'joint_order': [
            'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
            'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
            'RShoulder', 'RElbow', 'RWrist'
        ],
        'processing_params': {
            'resize_resolution': args.resize_resolution,
            'confidence_threshold': 0.3
        }
    }
    
    json_path = os.path.join(output_dir, f'{sequence_name}_mediapipe_2d_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved MediaPipe poses:")
    print(f"  Data: {np_path}")
    print(f"  Metadata: {json_path}")
    
    return np_path, json_path

def process_all_sequences(estimator, args):
    """Process all MPI-INF-3DHP test sequences."""
    test_sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    if args.sequence and args.sequence in test_sequences:
        sequences_to_process = [args.sequence]
    elif args.all_sequences:
        sequences_to_process = test_sequences
    else:
        sequences_to_process = [args.sequence or 'TS1']
    
    print(f"Will process sequences: {sequences_to_process}")
    
    results = {}
    total_start_time = time.time()
    
    for seq_name in sequences_to_process:
        seq_start_time = time.time()
        
        # Process sequence
        poses_2d = process_sequence_mediapipe(seq_name, estimator, args)
        
        if poses_2d is not None:
            # Save poses
            np_path, json_path = save_mediapipe_poses(seq_name, poses_2d, args.output_dir, args)
            
            seq_time = time.time() - seq_start_time
            results[seq_name] = {
                'success': True,
                'num_frames': len(poses_2d),
                'processing_time': seq_time,
                'np_path': np_path,
                'json_path': json_path
            }
            
            print(f"✓ {seq_name} completed in {seq_time:.1f}s ({len(poses_2d)/seq_time:.1f} fps)")
        else:
            results[seq_name] = {'success': False}
            print(f"✗ {seq_name} failed")
        
        # Clean up
        del poses_2d
        gc.collect()
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"MediaPipe 2D Pose Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Sequences processed: {len([r for r in results.values() if r['success']])}/{len(sequences_to_process)}")
    
    successful_sequences = {k: v for k, v in results.items() if v['success']}
    if successful_sequences:
        total_frames = sum(r['num_frames'] for r in successful_sequences.values())
        print(f"Total frames processed: {total_frames}")
        print(f"Average processing speed: {total_frames/total_time:.1f} fps")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'results': results,
            'total_time': total_time,
            'sequences_processed': list(successful_sequences.keys()),
            'args': vars(args)
        }, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_path}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess MediaPipe 2D poses for MPI-INF-3DHP test sequences')
    parser.add_argument('--sequence', type=str, default=None, 
                       help='Specific sequence to process (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--all-sequences', action='store_true',
                       help='Process all test sequences (TS1-TS6)')
    parser.add_argument('--output-dir', type=str, default='mediapipe_2d_poses',
                       help='Directory to save MediaPipe poses')
    parser.add_argument('--resize-resolution', type=int, nargs=2, default=[640, 480],
                       help='Resize images to W H for MediaPipe processing')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("MediaPipe 2D Pose Preprocessing for MPI-INF-3DHP Test Set")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Resize resolution: {args.resize_resolution}")
    
    if args.all_sequences:
        print("Processing: All sequences (TS1-TS6)")
    elif args.sequence:
        print(f"Processing: {args.sequence}")
    else:
        print("Processing: TS1 (default)")
    
    # Initialize MediaPipe
    print("\nInitializing MediaPipe...")
    estimator = MediaPipe2DPoseEstimator(resize_resolution=tuple(args.resize_resolution))
    
    try:
        # Process sequences
        results = process_all_sequences(estimator, args)
        
        print(f"\n✓ Preprocessing completed!")
        print(f"Use the saved .npz files for fast comparison with ground truth 2D poses")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.close()
        print("✓ MediaPipe estimator closed")

if __name__ == '__main__':
    main()