"""
Create MediaPipe 3D version of MPI-INF-3DHP test dataset
This replaces the 3D poses with MediaPipe 3D estimations while keeping everything else identical

python3 preprocess_mediapipe_3d.py
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
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

class MediaPipe3DPoseEstimator:
    def __init__(self, resize_resolution=(640, 480)):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            print("✓ MediaPipe 3D initialized successfully")
        except ImportError:
            print("ERROR: MediaPipe not installed. Install with: pip install mediapipe")
            sys.exit(1)
        
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
        
        # Missing joint estimation for better skeleton
        self.missing_joints_estimation = {
            1: [5, 2],   # neck from shoulders
            14: [11, 8], # hip from left/right hips
            15: [14, 1], # spine from hip and neck
        }

    def estimate_3d_pose_from_image(self, image):
        """Estimate 3D pose from image, return coordinates in mm (MPI-INF-3DHP format)"""
        if image is None:
            return np.zeros((17, 3), dtype=np.float32), np.zeros(17, dtype=np.float32)
        
        try:
            # Resize for MediaPipe processing if specified
            if self.resize_resolution:
                image = cv2.resize(image, self.resize_resolution, interpolation=cv2.INTER_AREA)
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            
            pose_3d = np.zeros((17, 3), dtype=np.float32)
            visibility = np.zeros(17, dtype=np.float32)
            
            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark
                confidence_threshold = 0.1

                # Map MediaPipe landmarks to MPI joints
                for mp_idx, mpi_idx in self.mp_to_mpi_mapping.items():
                    if mp_idx < len(landmarks) and landmarks[mp_idx].visibility > confidence_threshold:
                        # MediaPipe world coordinates are in meters, convert to mm
                        pose_3d[mpi_idx] = [
                            landmarks[mp_idx].x * 1000,  # Convert m to mm
                            landmarks[mp_idx].y * 1000,
                            landmarks[mp_idx].z * 1000
                        ]
                        visibility[mpi_idx] = landmarks[mp_idx].visibility

                # Estimate head top (joint 0) from nose if available
                if results.pose_landmarks and len(results.pose_landmarks.landmark) > 0:
                    nose_landmark = results.pose_landmarks.landmark[0]
                    if nose_landmark.visibility > confidence_threshold:
                        # Use world coordinates for head estimation
                        if 0 < len(landmarks):
                            nose_world = landmarks[0]
                            # Estimate head top by moving up from nose
                            pose_3d[0] = [
                                nose_world.x * 1000,
                                nose_world.y * 1000 - 50,  # Move up 50mm from nose
                                nose_world.z * 1000
                            ]
                            visibility[0] = nose_world.visibility * 0.8

                # Estimate missing joints using interpolation
                for missing_joint, source_joints in self.missing_joints_estimation.items():
                    valid_sources = [j for j in source_joints if visibility[j] > confidence_threshold]
                    if valid_sources:
                        pose_3d[missing_joint] = np.mean([pose_3d[j] for j in valid_sources], axis=0)
                        visibility[missing_joint] = np.mean([visibility[j] for j in valid_sources]) * 0.9

                # Make sure we have hip center (joint 14)
                if visibility[11] > confidence_threshold and visibility[8] > confidence_threshold:
                    pose_3d[14] = (pose_3d[11] + pose_3d[8]) / 2.0
                    visibility[14] = min(visibility[11], visibility[8])

                # CRITICAL FIX: Make root-relative BEFORE coordinate transformation
                if visibility[14] > 0.1:  # Hip joint
                    root_pos = pose_3d[14].copy()
                    pose_3d = pose_3d - root_pos[np.newaxis, :]

                # Apply coordinate transformation to match MPI-INF-3DHP coordinate system
                # MediaPipe: X=right, Y=down, Z=forward
                # MPI-INF-3DHP: X=right, Y=up, Z=backward
                cam2real = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
                pose_3d = pose_3d @ cam2real

            return pose_3d, visibility
            
        except Exception as e:
            print(f"Error processing image for 3D pose: {e}")
            return np.zeros((17, 3), dtype=np.float32), np.zeros(17, dtype=np.float32)

    def close(self):
        if hasattr(self, 'pose'):
            self.pose.close()

def get_sequence_image_dimensions(sequence_name):
    """Get the original image dimensions for a sequence"""
    # TS5 and TS6 use 1920x1080, others use 2048x2048
    if sequence_name in ['TS5', 'TS6']:
        return 1920, 1080
    else:
        return 2048, 2048

def load_original_dataset():
    """Load the original MPI-INF-3DHP test dataset"""
    dataset_path = project_root + 'data/preprocess/MediaPipeTest/data1/motion3d/data_test_3dhp.npz'

    if not os.path.exists(dataset_path):
        print(f"ERROR: Original dataset not found: {dataset_path}")
        return None
    
    print(f"Loading original dataset from: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)['data'].item()
    
    print(f"Found sequences: {list(data.keys())}")
    return data

def load_sequence_images(sequence_name):
    """Load all images for a sequence"""
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    
    if not os.path.exists(video_path):
        alternative_paths = [
            f'/nas-ctm01/datasets/public/mpi_inf_3dhp/{sequence_name}/imageSequence',
            f'/nas-ctm01/datasets/public/mpi_inf_3dhp/test/{sequence_name}/imageSequence'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                video_path = alt_path
                break
        else:
            print(f"ERROR: Video path not found for {sequence_name}")
            return None
    
    # Load image file paths
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    
    image_files.sort()
    
    if not image_files:
        print(f"ERROR: No image files found in {video_path}")
        return None
    
    print(f"Found {len(image_files)} images for sequence {sequence_name}")
    return image_files

def create_mediapipe_3d_dataset(original_data, estimator, output_path):
    """Create new dataset with MediaPipe 3D poses"""
    print("Creating MediaPipe 3D version of dataset...")
    
    # Copy original data structure
    mediapipe_data = {}
    
    for seq_name, seq_data in original_data.items():
        print(f"\nProcessing sequence: {seq_name}")
        
        # Load images for this sequence
        image_files = load_sequence_images(seq_name)
        if image_files is None:
            print(f"Skipping {seq_name}: No images found")
            continue
        
        # Copy all original data except 3D poses
        mediapipe_seq_data = {
            'data_2d': seq_data['data_2d'].copy(),  # Keep original 2D poses
            'valid': seq_data['valid'].copy(),
            'camera': seq_data['camera'].copy() if 'camera' in seq_data else None,
        }
        
        # Get original 3D data shape
        original_3d = seq_data['data_3d']  # Shape: (num_frames, 17, 3)
        num_frames = len(original_3d)
        
        print(f"  Original 3D shape: {original_3d.shape}")
        print(f"  Processing {num_frames} frames...")
        
        # Check original coordinate range for reference
        if original_3d.size > 0:
            print(f"  Original GT 3D coordinate range:")
            print(f"    X: [{np.min(original_3d[:, :, 0]):.1f}, {np.max(original_3d[:, :, 0]):.1f}] mm")
            print(f"    Y: [{np.min(original_3d[:, :, 1]):.1f}, {np.max(original_3d[:, :, 1]):.1f}] mm")
            print(f"    Z: [{np.min(original_3d[:, :, 2]):.1f}, {np.max(original_3d[:, :, 2]):.1f}] mm")
        
        # Process images with MediaPipe 3D
        mediapipe_poses_3d = []
        mediapipe_visibilities = []
        
        for frame_idx in tqdm(range(num_frames), desc=f"Processing {seq_name}"):
            if frame_idx < len(image_files):
                # Load and process image
                image_path = image_files[frame_idx]
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Get MediaPipe 3D pose
                    pose_3d, visibility = estimator.estimate_3d_pose_from_image(image)
                    mediapipe_poses_3d.append(pose_3d)
                    mediapipe_visibilities.append(visibility)
                else:
                    # Use zero pose for missing image
                    mediapipe_poses_3d.append(np.zeros((17, 3), dtype=np.float32))
                    mediapipe_visibilities.append(np.zeros(17, dtype=np.float32))
            else:
                # Use zero pose for missing frame
                mediapipe_poses_3d.append(np.zeros((17, 3), dtype=np.float32))
                mediapipe_visibilities.append(np.zeros(17, dtype=np.float32))
        
        # Convert to numpy array and store
        mediapipe_seq_data['data_3d'] = np.array(mediapipe_poses_3d, dtype=np.float32)
        
        print(f"  ✓ MediaPipe 3D shape: {mediapipe_seq_data['data_3d'].shape}")
        
        # Check MediaPipe coordinate range
        if mediapipe_seq_data['data_3d'].size > 0:
            # Filter out zero poses for coordinate range analysis
            non_zero_mask = ~np.all(mediapipe_seq_data['data_3d'] == 0, axis=(1, 2))
            if np.any(non_zero_mask):
                valid_poses = mediapipe_seq_data['data_3d'][non_zero_mask]
                print(f"  MediaPipe 3D coordinate range (valid poses only):")
                print(f"    X: [{np.min(valid_poses[:, :, 0]):.1f}, {np.max(valid_poses[:, :, 0]):.1f}] mm")
                print(f"    Y: [{np.min(valid_poses[:, :, 1]):.1f}, {np.max(valid_poses[:, :, 1]):.1f}] mm")
                print(f"    Z: [{np.min(valid_poses[:, :, 2]):.1f}, {np.max(valid_poses[:, :, 2]):.1f}] mm")
                
                # Calculate average visibility
                avg_visibility = np.mean([np.mean(v[v > 0]) for v in mediapipe_visibilities if np.any(v > 0)])
                print(f"    Average visibility: {avg_visibility:.3f}")
            else:
                print(f"  WARNING: All MediaPipe 3D poses are zero for {seq_name}")
        
        # Verify shapes match
        assert mediapipe_seq_data['data_3d'].shape == original_3d.shape, \
            f"Shape mismatch: MediaPipe {mediapipe_seq_data['data_3d'].shape} vs Original {original_3d.shape}"
        
        mediapipe_data[seq_name] = mediapipe_seq_data
        
        # Memory cleanup
        del mediapipe_poses_3d, mediapipe_visibilities
        gc.collect()
    
    # Save the new dataset
    print(f"\nSaving MediaPipe 3D dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, data=mediapipe_data)
    
    print("✓ MediaPipe 3D dataset created successfully!")
    return mediapipe_data

def verify_dataset(original_data, mediapipe_data):
    """Verify the MediaPipe 3D dataset structure matches the original"""
    print("\nVerifying dataset structure...")
    
    for seq_name in original_data.keys():
        if seq_name not in mediapipe_data:
            print(f"WARNING: Sequence {seq_name} missing in MediaPipe dataset")
            continue
        
        orig = original_data[seq_name]
        mp = mediapipe_data[seq_name]
        
        # Check 2D data is identical (we keep original 2D)
        assert np.array_equal(orig['data_2d'], mp['data_2d']), f"2D data mismatch in {seq_name}"
        
        # Check valid frames are identical
        assert np.array_equal(orig['valid'], mp['valid']), f"Valid frames mismatch in {seq_name}"
        
        # Check 3D data shapes match
        assert orig['data_3d'].shape == mp['data_3d'].shape, f"3D shape mismatch in {seq_name}"
        
        print(f"  ✓ {seq_name}: Shapes match, metadata preserved")
    
    print("✓ Dataset verification passed!")

def main():
    parser = argparse.ArgumentParser(description='Create MediaPipe 3D version of MPI-INF-3DHP test dataset')
    parser.add_argument('--output-path', type=str, default='data/motion3d/data_test_3dhp_mediapipe_3d.npz',
                       help='Output path for MediaPipe 3D dataset')
    parser.add_argument('--resize-resolution', type=int, nargs=2, default=[640, 480],
                       help='Resize images to W H for MediaPipe processing')
    args = parser.parse_args()
    
    print("Creating MediaPipe 3D version of MPI-INF-3DHP test dataset")
    print("=" * 60)
    print(f"Output path: {args.output_path}")
    print(f"Resize resolution: {args.resize_resolution}")
    print(f"Note: 3D coordinates will be in mm, root-relative, MPI-INF-3DHP coordinate system")
    
    # Load original dataset
    original_data = load_original_dataset()
    if original_data is None:
        return
    
    # Initialize MediaPipe 3D
    print("\nInitializing MediaPipe 3D...")
    estimator = MediaPipe3DPoseEstimator(resize_resolution=tuple(args.resize_resolution))
    
    try:
        # Create MediaPipe 3D dataset
        mediapipe_data = create_mediapipe_3d_dataset(original_data, estimator, args.output_path)
        
        # Verify dataset
        verify_dataset(original_data, mediapipe_data)
        
        print(f"\n✓ Success! MediaPipe 3D dataset saved to: {args.output_path}")
        print(f"\nCoordinate format: MPI-INF-3DHP compatible")
        print(f"- 3D coordinates in millimeters")
        print(f"- Root-relative poses (hip center at origin)")
        print(f"- Coordinate system: X=right, Y=up, Z=backward")
        print(f"\nTo use in comparison:")
        print(f"1. Use compare_gt_mediapipe_3d.py to compare with ground truth")
        print(f"2. Or modify train_3dhp.py to load this dataset for evaluation")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.close()
        print("✓ MediaPipe 3D estimator closed")

if __name__ == '__main__':
    main()