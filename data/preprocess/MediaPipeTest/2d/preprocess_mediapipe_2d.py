"""
Create MediaPipe version of MPI-INF-3DHP test dataset
This replaces the 2D poses with MediaPipe estimations while keeping everything else identical

python3 preprocess_mediapipe_2d.py
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

class MediaPipe2DPoseEstimator:
    def __init__(self, resize_resolution=(640, 480)):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ MediaPipe initialized successfully")
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
        
        # Missing joint estimation
        self.missing_joints_estimation = {
            1: [5, 2],   # neck from shoulders
            14: [11, 8], # hip from left/right hips
            15: [14, 1], # spine from hip and neck
        }

    def estimate_2d_pose_from_image(self, image, original_width, original_height):
        """Estimate 2D pose from image, return coordinates in original image pixel coordinates"""
        if image is None:
            return np.zeros((17, 3), dtype=np.float32)
        
        try:
            # Store original dimensions
            orig_h, orig_w = image.shape[:2]
            
            # Resize for MediaPipe processing if specified
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
                        # Convert MediaPipe normalized coordinates [0,1] to original image pixel coordinates
                        # MediaPipe gives coordinates relative to the processed (possibly resized) image
                        # We need to map them back to the original image coordinates
                        
                        if self.resize_resolution:
                            # MediaPipe coordinates are relative to resized image
                            # First convert to resized image pixels
                            x_resized_pixel = landmarks[mp_idx].x * self.resize_resolution[0]
                            y_resized_pixel = landmarks[mp_idx].y * self.resize_resolution[1]
                            
                            # Then scale back to original image dimensions
                            x_orig_pixel = x_resized_pixel * (original_width / self.resize_resolution[0])
                            y_orig_pixel = y_resized_pixel * (original_height / self.resize_resolution[1])
                        else:
                            # No resizing, direct conversion
                            x_orig_pixel = landmarks[mp_idx].x * original_width
                            y_orig_pixel = landmarks[mp_idx].y * original_height
                        
                        pose_2d[mpi_idx] = [x_orig_pixel, y_orig_pixel, landmarks[mp_idx].visibility]

                # Estimate missing joints
                for missing_joint, source_joints in self.missing_joints_estimation.items():
                    valid_sources = [j for j in source_joints if pose_2d[j, 2] > confidence_threshold]
                    if valid_sources:
                        pose_2d[missing_joint, :2] = np.mean([pose_2d[j, :2] for j in valid_sources], axis=0)
                        pose_2d[missing_joint, 2] = np.mean([pose_2d[j, 2] for j in valid_sources]) * 0.9

                # Estimate head top (joint 0) from facial landmarks like estimate_3d_pose_realtime.py
                if len(landmarks) > 10:
                    # Use eyebrow landmarks for head top estimation (similar to 3D version)
                    left_eyebrow_inner = landmarks[2] if len(landmarks) > 2 else None
                    right_eyebrow_inner = landmarks[5] if len(landmarks) > 5 else None
                    
                    if left_eyebrow_inner and right_eyebrow_inner:
                        # Calculate head top position from eyebrow landmarks
                        if self.resize_resolution:
                            x_left = left_eyebrow_inner.x * self.resize_resolution[0] * (original_width / self.resize_resolution[0])
                            y_left = left_eyebrow_inner.y * self.resize_resolution[1] * (original_height / self.resize_resolution[1])
                            x_right = right_eyebrow_inner.x * self.resize_resolution[0] * (original_width / self.resize_resolution[0])
                            y_right = right_eyebrow_inner.y * self.resize_resolution[1] * (original_height / self.resize_resolution[1])
                        else:
                            x_left = left_eyebrow_inner.x * original_width
                            y_left = left_eyebrow_inner.y * original_height
                            x_right = right_eyebrow_inner.x * original_width
                            y_right = right_eyebrow_inner.y * original_height
                        
                        # Head top is average of eyebrow positions, moved slightly up
                        pose_2d[0, 0] = (x_left + x_right) / 2.0
                        pose_2d[0, 1] = (y_left + y_right) / 2.0 - original_height * 0.02  # Move up by 2% of image height
                        pose_2d[0, 2] = (left_eyebrow_inner.visibility + right_eyebrow_inner.visibility) / 2.0

                # Root joint (joint 14) from hips
                if pose_2d[11, 2] > confidence_threshold and pose_2d[8, 2] > confidence_threshold:
                    pose_2d[14, :2] = (pose_2d[11, :2] + pose_2d[8, :2]) / 2.0
                    pose_2d[14, 2] = min(pose_2d[11, 2], pose_2d[8, 2])

            return pose_2d
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.zeros((17, 3), dtype=np.float32)

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
        # Try alternative paths
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

def create_mediapipe_dataset(original_data, estimator, output_path):
    """Create new dataset with MediaPipe 2D poses"""
    print("Creating MediaPipe version of dataset...")
    
    # Copy original data structure
    mediapipe_data = {}
    
    for seq_name, seq_data in original_data.items():
        print(f"\nProcessing sequence: {seq_name}")
        
        # Get original image dimensions for this sequence
        orig_width, orig_height = get_sequence_image_dimensions(seq_name)
        print(f"  Original image dimensions: {orig_width}x{orig_height}")
        
        # Load images for this sequence
        image_files = load_sequence_images(seq_name)
        if image_files is None:
            print(f"Skipping {seq_name}: No images found")
            continue
        
        # Copy all original data
        mediapipe_seq_data = {
            'data_3d': seq_data['data_3d'].copy(),
            'valid': seq_data['valid'].copy(),
            'camera': seq_data['camera'].copy() if 'camera' in seq_data else None,
        }
        
        # Get original 2D data shape
        original_2d = seq_data['data_2d']  # Shape: (num_frames, 17, 2) or (num_frames, 17, 3)
        num_frames = len(original_2d)
        
        print(f"  Original 2D shape: {original_2d.shape}")
        print(f"  Processing {num_frames} frames...")
        
        # Check original coordinate range for reference
        if original_2d.size > 0:
            print(f"  Original GT coordinate range:")
            print(f"    X: [{np.min(original_2d[:, :, 0]):.1f}, {np.max(original_2d[:, :, 0]):.1f}]")
            print(f"    Y: [{np.min(original_2d[:, :, 1]):.1f}, {np.max(original_2d[:, :, 1]):.1f}]")
        
        # Process images with MediaPipe
        mediapipe_poses_2d = []
        
        for frame_idx in tqdm(range(num_frames), desc=f"Processing {seq_name}"):
            if frame_idx < len(image_files):
                # Load and process image
                image_path = image_files[frame_idx]
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Get MediaPipe 2D pose in original pixel coordinates
                    pose_2d = estimator.estimate_2d_pose_from_image(image, orig_width, orig_height)
                    mediapipe_poses_2d.append(pose_2d[:, :2])  # Only x, y coordinates
                else:
                    # Use zero pose for missing image
                    mediapipe_poses_2d.append(np.zeros((17, 2), dtype=np.float32))
            else:
                # Use zero pose for missing frame
                mediapipe_poses_2d.append(np.zeros((17, 2), dtype=np.float32))
        
        # Convert to numpy array and store
        mediapipe_seq_data['data_2d'] = np.array(mediapipe_poses_2d, dtype=np.float32)
        
        print(f"  ✓ MediaPipe 2D shape: {mediapipe_seq_data['data_2d'].shape}")
        
        # Check MediaPipe coordinate range
        if mediapipe_seq_data['data_2d'].size > 0:
            # Filter out zero poses for coordinate range analysis
            non_zero_mask = ~np.all(mediapipe_seq_data['data_2d'] == 0, axis=(1, 2))
            if np.any(non_zero_mask):
                valid_poses = mediapipe_seq_data['data_2d'][non_zero_mask]
                print(f"  MediaPipe coordinate range (valid poses only):")
                print(f"    X: [{np.min(valid_poses[:, :, 0]):.1f}, {np.max(valid_poses[:, :, 0]):.1f}]")
                print(f"    Y: [{np.min(valid_poses[:, :, 1]):.1f}, {np.max(valid_poses[:, :, 1]):.1f}]")
            else:
                print(f"  WARNING: All MediaPipe poses are zero for {seq_name}")
        
        # Verify shapes match
        assert mediapipe_seq_data['data_2d'].shape[:2] == original_2d.shape[:2], \
            f"Shape mismatch: MediaPipe {mediapipe_seq_data['data_2d'].shape} vs Original {original_2d.shape}"
        
        mediapipe_data[seq_name] = mediapipe_seq_data
        
        # Memory cleanup
        del mediapipe_poses_2d
        gc.collect()
    
    # Save the new dataset
    print(f"\nSaving MediaPipe dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, data=mediapipe_data)
    
    print("✓ MediaPipe dataset created successfully!")
    return mediapipe_data

def verify_dataset(original_data, mediapipe_data):
    """Verify the MediaPipe dataset structure matches the original"""
    print("\nVerifying dataset structure...")
    
    for seq_name in original_data.keys():
        if seq_name not in mediapipe_data:
            print(f"WARNING: Sequence {seq_name} missing in MediaPipe dataset")
            continue
        
        orig = original_data[seq_name]
        mp = mediapipe_data[seq_name]
        
        # Check 3D data is identical
        assert np.array_equal(orig['data_3d'], mp['data_3d']), f"3D data mismatch in {seq_name}"
        
        # Check valid frames are identical
        assert np.array_equal(orig['valid'], mp['valid']), f"Valid frames mismatch in {seq_name}"
        
        # Check 2D data shapes match
        assert orig['data_2d'].shape[:2] == mp['data_2d'].shape[:2], f"2D shape mismatch in {seq_name}"
        
        print(f"  ✓ {seq_name}: Shapes match, metadata preserved")
    
    print("✓ Dataset verification passed!")


def main():
    parser = argparse.ArgumentParser(description='Create MediaPipe version of MPI-INF-3DHP test dataset')
    parser.add_argument('--output-path', type=str, default='data/preprocess/MediaPipeTest/data1/motion3d/data_test_3dhp_mediapipe.npz',
                       help='Output path for MediaPipe dataset')
    parser.add_argument('--resize-resolution', type=int, nargs=2, default=[640, 480],
                       help='Resize images to W H for MediaPipe processing')
    args = parser.parse_args()
    
    print("Creating MediaPipe version of MPI-INF-3DHP test dataset")
    print("=" * 60)
    print(f"Output path: {args.output_path}")
    print(f"Resize resolution: {args.resize_resolution}")
    print(f"Note: Coordinates will be stored in original image pixel coordinates")
    
    # Load original dataset
    original_data = load_original_dataset()
    if original_data is None:
        return
    
    # Initialize MediaPipe
    print("\nInitializing MediaPipe...")
    estimator = MediaPipe2DPoseEstimator(resize_resolution=tuple(args.resize_resolution))
    
    try:
        # Create MediaPipe dataset
        mediapipe_data = create_mediapipe_dataset(original_data, estimator, args.output_path)
        
        # Verify dataset
        verify_dataset(original_data, mediapipe_data)
        
        print(f"\n✓ Success! MediaPipe dataset saved to: {args.output_path}")
        print(f"\nCoordinate format: Original image pixel coordinates")
        print(f"- TS1-TS4: 2048x2048 pixel coordinates")
        print(f"- TS5-TS6: 1920x1080 pixel coordinates")
        print(f"\nTo use in training/evaluation:")
        print(f"1. Modify data_root in config to point to the new dataset")
        print(f"2. Or rename the file to replace the original")
        print(f"3. Run train_3dhp.py normally - it will use MediaPipe 2D poses!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.close()
        print("✓ MediaPipe estimator closed")

if __name__ == '__main__':
    main()