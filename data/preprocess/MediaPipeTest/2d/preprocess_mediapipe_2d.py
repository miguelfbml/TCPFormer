"""
Preprocess MediaPipe 2D poses for MPI-INF-3DHP test sequences and save to files
This creates MediaPipe equivalent of ground truth 2D poses for efficient comparison

IMPORTANT: This creates MediaPipe 2D poses in the EXACT same format as the ground truth
2D poses used in train_3dhp.py, so they can be used as drop-in replacements.

Usage:
# Run from project root directory:
cd ~/TCPFormerForked
python data/preprocess/MediaPipeTest/2d/preprocess_mediapipe_2d.py --sequence TS1 --output-dir mediapipe_2d_poses
python data/preprocess/MediaPipeTest/2d/preprocess_mediapipe_2d.py --all-sequences --output-dir mediapipe_2d_poses
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

# Try importing with error handling
try:
    from data.reader.motion_dataset import Fusion
    from utils.tools import get_config
    print("✓ Successfully imported project modules")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}")
    print("Please run this script from the project root directory:")
    print("cd ~/TCPFormerForked")
    print("python data/preprocess/MediaPipeTest/2d/preprocess_mediapipe_2d.py --all-sequences")
    sys.exit(1)

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
        
        # MediaPipe to MPI-INF-3DHP joint mapping (same as working files)
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
        
        # Missing joint estimation (same as working files)
        self.missing_joints_estimation = {
            0: [11, 8],  # root from hips
            1: [5, 2],   # neck from shoulders
            14: [11, 8], # hip from left/right hips
            15: [14, 1], # spine from hip and neck
        }

    def estimate_2d_pose_from_image(self, image):
        """
        Estimate 2D pose from image, return coordinates in EXACT same format as train_3dhp.py
        Returns: (17, 3) array with [x, y, confidence] where x,y are in [-1, 1] range
        """
        if image is None:
            return np.zeros((17, 3), dtype=np.float32)
        
        try:
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
                        # CRITICAL: Convert [0,1] to [-1,1] range to match train_3dhp.py format
                        x_norm = landmarks[mp_idx].x * 2.0 - 1.0  # [0,1] -> [-1,1]
                        y_norm = landmarks[mp_idx].y * 2.0 - 1.0  # [0,1] -> [-1,1]
                        pose_2d[mpi_idx] = [x_norm, y_norm, landmarks[mp_idx].visibility]

                # Estimate missing joints
                for missing_joint, source_joints in self.missing_joints_estimation.items():
                    valid_sources = [j for j in source_joints if pose_2d[j, 2] > confidence_threshold]
                    if valid_sources:
                        pose_2d[missing_joint, :2] = np.mean([pose_2d[j, :2] for j in valid_sources], axis=0)
                        pose_2d[missing_joint, 2] = np.mean([pose_2d[j, 2] for j in valid_sources]) * 0.9

                # Root joint from hips
                if pose_2d[11, 2] > confidence_threshold and pose_2d[8, 2] > confidence_threshold:
                    pose_2d[0, :2] = (pose_2d[11, :2] + pose_2d[8, :2]) / 2.0
                    pose_2d[0, 1] -= 0.1  # Upward offset (in [-1,1] coordinates)
                    pose_2d[0, 2] = min(pose_2d[11, 2], pose_2d[8, 2])

            return pose_2d
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.zeros((17, 3), dtype=np.float32)

    def close(self):
        if hasattr(self, 'pose'):
            self.pose.close()

def load_ground_truth_dataset_structure():
    """Load the actual dataset to understand the exact structure used in train_3dhp.py"""
    try:
        # Create a minimal config to load the dataset
        class DatasetArgs:
            def __init__(self):
                self.data_root = 'data/motion3d/'
                self.n_frames = 27
                self.stride = 9
                self.flip = False
                self.test_augmentation = False
                self.data_augmentation = False
                self.reverse_augmentation = False
                self.out_all = 1
                self.test_batch_size = 1

        dataset_args = DatasetArgs()
        dataset = Fusion(dataset_args, train=False)
        
        # Get one sample to understand the structure
        if len(dataset) > 0:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[0]
            print(f"✓ Dataset structure analysis:")
            print(f"  input_2D shape: {input_2D.shape}")
            print(f"  input_2D dtype: {input_2D.dtype}")
            print(f"  input_2D range: [{input_2D.min():.3f}, {input_2D.max():.3f}]")
            print(f"  gt_3D shape: {gt_3D.shape}")
            print(f"  sequence: {seq}")
            print(f"  scale: {scale}")
            return dataset, dataset_args
        else:
            print("ERROR: Dataset is empty")
            return None, None
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None, None

def map_mediapipe_to_dataset_samples(sequence_name, mediapipe_poses, dataset, dataset_args):
    """
    Map MediaPipe poses to match the exact sampling pattern used by the dataset.
    This creates sequences that can be used as drop-in replacements for ground truth 2D.
    """
    print(f"\nMapping MediaPipe poses to dataset format for sequence: {sequence_name}")
    
    # Find all samples from this sequence in the dataset
    sequence_samples = []
    sample_indices = []
    
    print("Scanning dataset for matching samples...")
    for i in tqdm(range(len(dataset)), desc="Scanning dataset"):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            current_seq_name = seq[0] if isinstance(seq, (list, tuple)) else str(seq)
            
            if current_seq_name == sequence_name:
                sample_indices.append(i)
                # Store the sample structure for reference
                sequence_samples.append({
                    'sample_idx': i,
                    'input_2D_shape': input_2D.shape,
                    'gt_3D_shape': gt_3D.shape,
                    'scale': scale,
                    'bb_box': bb_box
                })
                
            if len(sequence_samples) >= 500:  # Limit for processing
                break
                
        except Exception as e:
            continue
    
    print(f"Found {len(sequence_samples)} samples for sequence {sequence_name}")
    
    if not sequence_samples:
        return None
    
    # Create MediaPipe equivalents for each sample
    mediapipe_dataset_samples = []
    n_frames = dataset_args.n_frames
    stride = dataset_args.stride
    
    print(f"Creating MediaPipe dataset samples (n_frames={n_frames}, stride={stride})...")
    
    for sample_info in tqdm(sequence_samples, desc="Processing samples"):
        sample_idx = sample_info['sample_idx']
        
        # Calculate the frame range for this sample (same logic as dataset)
        center_frame = sample_idx * stride + (n_frames - 1) // 2
        start_frame = max(0, center_frame - (n_frames - 1) // 2)
        end_frame = min(len(mediapipe_poses), center_frame + (n_frames - 1) // 2 + 1)
        
        # Extract MediaPipe frames for this sample
        if end_frame - start_frame >= n_frames:
            mediapipe_sequence = mediapipe_poses[start_frame:start_frame + n_frames]  # (T, 17, 3)
            
            # Convert to the exact format expected by train_3dhp.py
            mediapipe_sequence = mediapipe_sequence.transpose(1, 0, 2)  # (17, T, 3)
            
            mediapipe_dataset_samples.append({
                'poses_2d': mediapipe_sequence,
                'sample_idx': sample_idx,
                'frame_range': (start_frame, start_frame + n_frames),
                'original_shape': sample_info['input_2D_shape'],
                'scale': sample_info['scale'],
                'bb_box': sample_info['bb_box']
            })
    
    print(f"Created {len(mediapipe_dataset_samples)} MediaPipe dataset samples")
    
    return mediapipe_dataset_samples

def load_sequence_images(sequence_name):
    """Load all images for a sequence."""
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video path not found: {video_path}")
        # Try alternative paths
        alternative_paths = [
            f'/nas-ctm01/datasets/public/mpi_inf_3dhp/{sequence_name}/imageSequence',
            f'/nas-ctm01/datasets/public/mpi_inf_3dhp/test/{sequence_name}/imageSequence'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                video_path = alt_path
                print(f"Found alternative path: {video_path}")
                break
        else:
            return None, None
    
    # Load image file paths
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    
    image_files.sort()  # Sort numerically: 000001.jpg, 000002.jpg, etc.
    
    if not image_files:
        print(f"ERROR: No image files found in {video_path}")
        print(f"Directory contents: {os.listdir(video_path) if os.path.exists(video_path) else 'Directory does not exist'}")
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
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 50
    
    print(f"Processing {len(image_files)} images in batches of {batch_size}...")
    
    start_time = time.time()
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
            
            # Get MediaPipe 2D pose (already in [-1,1] format)
            pose_2d = estimator.estimate_2d_pose_from_image(image)
            batch_poses.append(pose_2d)
        
        all_poses_2d.extend(batch_poses)
        
        # Clear memory
        del batch_poses
        gc.collect()
        
        # Progress update
        if batch_start % (batch_size * 10) == 0:
            valid_poses = sum(1 for p in all_poses_2d if np.sum(p[:, 2] > 0.1) > 5)
            elapsed = time.time() - start_time
            fps = len(all_poses_2d) / elapsed if elapsed > 0 else 0
            print(f"  Processed {len(all_poses_2d)}/{len(image_files)} images, {valid_poses} with valid poses, {fps:.1f} fps")
    
    # Convert to numpy array: (num_frames, 17, 3)
    sequence_poses = np.stack(all_poses_2d, axis=0)
    
    # Statistics
    valid_frames = np.sum([np.sum(pose[:, 2] > 0.1) > 5 for pose in sequence_poses])
    avg_confidence = np.mean(sequence_poses[:, :, 2])
    total_time = time.time() - start_time
    
    print(f"✓ Sequence {sequence_name} completed:")
    print(f"  Total frames: {len(sequence_poses)}")
    print(f"  Valid frames (>5 joints): {valid_frames}/{len(sequence_poses)} ({100*valid_frames/len(sequence_poses):.1f}%)")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Coordinate range: X[{np.min(sequence_poses[:, :, 0]):.3f}, {np.max(sequence_poses[:, :, 0]):.3f}], Y[{np.min(sequence_poses[:, :, 1]):.3f}, {np.max(sequence_poses[:, :, 1]):.3f}]")
    print(f"  Processing time: {total_time:.1f}s ({len(sequence_poses)/total_time:.1f} fps)")
    
    return sequence_poses

def save_mediapipe_poses_train_format(sequence_name, poses_2d, dataset_samples, output_dir, args):
    """Save MediaPipe poses in the exact format used by train_3dhp.py"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw MediaPipe poses (for analysis)
    raw_path = os.path.join(output_dir, f'{sequence_name}_mediapipe_2d_raw.npz')
    np.savez_compressed(raw_path, 
                       poses_2d=poses_2d,
                       sequence_name=sequence_name,
                       num_frames=len(poses_2d))
    
    # Save dataset-compatible samples (main output)
    dataset_path = os.path.join(output_dir, f'{sequence_name}_mediapipe_2d_dataset.npz')
    
    # Prepare data in dataset format
    dataset_data = {
        'sequence_name': sequence_name,
        'num_samples': len(dataset_samples),
        'n_frames': args.n_frames if hasattr(args, 'n_frames') else 27,
        'stride': args.stride if hasattr(args, 'stride') else 9
    }
    
    # Store each sample
    for i, sample in enumerate(dataset_samples):
        dataset_data[f'sample_{i:04d}'] = {
            'poses_2d': sample['poses_2d'],  # (17, T, 3) format
            'sample_idx': sample['sample_idx'],
            'frame_range': sample['frame_range'],
            'original_shape': sample['original_shape']
        }
    
    np.savez_compressed(dataset_path, **dataset_data)
    
    # Save metadata as JSON
    metadata = {
        'sequence_name': sequence_name,
        'total_frames': len(poses_2d),
        'dataset_samples': len(dataset_samples),
        'pose_shape_raw': list(poses_2d.shape),
        'valid_frames': int(np.sum([np.sum(pose[:, 2] > 0.1) > 5 for pose in poses_2d])),
        'avg_confidence': float(np.mean(poses_2d[:, :, 2])),
        'coordinate_system': 'normalized_[-1,1]_like_train_3dhp',
        'coordinate_range': {
            'x_min': float(np.min(poses_2d[:, :, 0])),
            'x_max': float(np.max(poses_2d[:, :, 0])),
            'y_min': float(np.min(poses_2d[:, :, 1])),
            'y_max': float(np.max(poses_2d[:, :, 1]))
        },
        'joint_order_mpi': [
            'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
            'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
            'RShoulder', 'RElbow', 'RWrist'
        ],
        'usage_note': 'Use sample_XXXX data for drop-in replacement of ground truth 2D poses in train_3dhp.py',
        'processing_params': {
            'resize_resolution': args.resize_resolution,
            'confidence_threshold': 0.3,
            'batch_size': getattr(args, 'batch_size', 50)
        }
    }
    
    json_path = os.path.join(output_dir, f'{sequence_name}_mediapipe_2d_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved MediaPipe poses in train_3dhp.py compatible format:")
    print(f"  Raw poses: {raw_path}")
    print(f"  Dataset format: {dataset_path}")
    print(f"  Metadata: {json_path}")
    print(f"  Dataset samples: {len(dataset_samples)}")
    
    return dataset_path, json_path

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
    
    # Load dataset structure first
    dataset, dataset_args = load_ground_truth_dataset_structure()
    if dataset is None:
        print("ERROR: Could not load dataset structure")
        return None
    
    # Store dataset parameters
    args.n_frames = dataset_args.n_frames
    args.stride = dataset_args.stride
    
    print(f"Using dataset parameters: n_frames={args.n_frames}, stride={args.stride}")
    
    results = {}
    total_start_time = time.time()
    
    for seq_name in sequences_to_process:
        seq_start_time = time.time()
        
        try:
            print(f"\n{'='*60}")
            print(f"Processing sequence: {seq_name}")
            print(f"{'='*60}")
            
            # Process sequence with MediaPipe
            poses_2d = process_sequence_mediapipe(seq_name, estimator, args)
            
            if poses_2d is not None:
                # Map to dataset format
                dataset_samples = map_mediapipe_to_dataset_samples(seq_name, poses_2d, dataset, dataset_args)
                
                if dataset_samples:
                    # Save poses
                    dataset_path, json_path = save_mediapipe_poses_train_format(
                        seq_name, poses_2d, dataset_samples, args.output_dir, args)
                    
                    seq_time = time.time() - seq_start_time
                    results[seq_name] = {
                        'success': True,
                        'num_frames': len(poses_2d),
                        'num_dataset_samples': len(dataset_samples),
                        'processing_time': seq_time,
                        'dataset_path': dataset_path,
                        'json_path': json_path
                    }
                    
                    print(f"✓ {seq_name} completed in {seq_time:.1f}s")
                    print(f"  {len(poses_2d)} frames -> {len(dataset_samples)} dataset samples")
                else:
                    results[seq_name] = {'success': False, 'error': 'Could not map to dataset format'}
                    print(f"✗ {seq_name} failed: Could not map to dataset format")
            else:
                results[seq_name] = {'success': False, 'error': 'Could not process images'}
                print(f"✗ {seq_name} failed: Could not process images")
        
        except Exception as e:
            results[seq_name] = {'success': False, 'error': str(e)}
            print(f"✗ {seq_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up
        if 'poses_2d' in locals():
            del poses_2d
        if 'dataset_samples' in locals():
            del dataset_samples
        gc.collect()
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"MediaPipe 2D Pose Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    
    successful_results = {k: v for k, v in results.items() if v['success']}
    failed_results = {k: v for k, v in results.items() if not v['success']}
    
    print(f"Sequences processed successfully: {len(successful_results)}/{len(sequences_to_process)}")
    
    if successful_results:
        total_frames = sum(r['num_frames'] for r in successful_results.values())
        total_samples = sum(r['num_dataset_samples'] for r in successful_results.values())
        print(f"Total frames processed: {total_frames}")
        print(f"Total dataset samples created: {total_samples}")
        print(f"Average processing speed: {total_frames/total_time:.1f} fps")
        
        print(f"\n✓ MediaPipe poses saved in train_3dhp.py compatible format!")
        print(f"You can now use these as drop-in replacements for ground truth 2D poses.")
    
    if failed_results:
        print(f"\nFailed sequences:")
        for seq_name, result in failed_results.items():
            print(f"  {seq_name}: {result.get('error', 'Unknown error')}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'results': results,
            'total_time': total_time,
            'sequences_processed': list(successful_results.keys()),
            'failed_sequences': list(failed_results.keys()),
            'dataset_params': {
                'n_frames': args.n_frames,
                'stride': args.stride
            },
            'args': vars(args)
        }, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_path}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess MediaPipe 2D poses for MPI-INF-3DHP test sequences in train_3dhp.py compatible format')
    parser.add_argument('--sequence', type=str, default=None, 
                       help='Specific sequence to process (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--all-sequences', action='store_true',
                       help='Process all test sequences (TS1-TS6)')
    parser.add_argument('--output-dir', type=str, default='mediapipe_2d_poses_train_format',
                       help='Directory to save MediaPipe poses in train_3dhp.py compatible format')
    parser.add_argument('--resize-resolution', type=int, nargs=2, default=[640, 480],
                       help='Resize images to W H for MediaPipe processing')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of images to process in each batch')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("MediaPipe 2D Pose Preprocessing for MPI-INF-3DHP Test Set")
    print("CREATING POSES IN EXACT FORMAT AS train_3dhp.py")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Resize resolution: {args.resize_resolution}")
    print(f"Batch size: {args.batch_size}")
    
    if args.all_sequences:
        print("Processing: All sequences (TS1-TS6)")
    elif args.sequence:
        print(f"Processing: {args.sequence}")
    else:
        print("Processing: TS1 (default)")
    
    print("\nIMPORTANT: Output will be compatible with train_3dhp.py format!")
    print("You can use the generated poses as drop-in replacements for ground truth 2D poses.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize MediaPipe
    print("\nInitializing MediaPipe...")
    estimator = MediaPipe2DPoseEstimator(resize_resolution=tuple(args.resize_resolution))
    
    try:
        # Process sequences
        results = process_all_sequences(estimator, args)
        
        print(f"\n✓ Preprocessing completed!")
        print(f"Results saved to: {os.path.abspath(args.output_dir)}")
        print(f"\nTo use these poses in training:")
        print(f"1. Load the dataset format files (*_mediapipe_2d_dataset.npz)")
        print(f"2. Use sample_XXXX data as input_2D replacement in train_3dhp.py")
        print(f"3. The poses are already in [-1,1] coordinate range like ground truth")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.close()
        print("✓ MediaPipe estimator closed")

if __name__ == '__main__':
    main()