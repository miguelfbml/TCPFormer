import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import glob
import gc
import json
import time

# Navigate to project root
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from data.reader.motion_dataset import Fusion
from utils.tools import get_config

class MediaPipe2DPoseEstimator:
    def __init__(self, resize_resolution=None):
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
        # OPTIMIZED: Use same mapping as working files
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
        
        # OPTIMIZED: Always resize to reduce processing time
        if self.resize_resolution:
            image = cv2.resize(image, self.resize_resolution, interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)  # Default faster size
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        pose_2d = np.zeros((17, 3), dtype=np.float32)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # OPTIMIZED: Use fixed threshold instead of computing average
            confidence_threshold = 0.3

            for mp_idx, mpi_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks) and landmarks[mp_idx].visibility > confidence_threshold:
                    pose_2d[mpi_idx] = [landmarks[mp_idx].x, landmarks[mp_idx].y, landmarks[mp_idx].visibility]

            # OPTIMIZED: Simplified head joint estimation
            if len(landmarks) > 10:
                # Head top from eyebrows
                if len(landmarks) > 5:
                    pose_2d[0] = [
                        (landmarks[2].x + landmarks[5].x) / 2.0,
                        (landmarks[2].y + landmarks[5].y) / 2.0,
                        (landmarks[2].visibility + landmarks[5].visibility) / 2.0
                    ]
                # Head from mouth
                if len(landmarks) > 10:
                    pose_2d[16] = [
                        (landmarks[9].x + landmarks[10].x) / 2.0,
                        (landmarks[9].y + landmarks[10].y) / 2.0,
                        (landmarks[9].visibility + landmarks[10].visibility) / 2.0
                    ]

            # OPTIMIZED: Faster missing joint estimation
            for missing_joint, source_joints in self.missing_joints_estimation.items():
                valid_sources = [j for j in source_joints if pose_2d[j, 2] > confidence_threshold]
                if valid_sources:
                    pose_2d[missing_joint, :2] = np.mean([pose_2d[j, :2] for j in valid_sources], axis=0)
                    pose_2d[missing_joint, 2] = np.mean([pose_2d[j, 2] for j in valid_sources]) * 0.9

            # Root joint from hips
            if pose_2d[11, 2] > confidence_threshold and pose_2d[8, 2] > confidence_threshold:
                pose_2d[0, :2] = (pose_2d[11, :2] + pose_2d[8, :2]) / 2.0
                pose_2d[0, 1] -= 0.05
                pose_2d[0, 2] = min(pose_2d[11, 2], pose_2d[8, 2])

        return pose_2d

    def close(self):
        self.pose.close()

def load_video_frames_for_sample(sequence_name, sample_idx, n_frames=27, stride=9):
    """OPTIMIZED: Load video frames with caching and reduced I/O."""
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    if not os.path.exists(video_path):
        return None

    # OPTIMIZED: Cache image file list to avoid repeated glob operations
    if not hasattr(load_video_frames_for_sample, '_image_cache'):
        load_video_frames_for_sample._image_cache = {}
    
    if sequence_name not in load_video_frames_for_sample._image_cache:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(video_path, ext)))
        image_files.sort()
        load_video_frames_for_sample._image_cache[sequence_name] = image_files
    else:
        image_files = load_video_frames_for_sample._image_cache[sequence_name]

    if not image_files:
        return None

    center_frame = sample_idx * stride + (n_frames - 1) // 2
    start_frame = max(0, center_frame - (n_frames - 1) // 2)
    end_frame = min(len(image_files), center_frame + (n_frames - 1) // 2 + 1)

    frames = []
    valid_frames = 0
    for frame_idx in range(start_frame, end_frame):
        if frame_idx < len(image_files):
            frame = cv2.imread(image_files[frame_idx])
            if frame is not None:
                valid_frames += 1
                frames.append(frame)
            else:
                frames.append(None)
        else:
            frames.append(None)

    return frames if valid_frames >= n_frames // 2 else None

def compute_2d_mpjpe(mediapipe_2d, gt_2d):
    """OPTIMIZED: Faster MPJPE computation."""
    if mediapipe_2d.shape[2] != gt_2d.shape[2]:
        return None
    diff = mediapipe_2d[:, :, :2] - gt_2d[:, :, :2]  # (N, T, J, 2)
    return torch.sqrt(torch.sum(diff ** 2, dim=-1)).mean().item()

def compare_mediapipe_2d(test_loader, estimator, args):
    """OPTIMIZED: Compare MediaPipe 2D poses with ground truth 2D poses."""
    pose_2d_comparison = {}
    mpjpe_2d_sum = 0.0
    valid_2d_samples = 0
    valid_samples = 0

    # OPTIMIZED: Process in batches and reduce memory usage
    for data in tqdm(test_loader, desc="Comparing 2D poses"):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        if torch.cuda.is_available():
            input_2D = input_2D.cuda()
        N = input_2D.size(0)

        for i in range(N):
            if args.sequence_name and seq[i] != args.sequence_name:
                continue
            if args.max_samples and valid_samples >= args.max_samples:
                break

            # OPTIMIZED: Reduce debug output frequency
            if valid_samples % 10 == 0:
                print(f"Processing sample {valid_samples + 1} from sequence {seq[i]}")
            
            frames = load_video_frames_for_sample(seq[i], valid_samples, args.n_frames, stride=9)
            if frames is None:
                if valid_samples % 50 == 0:  # Reduced warning frequency
                    print(f"Skipping sample {valid_samples + 1} from {seq[i]}: No valid frames")
                continue

            # OPTIMIZED: Process frames in smaller batches to reduce memory
            mediapipe_2d_sequence = []
            for j, frame in enumerate(frames):
                pose_2d = estimator.estimate_2d_pose_from_image(frame)
                mediapipe_2d_sequence.append(pose_2d)
                
                # OPTIMIZED: Clear frame from memory immediately
                frames[j] = None

            if len(mediapipe_2d_sequence) != args.n_frames:
                continue

            # OPTIMIZED: Convert to tensor more efficiently
            mediapipe_2d_tensor = torch.from_numpy(np.stack(mediapipe_2d_sequence, axis=0)).float().unsqueeze(0)
            if torch.cuda.is_available():
                mediapipe_2d_tensor = mediapipe_2d_tensor.cuda()

            # OPTIMIZED: Only store comparison data for first few samples to save memory
            seq_name = seq[i]
            if valid_samples < 100:  # Only store first 100 for analysis
                if seq_name not in pose_2d_comparison:
                    pose_2d_comparison[seq_name] = {}
                pose_2d_comparison[seq_name][valid_samples] = {
                    'mediapipe_2d': mediapipe_2d_tensor[0, ::9].cpu().numpy().tolist(),  # Store every 9th frame to reduce size
                    'ground_truth_2d': input_2D[i:i+1, ::9].cpu().numpy().tolist(),
                    'mpjpe_2d': None
                }

            # OPTIMIZED: Compute MPJPE only on center frame for speed
            center_idx = args.n_frames // 2
            mpjpe_2d = compute_2d_mpjpe(
                mediapipe_2d_tensor[:, center_idx:center_idx+1], 
                input_2D[i:i+1, center_idx:center_idx+1] if input_2D.shape[1] > center_idx else input_2D[i:i+1, :1]
            )
            
            if mpjpe_2d is not None:
                mpjpe_2d_sum += mpjpe_2d
                valid_2d_samples += 1
                if valid_samples < 100 and seq_name in pose_2d_comparison and valid_samples in pose_2d_comparison[seq_name]:
                    pose_2d_comparison[seq_name][valid_samples]['mpjpe_2d'] = mpjpe_2d

            valid_samples += 1

            # OPTIMIZED: More frequent memory cleanup
            if valid_samples % 5 == 0:
                del mediapipe_2d_tensor, mediapipe_2d_sequence
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # OPTIMIZED: Reduced progress updates
            if valid_samples % 50 == 0:
                current_mpjpe = mpjpe_2d_sum / valid_2d_samples if valid_2d_samples > 0 else 0.0
                print(f"Sample {valid_samples}: Current avg MPJPE 2D: {current_mpjpe:.4f}")

    mpjpe_2d_avg = mpjpe_2d_sum / valid_2d_samples if valid_2d_samples > 0 else 0.0
    print(f'MPJPE 2D (MediaPipe vs GT): {mpjpe_2d_avg:.4f} (normalized units)')
    print(f'Valid samples processed: {valid_samples}')
    print(f'Samples with valid 2D MPJPE: {valid_2d_samples}')

    return {
        'mpjpe_2d': mpjpe_2d_avg,
        'valid_samples': valid_samples,
        'valid_2d_samples': valid_2d_samples,
        'pose_2d_comparison': pose_2d_comparison
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Compare MediaPipe 2D poses with ground truth.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--sequence-name', type=str, default='TS1', help='Specific sequence to test (e.g., TS1)')
    parser.add_argument('--max-samples', type=int, default=100, help='Maximum samples to process')  # OPTIMIZED: Default limit
    parser.add_argument('--resize-resolution', type=int, nargs=2, default=[640, 480], help='Resize images to W H')  # OPTIMIZED: Default resize
    return parser.parse_args()

def main():
    opts = parse_args()
    args = get_config(opts.config)
    args.sequence_name = opts.sequence_name
    args.max_samples = opts.max_samples

    print("MediaPipe 2D Pose Comparison (Optimized)")
    print("=" * 60)
    print(f"Config: {opts.config}")
    print(f"Output directory: {opts.output_dir}")
    print(f"Sequence: {args.sequence_name}")
    print(f"Max samples: {args.max_samples}")
    print(f"Frames per sample: {args.n_frames}")
    print(f"Stride: 9")
    print(f"Resizing images to: {opts.resize_resolution}")

    os.makedirs(opts.output_dir, exist_ok=True)

    print("\nInitializing MediaPipe...")
    estimator = MediaPipe2DPoseEstimator(resize_resolution=tuple(opts.resize_resolution))

    print("Loading test dataset...")
    test_dataset = Fusion(args, train=False)
    from torch.utils.data import DataLoader
    
    # OPTIMIZED: Reduced number of workers and batch processing
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1,
                             num_workers=2,  # Reduced workers
                             pin_memory=True)

    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    print(f"Will process max {args.max_samples} samples")

    try:
        start_time = time.time()
        print("\nStarting 2D pose comparison...")
        results = compare_mediapipe_2d(test_loader, estimator, args)

        if results:
            elapsed_time = time.time() - start_time
            print(f"\n✓ Comparison completed successfully in {elapsed_time:.2f} seconds!")
            print(f"Average time per sample: {elapsed_time / results['valid_samples']:.2f} seconds")
            
            # OPTIMIZED: Only save essential data
            comparison_path = os.path.join(opts.output_dir, 'mediapipe_vs_gt_2d_summary.json')
            with open(comparison_path, 'w') as f:
                summary_data = {
                    'mpjpe_2d': results['mpjpe_2d'],
                    'valid_samples': results['valid_samples'],
                    'valid_2d_samples': results['valid_2d_samples'],
                    'sequence_name': args.sequence_name,
                    'processing_time': elapsed_time,
                    'sample_count': min(100, len(results['pose_2d_comparison'].get(args.sequence_name, {})))
                }
                json.dump(summary_data, f, indent=2)
            print(f"✓ Summary saved to: {comparison_path}")

            # OPTIMIZED: Save detailed comparison only for first few samples
            if results['pose_2d_comparison']:
                detailed_path = os.path.join(opts.output_dir, 'detailed_comparison_sample.json')
                with open(detailed_path, 'w') as f:
                    json.dump(results['pose_2d_comparison'], f, indent=2)
                print(f"✓ Detailed comparison (first 100 samples) saved to: {detailed_path}")

    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

    finally:
        estimator.close()
        print("✓ MediaPipe estimator closed")

if __name__ == '__main__':
    main()