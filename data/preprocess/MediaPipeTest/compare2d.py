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
            min_detection_confidence=0.3,  # FIXED: Lower detection threshold
            min_tracking_confidence=0.3   # FIXED: Lower tracking threshold
        )
        self.resize_resolution = resize_resolution or (480, 360)  # FIXED: Larger default size
        # Use same mapping as working files
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
        # FIXED: Add debug counters
        self._debug_counter = 0
        self._failed_detections = 0
        self._successful_detections = 0

    def estimate_2d_pose_from_image(self, image):
        """Estimate 2D pose from image with debug information."""
        if image is None:
            return np.zeros((17, 3), dtype=np.float32)
        
        # FIXED: Better image processing
        original_shape = image.shape
        image = cv2.resize(image, self.resize_resolution, interpolation=cv2.INTER_LINEAR)  # Better quality
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        pose_2d = np.zeros((17, 3), dtype=np.float32)

        self._debug_counter += 1

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            detection_count = 0
            total_visibility = 0
            
            # FIXED: Lower threshold and better mapping
            for mp_idx, mpi_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    total_visibility += landmark.visibility
                    if landmark.visibility > 0.1:  # FIXED: Much lower threshold
                        pose_2d[mpi_idx] = [landmark.x, landmark.y, landmark.visibility]
                        detection_count += 1
            
            self._successful_detections += 1
            
            # FIXED: Debug info every 50 detections
            if self._debug_counter % 50 == 0:
                avg_visibility = total_visibility / len(self.mp_to_mpi_mapping)
                print(f"DEBUG Sample {self._debug_counter}: Detected {detection_count}/13 keypoints, avg visibility: {avg_visibility:.3f}")
                print(f"  Image shape: {original_shape} -> {image.shape}")
                print(f"  Success rate: {self._successful_detections}/{self._debug_counter} = {100*self._successful_detections/self._debug_counter:.1f}%")
                if detection_count > 0:
                    valid_poses = pose_2d[pose_2d[:, 2] > 0.1]
                    if len(valid_poses) > 0:
                        print(f"  Pose range: x=[{valid_poses[:, 0].min():.3f}, {valid_poses[:, 0].max():.3f}], y=[{valid_poses[:, 1].min():.3f}, {valid_poses[:, 1].max():.3f}]")

        else:
            self._failed_detections += 1
            # FIXED: Debug failed detections
            if self._debug_counter % 50 == 0:
                print(f"DEBUG: {self._failed_detections} failed pose detections out of {self._debug_counter} attempts ({100*self._failed_detections/self._debug_counter:.1f}%)")

        return pose_2d

    def close(self):
        self.pose.close()

# ULTRA-FAST: Global image cache to avoid repeated disk I/O
_GLOBAL_IMAGE_CACHE = {}

def load_single_center_frame(sequence_name, sample_idx, stride=9, n_frames=27):
    """ULTRA-FAST: Load only the center frame instead of all 27 frames."""
    global _GLOBAL_IMAGE_CACHE
    
    cache_key = sequence_name
    if cache_key not in _GLOBAL_IMAGE_CACHE:
        video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
        if not os.path.exists(video_path):
            print(f"WARNING: Video path not found: {video_path}")
            return None
        
        # Load all image paths once
        image_files = []
        for ext in ['*.jpg']:
            image_files.extend(glob.glob(os.path.join(video_path, ext)))
        image_files.sort()
        _GLOBAL_IMAGE_CACHE[cache_key] = image_files
        print(f"DEBUG: Cached {len(image_files)} images for sequence {sequence_name}")
    
    image_files = _GLOBAL_IMAGE_CACHE[cache_key]
    if not image_files:
        return None
    
    # ULTRA-FAST: Load only center frame
    center_frame_idx = sample_idx * stride + (n_frames - 1) // 2
    if center_frame_idx < len(image_files):
        frame = cv2.imread(image_files[center_frame_idx])
        # FIXED: Debug first few frame loads
        if sample_idx < 5:
            print(f"DEBUG: Loaded frame {center_frame_idx}/{len(image_files)} for sample {sample_idx}, shape: {frame.shape if frame is not None else 'None'}")
        return frame
    return None

def compare_mediapipe_2d_ultrafast(test_loader, estimator, args):
    """ULTRA-FAST: Compare MediaPipe 2D poses processing only center frames."""
    mpjpe_2d_sum = 0.0
    valid_2d_samples = 0
    valid_samples = 0
    processing_times = []
    zero_detection_count = 0

    for data in tqdm(test_loader, desc="Comparing 2D poses (Ultra-Fast)"):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        if torch.cuda.is_available():
            input_2D = input_2D.cuda()
        N = input_2D.size(0)

        for i in range(N):
            if args.sequence_name and seq[i] != args.sequence_name:
                continue
            if args.max_samples and valid_samples >= args.max_samples:
                break

            start_time = time.time()
            
            # ULTRA-FAST: Load only center frame
            center_frame = load_single_center_frame(seq[i], valid_samples, stride=9, n_frames=args.n_frames)
            if center_frame is None:
                print(f"WARNING: Could not load frame for sample {valid_samples}, sequence {seq[i]}")
                continue

            # ULTRA-FAST: Process only center frame
            mediapipe_2d = estimator.estimate_2d_pose_from_image(center_frame)
            
            # FIXED: Debug MediaPipe output
            valid_keypoints = np.sum(mediapipe_2d[:, 2] > 0.1)
            if valid_keypoints == 0:
                zero_detection_count += 1
            
            if valid_samples < 10 or valid_samples % 100 == 0:
                print(f"Sample {valid_samples}: MediaPipe detected {valid_keypoints}/17 keypoints")
                if valid_keypoints > 0:
                    valid_poses = mediapipe_2d[mediapipe_2d[:, 2] > 0.1]
                    print(f"  MediaPipe range: x=[{valid_poses[:, 0].min():.3f}, {valid_poses[:, 0].max():.3f}], y=[{valid_poses[:, 1].min():.3f}, {valid_poses[:, 1].max():.3f}]")
                    
                    # FIXED: Show ground truth range for comparison
                    center_idx = args.n_frames // 2 if input_2D.shape[1] > 1 else 0
                    gt_center = input_2D[i:i+1, center_idx:center_idx+1, :, :2] if input_2D.shape[1] > center_idx else input_2D[i:i+1, :1, :, :2]
                    gt_numpy = gt_center.cpu().numpy()
                    print(f"  GT range: x=[{gt_numpy[0, 0, :, 0].min():.3f}, {gt_numpy[0, 0, :, 0].max():.3f}], y=[{gt_numpy[0, 0, :, 1].min():.3f}, {gt_numpy[0, 0, :, 1].max():.3f}]")
                else:
                    print(f"  No valid MediaPipe detections for this sample")
            
            # Convert to tensor for comparison
            mediapipe_2d_tensor = torch.from_numpy(mediapipe_2d[:, :2]).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 17, 2)
            if torch.cuda.is_available():
                mediapipe_2d_tensor = mediapipe_2d_tensor.cuda()

            # ULTRA-FAST: Compare only center frame
            center_idx = args.n_frames // 2 if input_2D.shape[1] > 1 else 0
            gt_center = input_2D[i:i+1, center_idx:center_idx+1, :, :2] if input_2D.shape[1] > center_idx else input_2D[i:i+1, :1, :, :2]
            
            # FIXED: Only compute MPJPE if we have valid detections
            if mediapipe_2d_tensor.shape[-1] == gt_center.shape[-1] and valid_keypoints > 0:
                diff = mediapipe_2d_tensor - gt_center
                mpjpe_2d = torch.sqrt(torch.sum(diff ** 2, dim=-1)).mean().item()
                
                if not np.isnan(mpjpe_2d) and not np.isinf(mpjpe_2d):
                    mpjpe_2d_sum += mpjpe_2d
                    valid_2d_samples += 1

            valid_samples += 1
            processing_times.append(time.time() - start_time)

            # ULTRA-FAST: Less frequent cleanup
            if valid_samples % 20 == 0:
                gc.collect()
                avg_time = np.mean(processing_times[-20:]) if len(processing_times) >= 20 else np.mean(processing_times)
                current_mpjpe = mpjpe_2d_sum / valid_2d_samples if valid_2d_samples > 0 else 0.0
                print(f"Sample {valid_samples}: MPJPE 2D: {current_mpjpe:.4f}, Avg time: {avg_time:.3f}s, Zero detections: {zero_detection_count}/{valid_samples}")

    mpjpe_2d_avg = mpjpe_2d_sum / valid_2d_samples if valid_2d_samples > 0 else 0.0
    avg_processing_time = np.mean(processing_times) if processing_times else 0.0
    
    print(f'MPJPE 2D (MediaPipe vs GT): {mpjpe_2d_avg:.4f} (normalized units)')
    print(f'Valid samples processed: {valid_samples}')
    print(f'Samples with valid 2D poses: {valid_2d_samples}')
    print(f'Samples with zero detections: {zero_detection_count}/{valid_samples} ({100*zero_detection_count/valid_samples:.1f}%)')
    print(f'Average processing time per sample: {avg_processing_time:.3f} seconds')

    return {
        'mpjpe_2d': mpjpe_2d_avg,
        'valid_samples': valid_samples,
        'valid_2d_samples': valid_2d_samples,
        'zero_detection_count': zero_detection_count,
        'avg_processing_time': avg_processing_time
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Compare MediaPipe 2D poses with ground truth (Ultra-Fast).')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--sequence-name', type=str, default='TS1', help='Specific sequence to test (e.g., TS1)')
    parser.add_argument('--max-samples', type=int, default=100, help='Maximum samples to process')  # FIXED: Smaller default for testing
    parser.add_argument('--resize-resolution', type=int, nargs=2, default=[480, 360], help='Resize images to W H')  # FIXED: Larger default
    return parser.parse_args()

def main():
    opts = parse_args()
    args = get_config(opts.config)
    args.sequence_name = opts.sequence_name
    args.max_samples = opts.max_samples

    print("MediaPipe 2D Pose Comparison (Ultra-Fast Mode with Debug)")
    print("=" * 60)
    print(f"Config: {opts.config}")
    print(f"Sequence: {args.sequence_name}")
    print(f"Max samples: {args.max_samples}")
    print(f"Frames per sample: {args.n_frames} (processing only center frame)")
    print(f"Resizing images to: {opts.resize_resolution}")
    print("Note: Processing only center frames for speed (like train_3dhp.py)")

    os.makedirs(opts.output_dir, exist_ok=True)

    print("\nInitializing MediaPipe...")
    estimator = MediaPipe2DPoseEstimator(resize_resolution=tuple(opts.resize_resolution))

    print("Loading test dataset...")
    test_dataset = Fusion(args, train=False)
    from torch.utils.data import DataLoader
    
    # ULTRA-FAST: Optimized DataLoader settings
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1,
                             num_workers=0,  # No multiprocessing for simpler debugging
                             pin_memory=False)  # Disable pin_memory for faster startup

    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    print(f"Will process max {args.max_samples} samples")

    try:
        start_time = time.time()
        print("\nStarting ultra-fast 2D pose comparison...")
        results = compare_mediapipe_2d_ultrafast(test_loader, estimator, args)

        if results:
            elapsed_time = time.time() - start_time
            print(f"\n✓ Comparison completed in {elapsed_time:.2f} seconds!")
            print(f"Throughput: {results['valid_samples'] / elapsed_time:.2f} samples/second")
            
            # Save results
            summary_path = os.path.join(opts.output_dir, 'ultrafast_comparison_results.json')
            with open(summary_path, 'w') as f:
                results['total_time'] = elapsed_time
                results['sequence_name'] = args.sequence_name
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to: {summary_path}")

    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

    finally:
        estimator.close()
        print("✓ MediaPipe estimator closed")

if __name__ == '__main__':
    main()