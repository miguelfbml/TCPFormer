"""
Fast comparison of preprocessed MediaPipe 2D poses with ground truth
This uses the preprocessed MediaPipe poses from preprocess_mediapipe_2d.py

Usage:
python compare_preprocessed_2d.py --config configs/mpi/TCPFormer_mpi_27.yaml --mediapipe-dir mediapipe_2d_poses --sequence TS1
"""

import argparse
import os
import numpy as np
import torch
import json
import time
from tqdm import tqdm

# Navigate to project root
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from data.reader.motion_dataset import Fusion
from utils.tools import get_config

def load_mediapipe_poses(sequence_name, mediapipe_dir):
    """Load preprocessed MediaPipe poses."""
    np_path = os.path.join(mediapipe_dir, f'{sequence_name}_mediapipe_2d.npz')
    json_path = os.path.join(mediapipe_dir, f'{sequence_name}_mediapipe_2d_metadata.json')
    
    if not os.path.exists(np_path):
        print(f"MediaPipe poses not found: {np_path}")
        return None, None
    
    # Load poses
    data = np.load(np_path)
    poses_2d = data['poses_2d']  # Shape: (num_frames, 17, 3)
    
    # Load metadata
    metadata = None
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    
    print(f"✓ Loaded MediaPipe poses for {sequence_name}: {poses_2d.shape}")
    return poses_2d, metadata

def sample_mediapipe_poses_for_gt(mediapipe_poses, sample_idx, n_frames=27, stride=9):
    """Sample MediaPipe poses to match ground truth sampling pattern."""
    # Calculate frame indices that correspond to the GT sample
    center_frame = sample_idx * stride + (n_frames - 1) // 2
    start_frame = max(0, center_frame - (n_frames - 1) // 2)
    end_frame = min(len(mediapipe_poses), center_frame + (n_frames - 1) // 2 + 1)
    
    # Extract the frames
    if end_frame - start_frame < n_frames:
        # Pad if necessary
        sampled_poses = np.zeros((n_frames, 17, 3), dtype=np.float32)
        actual_frames = mediapipe_poses[start_frame:end_frame]
        sampled_poses[:len(actual_frames)] = actual_frames
    else:
        sampled_poses = mediapipe_poses[start_frame:end_frame]
    
    return sampled_poses

def compare_preprocessed_2d(test_loader, mediapipe_dir, args):
    """Compare preprocessed MediaPipe 2D poses with ground truth."""
    print(f"Comparing preprocessed MediaPipe poses from: {mediapipe_dir}")
    
    # Load MediaPipe poses for the target sequence
    if args.sequence_name:
        target_sequence = args.sequence_name
    else:
        # Try to determine from first sample
        sample_data = next(iter(test_loader))
        target_sequence = sample_data[3][0]  # seq[0]
    
    mediapipe_poses, metadata = load_mediapipe_poses(target_sequence, mediapipe_dir)
    if mediapipe_poses is None:
        print(f"Could not load MediaPipe poses for sequence {target_sequence}")
        return None
    
    print(f"Loaded {len(mediapipe_poses)} MediaPipe poses for sequence {target_sequence}")
    
    # Comparison metrics
    mpjpe_2d_sum = 0.0
    valid_2d_samples = 0
    valid_samples = 0
    zero_detection_count = 0
    
    start_time = time.time()
    
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

            # Sample MediaPipe poses to match this GT sample
            sample_mediapipe = sample_mediapipe_poses_for_gt(
                mediapipe_poses, valid_samples, args.n_frames, stride=9)
            
            # Get center frame (like GT processing)
            center_idx = args.n_frames // 2
            mediapipe_center = sample_mediapipe[center_idx]  # (17, 3)
            
            # Check if we have valid detections
            valid_keypoints = np.sum(mediapipe_center[:, 2] > 0.1)
            if valid_keypoints == 0:
                zero_detection_count += 1
            
            # Convert MediaPipe [0,1] to match GT coordinate system
            mediapipe_2d_normalized = mediapipe_center.copy()
            mediapipe_2d_normalized[:, 0] = mediapipe_2d_normalized[:, 0] * 2.0 - 1.0  # x: [0,1] -> [-1,1]
            mediapipe_2d_normalized[:, 1] = mediapipe_2d_normalized[:, 1] * 2.0 - 1.0  # y: [0,1] -> [-1,1]
            
            # Convert to tensor for comparison
            mediapipe_2d_tensor = torch.from_numpy(mediapipe_2d_normalized[:, :2]).float().unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                mediapipe_2d_tensor = mediapipe_2d_tensor.cuda()

            # Get GT center frame
            gt_center_idx = args.n_frames // 2 if input_2D.shape[1] > 1 else 0
            gt_center = input_2D[i:i+1, gt_center_idx:gt_center_idx+1, :, :2] if input_2D.shape[1] > gt_center_idx else input_2D[i:i+1, :1, :, :2]
            
            # Compute MPJPE if we have valid detections
            if mediapipe_2d_tensor.shape[-1] == gt_center.shape[-1] and valid_keypoints > 0:
                diff = mediapipe_2d_tensor - gt_center
                mpjpe_2d = torch.sqrt(torch.sum(diff ** 2, dim=-1)).mean().item()
                
                if not np.isnan(mpjpe_2d) and not np.isinf(mpjpe_2d):
                    mpjpe_2d_sum += mpjpe_2d
                    valid_2d_samples += 1

            valid_samples += 1
            
            # Progress update
            if valid_samples % 100 == 0:
                current_mpjpe = mpjpe_2d_sum / valid_2d_samples if valid_2d_samples > 0 else 0.0
                print(f"Sample {valid_samples}: MPJPE 2D: {current_mpjpe:.4f}, Valid samples: {valid_2d_samples}, Zero detections: {zero_detection_count}")

    elapsed_time = time.time() - start_time
    
    # Final results
    mpjpe_2d_avg = mpjpe_2d_sum / valid_2d_samples if valid_2d_samples > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"2D Pose Comparison Results")
    print(f"{'='*60}")
    print(f'MPJPE 2D (MediaPipe vs GT): {mpjpe_2d_avg:.4f} (normalized units)')
    print(f'Valid samples processed: {valid_samples}')
    print(f'Samples with valid 2D poses: {valid_2d_samples}')
    print(f'Samples with zero detections: {zero_detection_count}/{valid_samples} ({100*zero_detection_count/valid_samples:.1f}%)')
    print(f'Processing time: {elapsed_time:.2f}s ({valid_samples/elapsed_time:.1f} samples/sec)')
    
    return {
        'mpjpe_2d': mpjpe_2d_avg,
        'valid_samples': valid_samples,
        'valid_2d_samples': valid_2d_samples,
        'zero_detection_count': zero_detection_count,
        'processing_time': elapsed_time,
        'sequence_name': target_sequence
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Compare preprocessed MediaPipe 2D poses with ground truth')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--mediapipe-dir', type=str, default='mediapipe_2d_poses',
                       help='Directory containing preprocessed MediaPipe poses')
    parser.add_argument('--sequence', type=str, default='TS1',
                       help='Sequence to compare (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (None for all)')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Directory to save comparison results')
    return parser.parse_args()

def main():
    args = parse_args()
    config_args = get_config(args.config)
    config_args.sequence_name = args.sequence
    config_args.max_samples = args.max_samples

    print("Fast 2D Pose Comparison using Preprocessed MediaPipe Poses")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"MediaPipe poses directory: {args.mediapipe_dir}")
    print(f"Target sequence: {args.sequence}")
    print(f"Max samples: {args.max_samples or 'All'}")

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = Fusion(config_args, train=False)
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1,
                             num_workers=2,
                             pin_memory=True)

    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

    try:
        # Run comparison
        results = compare_preprocessed_2d(test_loader, args.mediapipe_dir, config_args)
        
        if results:
            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            results_path = os.path.join(args.output_dir, f'comparison_results_{args.sequence}.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to: {results_path}")

    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()