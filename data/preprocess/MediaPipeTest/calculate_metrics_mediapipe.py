"""
Calculate metrics for TCPFormer using MediaPipe 2D pose estimation on MPI-INF-3DHP test set
This evaluates the model's performance when using MediaPipe 2D keypoints instead of ground truth 2D poses
Usage: python calculate_metrics_mediapipe.py --config configs/mpi/TCPFormer_mpi_27.yaml --checkpoint checkpoint_mpi --checkpoint-file TCPFormer_mpi_27.pth.tr

# Basic evaluation
python calculate_metrics_mediapipe.py \
    --config configs/mpi/TCPFormer_mpi_27.yaml \
    --checkpoint checkpoint_mpi \
    --checkpoint-file TCPFormer_mpi_27.pth.tr

# Evaluate specific sequence
python calculate_metrics_mediapipe.py \
    --config configs/mpi/TCPFormer_mpi_27.yaml \
    --checkpoint checkpoint_mpi \
    --checkpoint-file TCPFormer_mpi_27.pth.tr \
    --sequence-name TS1

# Test with limited samples
python calculate_metrics_mediapipe.py \
    --config configs/mpi/TCPFormer_mpi_27.yaml \
    --checkpoint checkpoint_mpi \
    --checkpoint-file TCPFormer_mpi_27.pth.tr \
    --max-samples 100
"""

import argparse
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from tqdm import tqdm
import glob
import gc

# Navigate to project root
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from data.reader.motion_dataset import Fusion
from utils.tools import get_config
from utils.learning import load_model_TCPFormer
from utils.utils_3dhp import *
from utils.data import denormalize

class MediaPipe2DPoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Lighter model for faster processing
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
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
        image = cv2.resize(image, (640, 480))  # Resize for faster processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        pose_2d = np.zeros((17, 3), dtype=np.float32)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            avg_visibility = np.mean([landmarks[i].visibility for i in self.mp_to_mpi_mapping.keys() if i < len(landmarks)])
            confidence_threshold = max(0.3, avg_visibility * 0.5)

            for mp_idx, mpi_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks) and landmarks[mp_idx].visibility > confidence_threshold:
                    pose_2d[mpi_idx] = [landmarks[mp_idx].x, landmarks[mp_idx].y, landmarks[mp_idx].visibility]

            for missing_joint, source_joints in self.missing_joints_estimation.items():
                valid_sources = [j for j in source_joints if pose_2d[j, 2] > confidence_threshold]
                if valid_sources:
                    pose_2d[missing_joint, :2] = np.mean([pose_2d[j, :2] for j in valid_sources], axis=0)
                    pose_2d[missing_joint, 2] = np.mean([pose_2d[j, 2] for j in valid_sources]) * 0.9

            if pose_2d[11, 2] > confidence_threshold and pose_2d[8, 2] > confidence_threshold:
                pose_2d[0, :2] = (pose_2d[11, :2] + pose_2d[8, :2]) / 2.0
                pose_2d[0, 1] -= 0.05  # Upward offset for root
                pose_2d[0, 2] = min(pose_2d[11, 2], pose_2d[8, 2])

        return pose_2d

    def close(self):
        self.pose.close()

def load_video_frames_for_sample(sequence_name, sample_idx, n_frames=27, stride=9):
    """Load video frames for a single sample."""
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    if not os.path.exists(video_path):
        print(f"Video frames not found at: {video_path}")
        return None

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    image_files.sort()

    if not image_files:
        print(f"No image files found in {video_path}")
        return None

    center_frame = sample_idx * stride + (n_frames - 1) // 2
    start_frame = max(0, center_frame - (n_frames - 1) // 2)
    end_frame = min(len(image_files), center_frame + (n_frames - 1) // 2 + 1)

    frames = []
    valid_frames = 0
    for frame_idx in range(start_frame, end_frame):
        frame = cv2.imread(image_files[frame_idx]) if frame_idx < len(image_files) else None
        if frame is not None:
            valid_frames += 1
        frames.append(frame)

    return frames if valid_frames >= n_frames // 2 else None

def input_augmentation_mediapipe(input_2D, model, joints_left, joints_right):
    """Apply test-time augmentation using MediaPipe 2D poses."""
    N, T, J, C = input_2D.shape
    input_2D_flip = input_2D.clone()
    input_2D_flip[..., 0] = 1.0 - input_2D_flip[..., 0]
    input_2D_flip[:, :, joints_left + joints_right, :] = input_2D_flip[:, :, joints_right + joints_left, :]

    output_3D_non_flip = model(input_2D)
    output_3D_flip = model(input_2D_flip)
    output_3D_flip[..., 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    return input_2D, output_3D

def evaluate_with_mediapipe_2d(model, test_loader, estimator, args):
    """Evaluate model using MediaPipe 2D poses iteratively."""
    model.eval()
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]
    
    data_inference = {}
    error_sum_test = AccumLoss()
    pck_results = {
        'PCK@90%_torso': 0.0, 'PCK@80%_torso': 0.0, 'PCK@70%_torso': 0.0,
        'PCK@90%_150mm': 0.0, 'PCK@80%_150mm': 0.0, 'PCK@70%_150mm': 0.0
    }
    auc_sum = 0.0
    valid_samples = 0

    for data in tqdm(test_loader, desc="Evaluating samples"):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('test', [input_2D, gt_3D, batch_cam, scale, bb_box])
        N = input_2D.size(0)

        for i in range(N):
            if args.sequence_name and seq[i] != args.sequence_name:
                continue
            if args.max_samples and valid_samples >= args.max_samples:
                break

            print(f"Processing sample {valid_samples + 1} from sequence {seq[i]}")
            frames = load_video_frames_for_sample(seq[i], i, args.n_frames, stride=9)
            if frames is None:
                print(f"Skipping sample {i} from {seq[i]}: No valid frames")
                continue

            mediapipe_2d_sequence = []
            for frame in frames:
                pose_2d = estimator.estimate_2d_pose_from_image(frame)  # (17, 3) with x, y, confidence
                mediapipe_2d_sequence.append(pose_2d)

            if len(mediapipe_2d_sequence) != args.n_frames:
                print(f"Skipping sample {i} from {seq[i]}: Incomplete frame sequence")
                continue

            mediapipe_2d_tensor = torch.from_numpy(np.stack(mediapipe_2d_sequence, axis=0)).float().unsqueeze(0)  # (1, 27, 17, 3)
            print(f"MediaPipe 2D tensor shape: {mediapipe_2d_tensor.shape}")
            if torch.cuda.is_available():
                mediapipe_2d_tensor = mediapipe_2d_tensor.cuda()
                gt_3D = gt_3D.cuda()
                scale = scale.cuda()

            out_target = gt_3D[i:i+1].clone()  # (1, 27, 17, 3)
            out_target[:, :, 14] = 0
            print(f"Ground truth shape: {out_target.shape}")

            with torch.no_grad():
                mediapipe_2d_tensor, output_3D = input_augmentation_mediapipe(
                    mediapipe_2d_tensor, model, joints_left, joints_right)
                print(f"Model output shape: {output_3D.shape}")
                output_3D = output_3D * scale[i:i+1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            pad = (args.n_frames - 1) // 2
            pred_out = output_3D[:, pad].unsqueeze(1)  # (1, 1, 17, 3)
            pred_out[..., 14, :] = 0
            pred_out = denormalize(pred_out, [seq[i]])

            pred_out_relative = pred_out - pred_out[..., 14:15, :]
            inference_out = pred_out + out_target[:, pad:pad+1, 14:15, :]
            out_target_relative = out_target[:, pad:pad+1] - out_target[:, pad:pad+1, 14:15, :]

            joint_error_test = mpjpe_cal(pred_out_relative, out_target_relative).item()
            error_sum_test.update(joint_error_test, 1)

            pred_frame = pred_out_relative[:, 0]
            gt_frame = out_target_relative[:, 0]
            torso_diameters = calculate_torso_diameter(out_target)
            print(f"pred_frame shape: {pred_frame.shape}, gt_frame shape: {gt_frame.shape}")
            batch_pck = compute_pck(pred_frame, gt_frame, torso_diameters, fixed_threshold=150.0)
            for key in pck_results:
                pck_results[key] += batch_pck[key]
            auc_sum += compute_auc(pred_frame, gt_frame)

            seq_name = seq[i]
            if seq_name in data_inference:
                data_inference[seq_name] = np.concatenate(
                    (data_inference[seq_name], inference_out.cpu().numpy().transpose(2, 1, 0)), axis=2)
            else:
                data_inference[seq_name] = inference_out.cpu().numpy().transpose(2, 1, 0)

            valid_samples += 1
            if valid_samples % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    if valid_samples == 0:
        print("No valid samples processed!")
        return None

    for seq_name in data_inference.keys():
        data_inference[seq_name] = data_inference[seq_name][:, :, None, :]

    mpjpe_avg = error_sum_test.avg
    for key in pck_results:
        pck_results[key] /= valid_samples
    auc_avg = auc_sum / valid_samples

    print(f'\n{"="*60}')
    print(f'TCPFormer Results with MediaPipe 2D Input')
    print(f'Protocol #1 Error (MPJPE): {mpjpe_avg:.2f} mm')
    for key, value in pck_results.items():
        print(f'{key}: {value*100:.2f}%')
    print(f'AUC: {auc_avg:.4f}')
    print(f'Valid samples: {valid_samples}')
    print(f'{"="*60}')

    return {
        'mpjpe': mpjpe_avg,
        'pck_results': pck_results,
        'auc': auc_avg,
        'valid_samples': valid_samples,
        'data_inference': data_inference
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, default='best_epoch.pth.tr', help="Checkpoint file name")
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--sequence-name', type=str, default=None, help='Specific sequence to test (TS1, TS2, etc.)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples to process (for testing)')
    return parser.parse_args()

def main():
    opts = parse_args()
    args = get_config(opts.config)
    args.sequence_name = opts.sequence_name
    args.max_samples = opts.max_samples
    args.batch_size = opts.batch_size

    print("TCPFormer Evaluation with MediaPipe 2D Poses")
    print("=" * 60)
    print(f"Config: {opts.config}")
    print(f"Checkpoint: {opts.checkpoint}/{opts.checkpoint_file}")
    print(f"Frames per sample: {args.n_frames}")
    print(f"Stride: 9")
    if args.sequence_name:
        print(f"Filtering for sequence: {args.sequence_name}")

    print("\nInitializing MediaPipe...")
    estimator = MediaPipe2DPoseEstimator()

    print("Loading model...")
    model = load_model_TCPFormer(args)
    checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'module.' in list(checkpoint['model'].keys())[0]:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        if 'min_mpjpe' in checkpoint:
            print(f"  Best MPJPE from training: {checkpoint['min_mpjpe']:.2f} mm")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    if torch.cuda.is_available():
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, device_ids=[0])
        model = model.cuda()
        print("✓ Model moved to GPU")

    model.eval()

    print("Loading test dataset...")
    test_dataset = Fusion(args, train=False)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=args.batch_size,
                             num_workers=2,
                             pin_memory=True)

    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

    try:
        print("\nStarting evaluation with MediaPipe 2D poses...")
        with torch.no_grad():
            results = evaluate_with_mediapipe_2d(model, test_loader, estimator, args)

        if results:
            print("\n✓ Evaluation completed successfully!")
            import json
            results_path = os.path.join(opts.checkpoint, 'mediapipe_evaluation_results.json')
            with open(results_path, 'w') as f:
                json_results = {
                    key: float(value) if isinstance(value, (np.floating, np.integer)) else value
                    for key, value in results.items()
                    if key != 'data_inference'
                }
                json.dump(json_results, f, indent=2)
            print(f"✓ Results saved to: {results_path}")

            import scipy.io as scio
            mat_path = os.path.join(opts.checkpoint, 'inference_data_mediapipe.mat')
            scio.savemat(mat_path, results['data_inference'])
            print(f"✓ Inference data saved to: {mat_path}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        estimator.close()
        print("✓ MediaPipe estimator closed")

if __name__ == '__main__':
    main()