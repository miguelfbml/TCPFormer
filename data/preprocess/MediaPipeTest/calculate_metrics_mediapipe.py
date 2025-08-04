import argparse
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
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
from utils.learning import load_model_TCPFormer
from utils.utils_3dhp import *
from utils.data import denormalize

class MediaPipe2DPoseEstimator:
    def __init__(self, resize_resolution=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.resize_resolution = resize_resolution
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
        
        # Optional resizing
        if self.resize_resolution:
            image = cv2.resize(image, self.resize_resolution, interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
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

            if len(landmarks) > 10:
                if len(landmarks) > 5:
                    left_eyebrow_inner = landmarks[2] if len(landmarks) > 2 else landmarks[0]
                    right_eyebrow_inner = landmarks[5] if len(landmarks) > 5 else landmarks[0]
                    pose_2d[0] = [
                        (left_eyebrow_inner.x + right_eyebrow_inner.x) / 2.0,
                        (left_eyebrow_inner.y + right_eyebrow_inner.y) / 2.0,
                        (left_eyebrow_inner.visibility + right_eyebrow_inner.visibility) / 2.0
                    ]
                if len(landmarks) > 10:
                    mouth_left = landmarks[9] if len(landmarks) > 9 else landmarks[0]
                    mouth_right = landmarks[10] if len(landmarks) > 10 else landmarks[0]
                    pose_2d[16] = [
                        (mouth_left.x + mouth_right.x) / 2.0,
                        (mouth_left.y + mouth_right.y) / 2.0,
                        (mouth_left.visibility + mouth_right.visibility) / 2.0
                    ]

            for missing_joint, source_joints in self.missing_joints_estimation.items():
                valid_sources = [j for j in source_joints if pose_2d[j, 2] > confidence_threshold]
                if valid_sources:
                    pose_2d[missing_joint, :2] = np.mean([pose_2d[j, :2] for j in valid_sources], axis=0)
                    pose_2d[missing_joint, 2] = np.mean([pose_2d[j, 2] for j in valid_sources]) * 0.9

            if pose_2d[11, 2] > confidence_threshold and pose_2d[8, 2] > confidence_threshold:
                pose_2d[0, :2] = (pose_2d[11, :2] + pose_2d[8, :2]) / 2.0
                pose_2d[0, 1] -= 0.05
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

    print(f"Found {len(image_files)} image files in {video_path}")
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

def input_augmentation_mediapipe(input_2D, model, joints_left, joints_right, use_augmentation=True):
    """Apply test-time augmentation using MediaPipe 2D poses."""
    N, T, J, C = input_2D.shape
    if not use_augmentation:
        return input_2D, model(input_2D)

    input_2D_flip = input_2D.clone()
    input_2D_flip[..., 0] = 1.0 - input_2D_flip[..., 0]
    input_2D_flip[:, :, joints_left + joints_right, :] = input_2D_flip[:, :, joints_right + joints_left, :]

    output_3D_non_flip = model(input_2D)
    output_3D_flip = model(input_2D_flip)
    output_3D_flip[..., 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    return input_2D, output_3D

def compute_2d_mpjpe(mediapipe_2d, gt_2d):
    """Compute MPJPE between MediaPipe and ground truth 2D poses."""
    if mediapipe_2d.shape[2] != gt_2d.shape[2]:
        print(f"Warning: Joint dimension mismatch - MediaPipe: {mediapipe_2d.shape[2]}, Ground Truth: {gt_2d.shape[2]}")
        return None
    diff = mediapipe_2d[:, :, :2] - gt_2d[:, :, :2]  # (N, T, J, 2)
    return torch.sqrt(torch.sum(diff ** 2, dim=-1)).mean().item()

def evaluate_with_mediapipe_2d(model, test_loader, estimator, args):
    """Evaluate model using MediaPipe 2D poses iteratively and compare with ground truth 2D."""
    model.eval()
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]
    
    data_inference = {}
    pose_2d_comparison = {}
    error_sum_test = AccumLoss()
    pck_results = {
        'PCK@90%_torso': 0.0, 'PCK@80%_torso': 0.0, 'PCK@70%_torso': 0.0,
        'PCK@90%_150mm': 0.0, 'PCK@80%_150mm': 0.0, 'PCK@70%_150mm': 0.0
    }
    auc_sum = 0.0
    mpjpe_2d_sum = 0.0
    valid_2d_samples = 0
    valid_samples = 0

    for data in tqdm(test_loader, desc="Evaluating samples"):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('test', [input_2D, gt_3D, batch_cam, scale, bb_box])
        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, 17, 3)
        out_target[:, :, 14] = 0
        gt_3D_original = gt_3D.view(N, -1, 17, 3).type(torch.cuda.FloatTensor)

        for i in range(N):
            if args.sequence_name and seq[i] != args.sequence_name:
                continue
            if args.max_samples and valid_samples >= args.max_samples:
                break

            start_time = time.time()
            print(f"Processing sample {valid_samples + 1} from sequence {seq[i]}")
            frames = load_video_frames_for_sample(seq[i], valid_samples, args.n_frames, stride=9)
            if frames is None:
                print(f"Skipping sample {valid_samples + 1} from {seq[i]}: No valid frames")
                continue

            mediapipe_2d_sequence = []
            for frame in frames:
                pose_2d = estimator.estimate_2d_pose_from_image(frame)
                mediapipe_2d_sequence.append(pose_2d)

            if len(mediapipe_2d_sequence) != args.n_frames:
                print(f"Skipping sample {valid_samples + 1} from {seq[i]}: Incomplete frame sequence")
                continue

            mediapipe_2d_tensor = torch.from_numpy(np.stack(mediapipe_2d_sequence, axis=0)).float().unsqueeze(0)
            
            if torch.cuda.is_available():
                mediapipe_2d_tensor = mediapipe_2d_tensor.cuda()

            print(f"MediaPipe 2D shape: {mediapipe_2d_tensor.shape}")
            print(f"Ground Truth 2D shape: {input_2D[i:i+1].shape}")

            seq_name = seq[i]
            if seq_name not in pose_2d_comparison:
                pose_2d_comparison[seq_name] = {}
            pose_2d_comparison[seq_name][valid_samples] = {
                'mediapipe_2d': mediapipe_2d_tensor[0].cpu().numpy().tolist(),
                'ground_truth_2d': input_2D[i:i+1].cpu().numpy().tolist()
            }

            mpjpe_2d = compute_2d_mpjpe(mediapipe_2d_tensor, input_2D[i:i+1])
            if mpjpe_2d is not None:
                mpjpe_2d_sum += mpjpe_2d
                valid_2d_samples += 1

            with torch.no_grad():
                mediapipe_2d_tensor, output_3D = input_augmentation_mediapipe(
                    mediapipe_2d_tensor, model, joints_left, joints_right, use_augmentation=args.use_augmentation)

                output_3D = output_3D * scale[i:i+1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
                
                pad = (args.n_frames - 1) // 2
                pred_out = output_3D[:, pad].unsqueeze(1)

                pred_out[..., 14, :] = 0
                pred_out = denormalize(pred_out, [seq[i]])

                cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
                pred_out_np = pred_out.cpu().numpy()
                pred_out_np = pred_out_np @ cam2real
                pred_out = torch.from_numpy(pred_out_np).to(pred_out.device)

                pred_out = pred_out - pred_out[..., 14:15, :]
                inference_out = pred_out + out_target[i:i+1, :, 14:15, :]
                out_target_sample = out_target[i:i+1] - out_target[i:i+1, :, 14:15, :]

                joint_error_test = mpjpe_cal(pred_out, out_target_sample).item()
                error_sum_test.update(joint_error_test * 1, 1)

                torso_diameters = calculate_torso_diameter(gt_3D_original[i:i+1])

                pred_frame = pred_out[:, 0]
                gt_frame = out_target_sample[:, 0]
                
                batch_pck = compute_pck(pred_frame, gt_frame, torso_diameters, fixed_threshold=150.0)
                for key in pck_results:
                    pck_results[key] += batch_pck[key] * 1

                auc = compute_auc(pred_frame, gt_frame)
                auc_sum += auc * 1

                valid_samples += 1

                seq_name = seq[i]
                inference_data = inference_out[0].permute(2, 1, 0).cpu().numpy()
                
                if seq_name in data_inference:
                    data_inference[seq_name] = np.concatenate(
                        (data_inference[seq_name], inference_data), axis=2)
                else:
                    data_inference[seq_name] = inference_data

                if valid_samples % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            print(f"Sample {valid_samples} processing time: {time.time() - start_time:.2f} seconds")

    if valid_samples == 0:
        print("No valid samples processed!")
        return None

    for seq_name in data_inference.keys():
        data_inference[seq_name] = data_inference[seq_name][:, :, None, :]

    mpjpe_avg = error_sum_test.avg
    mpjpe_2d_avg = mpjpe_2d_sum / valid_2d_samples if valid_2d_samples > 0 else 0.0
    for key in pck_results:
        pck_results[key] /= valid_samples if valid_samples > 0 else 1
    auc_avg = auc_sum / valid_samples if valid_samples > 0 else 0.0

    print(f'Protocol #1 Error (MPJPE 3D): {mpjpe_avg:.2f} mm')
    print(f'MPJPE 2D (MediaPipe vs GT): {mpjpe_2d_avg:.4f} (normalized units)')
    for key, value in pck_results.items():
        print(f'{key}: {value*100:.2f}%')
    print(f'AUC: {auc_avg:.4f}')

    return {
        'mpjpe': mpjpe_avg,
        'mpjpe_2d': mpjpe_2d_avg,
        'pck_results': pck_results,
        'auc': auc_avg,
        'valid_samples': valid_samples,
        'data_inference': data_inference,
        'pose_2d_comparison': pose_2d_comparison
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, default='best_epoch.pth.tr', help="Checkpoint file name")
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--sequence-name', type=str, default=None, help='Specific sequence to test (TS1, TS2, etc.)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples to process (for testing)')
    parser.add_argument('--resize-resolution', type=int, nargs=2, default=None, help='Resize images to W H (e.g., 1280 720)')
    parser.add_argument('--no-augmentation', action='store_false', dest='use_augmentation', help='Disable test-time augmentation')
    parser.set_defaults(use_augmentation=True)
    return parser.parse_args()

def main():
    opts = parse_args()
    args = get_config(opts.config)
    args.sequence_name = opts.sequence_name
    args.max_samples = opts.max_samples
    args.batch_size = opts.batch_size
    args.use_augmentation = opts.use_augmentation

    print("TCPFormer Evaluation with MediaPipe 2D Poses")
    print("=" * 60)
    print(f"Config: {opts.config}")
    print(f"Checkpoint: {opts.checkpoint}/{opts.checkpoint_file}")
    print(f"Frames per sample: {args.n_frames}")
    print(f"Stride: 9")
    if args.sequence_name:
        print(f"Filtering for sequence: {args.sequence_name}")
    if opts.resize_resolution:
        print(f"Resizing images to: {opts.resize_resolution}")
    print(f"Test-time augmentation: {'Enabled' if args.use_augmentation else 'Disabled'}")

    print("\nInitializing MediaPipe...")
    estimator = MediaPipe2DPoseEstimator(resize_resolution=opts.resize_resolution)

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
                             num_workers=4,
                             pin_memory=True)

    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

    try:
        print("\nStarting evaluation with MediaPipe 2D poses...")
        with torch.no_grad():
            results = evaluate_with_mediapipe_2d(model, test_loader, estimator, args)

        if results:
            print("\n✓ Evaluation completed successfully!")
            results_path = os.path.join(opts.checkpoint, 'mediapipe_evaluation_results.json')
            with open(results_path, 'w') as f:
                json_results = {
                    key: float(value) if isinstance(value, (np.floating, np.integer)) else value
                    for key, value in results.items()
                    if key not in ['data_inference', 'pose_2d_comparison']
                }
                json.dump(json_results, f, indent=2)
            print(f"✓ Results saved to: {results_path}")

            comparison_path = os.path.join(opts.checkpoint, 'mediapipe_vs_gt_2d.json')
            with open(comparison_path, 'w') as f:
                json.dump(results['pose_2d_comparison'], f, indent=2)
            print(f"✓ 2D pose comparison saved to: {comparison_path}")

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