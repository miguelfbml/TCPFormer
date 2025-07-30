"""
Calculate metrics for TCPFormer using MediaPipe 2D pose estimation on MPI-INF-3DHP test set
This evaluates the model's performance when using MediaPipe 2D keypoints instead of ground truth 2D poses
Usage: python calculate_metrics_mediapipe.py --config configs/mpi/TCPFormer_mpi_81.yaml --checkpoint checkpoint_mpi --checkpoint-file best_epoch.pth.tr

# Basic evaluation
python calculate_metrics_mediapipe.py \
    --config ../../../configs/mpi/TCPFormer_mpi_27.yaml \
    --checkpoint ../../../checkpoint_mpi \
    --checkpoint-file TCPFormer_mpi_27.pth.tr

# Evaluate specific sequence
python calculate_metrics_mediapipe.py \
    --config ../../../configs/mpi/TCPFormer_mpi_27.yaml \
    --checkpoint ../../../checkpoint_mpi \
    --checkpoint-file TCPFormer_mpi_27.pth.tr \
    --sequence-name TS1

# Test with limited samples
python calculate_metrics_mediapipe.py \
    --config ../../../configs/mpi/TCPFormer_mpi_27.yaml \
    --checkpoint ../../../checkpoint_mpi \
    --checkpoint-file TCPFormer_mpi_27.pth.tr \
    --max-samples 100
"""

import argparse
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from dataclasses import dataclass
from tqdm import tqdm
import glob
import gc

# FIXED: Navigate to project root correctly
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from data.reader.motion_dataset import MPI3DHP, Fusion
from utils.tools import get_config
from utils.learning import load_model_TCPFormer
from utils.utils_3dhp import *
from utils.data import denormalize

class MediaPipe2DPoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe to MPI-INF-3DHP joint mapping (17 joints)
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
        
        # Estimate missing joints from available ones
        self.missing_joints_estimation = {
            0: [16],     # root from head (will be estimated later)
            1: [5, 2],   # neck from shoulders
            14: [11, 8], # hip from left/right hips  
            15: [14, 1], # spine from hip and neck
        }

    def estimate_2d_pose_from_image(self, image):
        """Estimate 2D pose from image, return normalized coordinates [0,1]"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        pose_2d = np.zeros((17, 3), dtype=np.float32)  # x, y, confidence
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Map MediaPipe landmarks to MPI joints
            for mp_idx, mpi_idx in self.mp_to_mpi_mapping.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    pose_2d[mpi_idx] = [landmark.x, landmark.y, landmark.visibility]
            
            # Estimate missing joints
            for missing_joint, source_joints in self.missing_joints_estimation.items():
                valid_sources = [j for j in source_joints if pose_2d[j, 2] > 0.1]
                if valid_sources:
                    pose_2d[missing_joint, 0] = np.mean([pose_2d[j, 0] for j in valid_sources])
                    pose_2d[missing_joint, 1] = np.mean([pose_2d[j, 1] for j in valid_sources]) 
                    pose_2d[missing_joint, 2] = np.mean([pose_2d[j, 2] for j in valid_sources]) * 0.8
            
            # Special handling for root joint (joint 0) - place between hips
            if pose_2d[11, 2] > 0.1 and pose_2d[8, 2] > 0.1:  # both hips valid
                pose_2d[0, 0] = (pose_2d[11, 0] + pose_2d[8, 0]) / 2.0
                pose_2d[0, 1] = (pose_2d[11, 1] + pose_2d[8, 1]) / 2.0  
                pose_2d[0, 2] = min(pose_2d[11, 2], pose_2d[8, 2])
        
        return pose_2d

    def close(self):
        self.pose.close()

def load_video_frames_for_sequence(sequence_name, sample_indices, n_frames=27, stride=9):
    """Load video frames for specific sample indices"""
    video_path = f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence'
    
    if not os.path.exists(video_path):
        print(f"Video frames not found at: {video_path}")
        return None
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(video_path, ext)))
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"No image files found in {video_path}")
        return None
    
    # Load frames for each sample
    sample_frames = {}
    
    for sample_idx in tqdm(sample_indices, desc=f"Loading frames for {sequence_name}"):
        # Calculate frame range for this sample (same logic as dataset)
        center_frame = sample_idx * stride + (n_frames - 1) // 2
        start_frame = center_frame - (n_frames - 1) // 2
        end_frame = center_frame + (n_frames - 1) // 2 + 1
        
        frames = []
        valid_frames = 0
        
        for frame_idx in range(start_frame, end_frame):
            if 0 <= frame_idx < len(image_files):
                frame = cv2.imread(image_files[frame_idx])
                if frame is not None:
                    frames.append(frame)
                    valid_frames += 1
                else:
                    frames.append(None)
            else:
                frames.append(None)
        
        if valid_frames >= n_frames // 2:  # At least half the frames should be valid
            sample_frames[sample_idx] = frames
    
    return sample_frames

def input_augmentation_mediapipe(input_2D, model, joints_left, joints_right):
    """Apply test-time augmentation using MediaPipe 2D poses"""
    N, T, J, C = input_2D.shape
    
    # Create flipped version
    input_2D_flip = input_2D.clone()
    input_2D_flip[..., 0] = 1.0 - input_2D_flip[..., 0]  # Flip x coordinates
    input_2D_flip[:, :, joints_left + joints_right, :] = input_2D_flip[:, :, joints_right + joints_left, :]
    
    # Get predictions from both original and flipped
    output_3D_non_flip = model(input_2D)
    output_3D_flip = model(input_2D_flip)
    
    # Flip the flipped prediction back
    output_3D_flip[..., 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
    
    # Average the predictions
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    
    return input_2D, output_3D

def evaluate_with_mediapipe_2d(model, test_loader, estimator, args):
    """Evaluate model using MediaPipe 2D poses instead of ground truth"""
    model.eval()
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]
    
    error_sum_test = AccumLoss()
    pck_results = {
        'PCK@90%_torso': 0.0, 'PCK@80%_torso': 0.0, 'PCK@70%_torso': 0.0,
        'PCK@90%_150mm': 0.0, 'PCK@80%_150mm': 0.0, 'PCK@70%_150mm': 0.0
    }
    auc_sum = 0.0
    valid_samples = 0
    
    # Group samples by sequence for efficient video loading
    sequence_samples = {}
    sample_info = []
    
    print("Collecting sample information...")
    for data in tqdm(test_loader, desc="Analyzing test data"):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        
        for i in range(len(seq)):
            seq_name = seq[i]
            if seq_name not in sequence_samples:
                sequence_samples[seq_name] = []
            
            sample_info.append({
                'seq_name': seq_name,
                'batch_data': data,
                'batch_idx': i,
                'sample_count': len(sequence_samples[seq_name])
            })
            sequence_samples[seq_name].append(len(sample_info) - 1)
    
    print(f"Found {len(sample_info)} samples across {len(sequence_samples)} sequences")
    
    # Process each sequence
    for seq_name, sample_indices in sequence_samples.items():
        print(f"\nProcessing sequence: {seq_name} ({len(sample_indices)} samples)")
        
        # Load video frames for this sequence
        video_frames = load_video_frames_for_sequence(seq_name, 
                                                     range(len(sample_indices)), 
                                                     args.n_frames, 
                                                     stride=9)
        
        if video_frames is None:
            print(f"Skipping sequence {seq_name} - no video frames")
            continue
        
        # Process each sample in this sequence
        for local_idx, global_idx in enumerate(tqdm(sample_indices, desc=f"Processing {seq_name}")):
            if local_idx not in video_frames:
                continue
                
            info = sample_info[global_idx]
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = info['batch_data']
            batch_idx = info['batch_idx']
            
            # Get ground truth for this sample
            gt_3D_sample = gt_3D[batch_idx:batch_idx+1]
            scale_sample = scale[batch_idx:batch_idx+1] 
            seq_sample = [seq[batch_idx]]
            
            # Process MediaPipe 2D poses for this sample's frames
            frames = video_frames[local_idx]
            mediapipe_2d_sequence = []
            
            for frame in frames:
                if frame is not None:
                    pose_2d = estimator.estimate_2d_pose_from_image(frame)
                    mediapipe_2d_sequence.append(pose_2d[:, :2])  # Only x, y coordinates
                else:
                    # Use previous frame or zeros if no previous frame
                    if len(mediapipe_2d_sequence) > 0:
                        mediapipe_2d_sequence.append(mediapipe_2d_sequence[-1])
                    else:
                        mediapipe_2d_sequence.append(np.zeros((17, 2)))
            
            if len(mediapipe_2d_sequence) != args.n_frames:
                continue
                
            # Convert to tensor format matching original input
            mediapipe_2d_tensor = torch.from_numpy(np.stack(mediapipe_2d_sequence, axis=0)).float()  # (T, 17, 2)
            mediapipe_2d_tensor = mediapipe_2d_tensor.unsqueeze(0)  # (1, T, 17, 2)
            
            if torch.cuda.is_available():
                mediapipe_2d_tensor = mediapipe_2d_tensor.cuda()
                gt_3D_sample = gt_3D_sample.cuda()
                scale_sample = scale_sample.cuda()
            
            # Prepare ground truth (same as train_3dhp.py)
            out_target = gt_3D_sample.clone().view(1, -1, 17, 3)
            out_target[:, :, 14] = 0
            
            # Model inference with MediaPipe 2D poses
            mediapipe_2d_tensor, output_3D = input_augmentation_mediapipe(
                mediapipe_2d_tensor, model, joints_left, joints_right)
            
            # Apply scale (same as train_3dhp.py) 
            output_3D = output_3D * scale_sample.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
            
            # Extract center frame
            pad = (args.n_frames - 1) // 2
            pred_out = output_3D[:, pad].unsqueeze(1)
            
            # Post-processing (same as train_3dhp.py)
            pred_out[..., 14, :] = 0
            pred_out = denormalize(pred_out, seq_sample)
            
            # Make root-relative for MPJPE
            pred_out_relative = pred_out - pred_out[..., 14:15, :]
            out_target_relative = out_target - out_target[..., 14:15, :]
            
            # Calculate MPJPE  
            joint_error_test = mpjpe_cal(pred_out_relative, out_target_relative).item()
            error_sum_test.update(joint_error_test, 1)
            
            # Calculate additional metrics
            pred_frame = pred_out_relative[:, 0].cpu().numpy()  # (1, 17, 3)
            gt_frame = out_target_relative[:, 0].cpu().numpy()  # (1, 17, 3)
            
            # Calculate torso diameters
            torso_diameters = calculate_torso_diameter(gt_frame)
            
            # Compute PCK
            batch_pck = compute_pck(pred_frame, gt_frame, torso_diameters, fixed_threshold=150.0)
            for key in pck_results:
                pck_results[key] += batch_pck[key]
            
            # Compute AUC
            auc = compute_auc(pred_frame, gt_frame)
            auc_sum += auc
            
            valid_samples += 1
            
            # Cleanup
            del mediapipe_2d_tensor, output_3D, pred_out
            if valid_samples % 50 == 0:
                gc.collect()
    
    if valid_samples == 0:
        print("No valid samples processed!")
        return None
    
    # Average metrics
    mpjpe_avg = error_sum_test.avg
    for key in pck_results:
        pck_results[key] /= valid_samples
    auc_avg = auc_sum / valid_samples
    
    # Print results (same format as train_3dhp.py)
    print(f'\n{"="*60}')
    print(f'TCPFormer Results with MediaPipe 2D Input')
    print(f'{"="*60}')
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
        'valid_samples': valid_samples
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, default='best_epoch.pth.tr', help="checkpoint file name")
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--sequence-name', type=str, default=None, help='Specific sequence to test (TS1, TS2, etc.)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples to process (for testing)')
    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()
    
    # Load config
    args = get_config(opts.config)
    
    print("TCPFormer Evaluation with MediaPipe 2D Poses")
    print("=" * 60)
    print(f"Config: {opts.config}")
    print(f"Checkpoint: {opts.checkpoint}/{opts.checkpoint_file}")
    print(f"Frames per sample: {args.n_frames}")
    print(f"Stride: 9")
    
    # Initialize MediaPipe
    print("\nInitializing MediaPipe...")
    estimator = MediaPipe2DPoseEstimator()
    
    # Load model
    print("Loading model...")
    model = load_model_TCPFormer(args)
    
    # Load checkpoint
    checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle DataParallel wrapper
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
    
    # Create test dataset  
    print("Loading test dataset...")
    test_dataset = Fusion(args, train=False)
    
    # Filter by sequence if specified
    if opts.sequence_name:
        print(f"Filtering for sequence: {opts.sequence_name}")
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, 
                           shuffle=False, 
                           batch_size=opts.batch_size,
                           num_workers=2, 
                           pin_memory=True)
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    try:
        # Run evaluation
        print("\nStarting evaluation with MediaPipe 2D poses...")
        with torch.no_grad():
            results = evaluate_with_mediapipe_2d(model, test_loader, estimator, args)
        
        if results:
            print("\n✓ Evaluation completed successfully!")
            
            # Save results
            import json
            results_path = os.path.join(opts.checkpoint, 'mediapipe_evaluation_results.json')
            with open(results_path, 'w') as f:
                # Convert numpy values to python types for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        json_results[key] = {k: float(v) if hasattr(v, 'item') else v for k, v in value.items()}
                    else:
                        json_results[key] = float(value) if hasattr(value, 'item') else value
                json.dump(json_results, f, indent=2)
            print(f"✓ Results saved to: {results_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.close()
        print("✓ MediaPipe estimator closed")

if __name__ == '__main__':
    main()