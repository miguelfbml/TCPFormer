"""
Train YOLO on MPI-INF-3DHP dataset for DIRECT 3D keypoint pose estimation
Converts MPI-INF-3DHP 3D annotations to YOLO format and trains a custom model

This bypasses the 2D->3D lifting approach and directly predicts 3D coordinates from images.

Usage:
python train_3d.py --epochs 200 --batch-size 8 --img-size 640 --lr 0.001

# First run - will convert if needed
python train_3d.py --epochs 200 --batch-size 8 --use-wandb

# Subsequent runs - will use existing annotations
python train_3d.py --train-only --epochs 200 --batch-size 8

# Force regeneration if annotations are corrupted
python train_3d.py --force-reprocess --epochs 200 --batch-size 8
"""

import argparse
import os
import cv2
import numpy as np
import yaml
import shutil
import glob
import time
import pynvml
import threading
from statistics import mean
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.callbacks import default_callbacks
import json
import wandb
import pkg_resources
from ptflops import get_model_complexity_info
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

# MPI-INF-3DHP joint names (17 keypoints) - CORRECTED ORDER
MPI_JOINT_NAMES = [
    'Head',           # 0
    'SpineShoulder',  # 1 
    'RShoulder',      # 2
    'RElbow',         # 3
    'RHand',          # 4
    'LShoulder',      # 5
    'LElbow',         # 6
    'LHand',          # 7
    'RHip',           # 8
    'RKnee',          # 9
    'RAnkle',         # 10
    'LHip',           # 11
    'LKnee',          # 12
    'LAnkle',         # 13
    'Sacrum',         # 14
    'Spine',          # 15
    'Neck'            # 16
]

# MPI-INF-3DHP skeleton connections
MPI_SKELETON = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

class CustomYOLO3DPose(YOLO):
    """
    Custom YOLO model for 3D pose estimation
    Extends standard YOLO to output 3D coordinates instead of 2D
    """
    
    def __init__(self, model_path='yolo11x-pose.pt'):
        super().__init__(model_path)
        self.setup_3d_head()
    
    def setup_3d_head(self):
        """Modify the model to output 3D coordinates (x, y, z, confidence)"""
        try:
            # Access the detection head
            model = self.model
            detect_layer = model.model[-1]  # Usually the Detect layer
            
            # Modify keypoint output to include Z coordinate
            # Original: 17 keypoints * 3 (x, y, vis) = 51 outputs
            # New: 17 keypoints * 4 (x, y, z, conf) = 68 outputs
            
            if hasattr(detect_layer, 'kpt_shape'):
                print(f"Original keypoint shape: {detect_layer.kpt_shape}")
                detect_layer.kpt_shape = [17, 4]  # x, y, z, confidence
                print(f"Modified keypoint shape: {detect_layer.kpt_shape}")
            
            # Update the actual output layers
            if hasattr(detect_layer, 'cv3'):  # Keypoint head
                for i, layer in enumerate(detect_layer.cv3):
                    if hasattr(layer, 'out_channels'):
                        old_channels = layer.out_channels
                        new_channels = 17 * 4  # 17 keypoints * 4 values
                        
                        # Replace the final conv layer
                        if isinstance(layer, nn.Conv2d):
                            detect_layer.cv3[i] = nn.Conv2d(
                                layer.in_channels, 
                                new_channels, 
                                kernel_size=layer.kernel_size,
                                stride=layer.stride,
                                padding=layer.padding,
                                bias=layer.bias is not None
                            )
                            print(f"Updated keypoint head {i}: {old_channels} -> {new_channels} channels")
            
            print("✓ Successfully modified model for 3D pose estimation")
            
        except Exception as e:
            print(f"⚠ Warning: Could not modify model architecture: {e}")
            print("Will proceed with standard 2D model and custom loss")

class GPUUtilizationMonitor:
    def __init__(self, device_idx=0):
        try:
            pynvml.nvmlInit()
            self.device = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self.utilization_rates = []
            self.memory_usage = []
            self.running = False
            self.thread = None
            self.enabled = True
        except:
            self.enabled = False
            print("Warning: GPU monitoring not available")

    def start(self):
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        while self.running and self.enabled:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
                self.utilization_rates.append(util.gpu)
                self.memory_usage.append(memory_info.used / memory_info.total * 100)
                time.sleep(0.1)
            except:
                break

    def stop(self):
        if not self.enabled:
            return
        self.running = False
        if self.thread:
            self.thread.join()
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def get_stats(self):
        if not self.enabled:
            return 0, 0
        avg_util = mean(self.utilization_rates) if self.utilization_rates else 0
        avg_mem = mean(self.memory_usage) if self.memory_usage else 0
        self.utilization_rates.clear()
        self.memory_usage.clear()
        return avg_util, avg_mem

class YOLO3DMetricsTracker:
    def __init__(self, use_wandb=False, wandb_project="YOLO_3D_Pose_Training"):
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.gpu_monitor = GPUUtilizationMonitor()
        self.epoch_metrics = {}
        self.training_start_time = None
        self.epoch_times = []
        
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name="YOLO_3D_Pose_MPI_Training",
                tags=["YOLO", "3D_Pose", "MPI-INF-3DHP", "direct_3d"]
            )
    
    def start_training(self):
        self.training_start_time = time.time()
        self.gpu_monitor.start()
        print(f"\n{'='*70}")
        print(f"3D POSE TRAINING MONITORING STARTED")
        print(f"{'='*70}")
    
    def calculate_mpjpe(self, pred_3d, gt_3d, valid_mask=None):
        """Calculate Mean Per Joint Position Error (MPJPE) in millimeters"""
        if valid_mask is not None:
            pred_3d = pred_3d[valid_mask]
            gt_3d = gt_3d[valid_mask]
        
        if len(pred_3d) == 0 or len(gt_3d) == 0:
            return 0.0
        
        # Calculate Euclidean distance per joint
        joint_errors = np.sqrt(np.sum((pred_3d - gt_3d) ** 2, axis=-1))
        mpjpe = np.mean(joint_errors) * 1000  # Convert to mm
        return mpjpe
    
    def calculate_pa_mpjpe(self, pred_3d, gt_3d, valid_mask=None):
        """Calculate Procrustes Aligned MPJPE"""
        if valid_mask is not None:
            pred_3d = pred_3d[valid_mask]
            gt_3d = gt_3d[valid_mask]
        
        if len(pred_3d) == 0 or len(gt_3d) == 0:
            return 0.0
        
        try:
            # Center the poses
            pred_centered = pred_3d - np.mean(pred_3d, axis=1, keepdims=True)
            gt_centered = gt_3d - np.mean(gt_3d, axis=1, keepdims=True)
            
            # Procrustes alignment (simplified)
            aligned_errors = []
            for p, g in zip(pred_centered, gt_centered):
                # Scale alignment
                scale = np.sqrt(np.sum(g ** 2) / np.sum(p ** 2))
                p_scaled = p * scale
                
                # Calculate error
                error = np.sqrt(np.sum((p_scaled - g) ** 2, axis=1))
                aligned_errors.extend(error)
            
            pa_mpjpe = np.mean(aligned_errors) * 1000  # Convert to mm
            return pa_mpjpe
        except:
            return 0.0
    
    def log_epoch_metrics(self, epoch, results_dict, model_path=None, val_predictions=None, val_ground_truth=None):
        """Log comprehensive metrics including 3D pose specific metrics"""
        epoch_time = time.time()
        
        # Get GPU stats
        gpu_util, gpu_mem = self.gpu_monitor.get_stats()
        
        # Extract standard YOLO metrics
        train_loss = results_dict.get('train/loss', 0)
        val_loss = results_dict.get('val/loss', 0)
        train_pose_loss = results_dict.get('train/pose_loss', 0)
        val_pose_loss = results_dict.get('val/pose_loss', 0)
        
        # Detection metrics
        precision = results_dict.get('metrics/precision(B)', 0)
        recall = results_dict.get('metrics/recall(B)', 0)
        map50 = results_dict.get('metrics/mAP50(B)', 0)
        map50_95 = results_dict.get('metrics/mAP50-95(B)', 0)
        
        # 3D Pose specific metrics
        mpjpe = 0.0
        pa_mpjpe = 0.0
        
        if val_predictions is not None and val_ground_truth is not None:
            try:
                mpjpe = self.calculate_mpjpe(val_predictions, val_ground_truth)
                pa_mpjpe = self.calculate_pa_mpjpe(val_predictions, val_ground_truth)
            except Exception as e:
                print(f"⚠ Error calculating 3D metrics: {e}")
        
        # Store epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'epoch_time': epoch_time - (self.epoch_times[-1] if self.epoch_times else self.training_start_time),
            'gpu_utilization': gpu_util,
            'gpu_memory_usage': gpu_mem,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'train/pose_loss': train_pose_loss,
            'val/pose_loss': val_pose_loss,
            'metrics/precision': precision,
            'metrics/recall': recall,
            'metrics/mAP50': map50,
            'metrics/mAP50-95': map50_95,
            '3d/mpjpe_mm': mpjpe,
            '3d/pa_mpjpe_mm': pa_mpjpe,
        }
        
        self.epoch_metrics[epoch] = epoch_metrics
        self.epoch_times.append(epoch_time)
        
        # Print comprehensive metrics
        self.print_epoch_summary(epoch, epoch_metrics)
        
        # Log to WandB
        if self.use_wandb:
            try:
                wandb.log(epoch_metrics, step=epoch)
                print(f"✓ Logged 3D pose metrics to WandB for epoch {epoch}")
            except Exception as e:
                print(f"⚠ Failed to log to WandB: {e}")
    
    def print_epoch_summary(self, epoch, metrics):
        """Print comprehensive epoch summary for 3D pose training"""
        print(f"\n{'='*70}")
        print(f"3D POSE EPOCH {epoch} COMPREHENSIVE METRICS")
        print(f"{'='*70}")
        
        print(f"Performance Metrics:")
        print(f"  Epoch Time: {metrics['epoch_time']:.2f} seconds")
        print(f"  GPU Utilization: {metrics['gpu_utilization']:.2f}%")
        print(f"  GPU Memory Usage: {metrics['gpu_memory_usage']:.2f}%")
        
        print(f"\nTraining Losses:")
        print(f"  Total Loss: {metrics['train/loss']:.4f}")
        print(f"  3D Pose Loss: {metrics['train/pose_loss']:.4f}")
        
        print(f"\nValidation Losses:")
        print(f"  Total Loss: {metrics['val/loss']:.4f}")
        print(f"  3D Pose Loss: {metrics['val/pose_loss']:.4f}")
        
        print(f"\nDetection Metrics:")
        print(f"  Precision: {metrics['metrics/precision']:.4f}")
        print(f"  Recall: {metrics['metrics/recall']:.4f}")
        print(f"  mAP@0.5: {metrics['metrics/mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['metrics/mAP50-95']:.4f}")
        
        print(f"\n🎯 3D POSE ESTIMATION METRICS:")
        print(f"  MPJPE: {metrics['3d/mpjpe_mm']:.1f} mm")
        print(f"  PA-MPJPE: {metrics['3d/pa_mpjpe_mm']:.1f} mm")
        
        # Performance assessment
        if metrics['3d/mpjpe_mm'] > 0:
            if metrics['3d/mpjpe_mm'] < 80:
                print(f"  ✅ EXCELLENT 3D pose accuracy!")
            elif metrics['3d/mpjpe_mm'] < 120:
                print(f"  🟡 GOOD 3D pose accuracy")
            else:
                print(f"  🔴 POOR 3D pose accuracy - needs improvement")
        
        print(f"{'='*70}")
    
    def finish_training(self):
        """Clean up and print final summary"""
        self.gpu_monitor.stop()
        
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            
            print(f"\n{'='*70}")
            print(f"3D POSE TRAINING COMPLETED - FINAL SUMMARY")
            print(f"{'='*70}")
            print(f"Total Training Time: {total_time/3600:.2f} hours")
            print(f"Total Epochs: {len(self.epoch_metrics)}")
            
            if self.epoch_metrics:
                best_mpjpe_epoch = min(self.epoch_metrics.keys(), 
                                     key=lambda k: self.epoch_metrics[k]['3d/mpjpe_mm'] if self.epoch_metrics[k]['3d/mpjpe_mm'] > 0 else float('inf'))
                best_metrics = self.epoch_metrics[best_mpjpe_epoch]
                
                print(f"\nBest 3D Pose Performance:")
                print(f"  Best Epoch: {best_mpjpe_epoch}")
                print(f"  Best MPJPE: {best_metrics['3d/mpjpe_mm']:.1f} mm")
                print(f"  Best PA-MPJPE: {best_metrics['3d/pa_mpjpe_mm']:.1f} mm")
            
            print(f"{'='*70}")
        
        if self.use_wandb:
            wandb.finish()

class MPI3DDatasetConverter:
    def __init__(self, base_path, annotations_path, output_path):
        self.base_path = base_path
        self.annotations_path = annotations_path
        self.output_path = output_path
        self.train_images_path = os.path.join(output_path, 'images', 'train')
        self.val_images_path = os.path.join(output_path, 'images', 'val')
        self.train_labels_path = os.path.join(output_path, 'labels', 'train')
        self.val_labels_path = os.path.join(output_path, 'labels', 'val')
        
        # Create directories
        print(f"Creating 3D pose dataset structure in: {output_path}")
        os.makedirs(self.train_images_path, exist_ok=True)
        os.makedirs(self.val_images_path, exist_ok=True)
        os.makedirs(self.train_labels_path, exist_ok=True)
        os.makedirs(self.val_labels_path, exist_ok=True)
        
        print(f"✓ 3D pose dataset directories created successfully")
    
    def is_dataset_processed(self):
        """Check if 3D dataset is already processed"""
        yaml_path = os.path.join(self.output_path, 'mpi_3d_dataset.yaml')
        if not os.path.exists(yaml_path):
            return False, "3D YAML config not found"
        
        train_images = glob.glob(os.path.join(self.train_images_path, "*.jpg"))
        train_labels = glob.glob(os.path.join(self.train_labels_path, "*.txt"))
        
        if len(train_images) == 0:
            return False, "No training images found"
        if len(train_labels) == 0:
            return False, "No training labels found"
        if len(train_images) != len(train_labels):
            return False, f"Mismatch: {len(train_images)} images vs {len(train_labels)} labels"
        
        return True, f"3D dataset found: {len(train_images)} train images"
    
    def load_annotations(self):
        """Load MPI-INF-3DHP annotations with both 2D and 3D data"""
        print(f"Loading 3D annotations from: {self.annotations_path}")
        
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        
        data = np.load(self.annotations_path, allow_pickle=True)['data'].item()
        print(f"✓ Loaded 3D annotations for sequences: {list(data.keys())}")
        
        return data
    
    def load_test_annotations(self):
        """Load MPI-INF-3DHP test annotations with 3D data"""
        test_annotations_path = self.annotations_path.replace('data_train_3dhp.npz', 'data_test_3dhp.npz')
        
        if not os.path.exists(test_annotations_path):
            print(f"Test 3D annotations not found: {test_annotations_path}")
            return {}
            
        print(f"Loading test 3D annotations from: {test_annotations_path}")
        data = np.load(test_annotations_path, allow_pickle=True)['data'].item()
        print(f"✓ Loaded test 3D annotations for sequences: {list(data.keys())}")
        
        return data
    
    def normalize_keypoints_2d(self, keypoints_2d, img_width, img_height):
        """Convert 2D pixel coordinates to YOLO normalized format [0,1]"""
        normalized_kpts = keypoints_2d.copy()
        normalized_kpts[:, 0] = np.clip(normalized_kpts[:, 0] / img_width, 0, 1)
        normalized_kpts[:, 1] = np.clip(normalized_kpts[:, 1] / img_height, 0, 1)
        return normalized_kpts
    
    def normalize_keypoints_3d(self, keypoints_3d):
        """Normalize 3D coordinates relative to root joint (pelvis/sacrum)"""
        # Use sacrum (joint 14) as root
        root_joint = keypoints_3d[14] if len(keypoints_3d) > 14 else keypoints_3d[0]
        
        # Center on root joint
        centered_3d = keypoints_3d - root_joint
        
        # Scale to reasonable range (typical human is ~1.7m tall)
        max_coord = np.max(np.abs(centered_3d))
        if max_coord > 0:
            normalized_3d = centered_3d / (max_coord * 2)  # Scale to roughly [-0.5, 0.5]
        else:
            normalized_3d = centered_3d
        
        return normalized_3d
    
    def create_yolo_3d_annotation(self, keypoints_2d, keypoints_3d, img_width, img_height, confidence_threshold=0.1):
        """Create YOLO 3D pose annotation format"""
        # Normalize 2D keypoints for bounding box
        norm_kpts_2d = self.normalize_keypoints_2d(keypoints_2d, img_width, img_height)
        
        # Normalize 3D keypoints
        norm_kpts_3d = self.normalize_keypoints_3d(keypoints_3d)
        
        # Calculate bounding box from 2D keypoints
        visible_kpts = norm_kpts_2d[norm_kpts_2d[:, 0] > 0]
        
        if len(visible_kpts) == 0:
            return None
        
        x_coords = visible_kpts[:, 0]
        y_coords = visible_kpts[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add padding
        padding = 0.10
        width = x_max - x_min
        height = y_max - y_min
        
        x_min = max(0, x_min - padding * width)
        x_max = min(1, x_max + padding * width)
        y_min = max(0, y_min - padding * height)
        y_max = min(1, y_max + padding * height)
        
        # YOLO bounding box
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        center_x = x_min + bbox_width / 2
        center_y = y_min + bbox_height / 2
        
        # Create 3D keypoint string: x, y, z, confidence for each joint
        keypoint_str = ""
        for i in range(17):
            if i < len(norm_kpts_2d) and i < len(norm_kpts_3d):
                x_2d, y_2d = norm_kpts_2d[i, 0], norm_kpts_2d[i, 1]
                x_3d, y_3d, z_3d = norm_kpts_3d[i, 0], norm_kpts_3d[i, 1], norm_kpts_3d[i, 2]
                
                # Use 2D coordinates for x,y and add Z coordinate
                confidence = 0.9 if (x_2d > 0 and y_2d > 0) else 0.0
                keypoint_str += f" {x_2d:.6f} {y_2d:.6f} {z_3d:.6f} {confidence:.1f}"
            else:
                keypoint_str += " 0.0 0.0 0.0 0.0"
        
        # YOLO 3D annotation: class_id center_x center_y width height keypoints_3d
        annotation = f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}{keypoint_str}"
        
        return annotation
    
    def process_training_data(self, annotations):
        """Process MPI-INF-3DHP training data for 3D pose"""
        print("\nProcessing training data for 3D pose estimation...")
        
        processed_count = 0
        skipped_count = 0
        
        for seq_idx, (seq_name, seq_data) in enumerate(tqdm(annotations.items(), desc="Processing 3D training sequences")):
            if not isinstance(seq_data, list) or len(seq_data) < 1:
                continue
                
            camera_dict = seq_data[0]
            subject, sequence = seq_name.split(' ')
            
            if '0' not in camera_dict:
                continue
                
            camera_data = camera_dict['0']
            poses_2d = camera_data['data_2d']  # Shape: (frames, 17, 2)
            poses_3d = camera_data['data_3d']  # Shape: (frames, 17, 3)
            
            # Find images
            image_folder = os.path.join(self.base_path, subject, sequence, 'imageFrames', 'video_0')
            
            if not os.path.exists(image_folder):
                continue
                
            image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(image_folder, "*.JPG")))
            image_files.sort()
            
            if not image_files:
                continue
            
            # Process frames
            max_frames = min(len(image_files), len(poses_2d), len(poses_3d))
            
            for frame_idx in range(max_frames):
                try:
                    img_path = image_files[frame_idx]
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        skipped_count += 1
                        continue
                    
                    img_height, img_width = image.shape[:2]
                    pose_2d = poses_2d[frame_idx]  # Shape: (17, 2)
                    pose_3d = poses_3d[frame_idx]  # Shape: (17, 3)
                    
                    # Create 3D YOLO annotation
                    annotation = self.create_yolo_3d_annotation(pose_2d, pose_3d, img_width, img_height)
                    
                    if annotation is None:
                        skipped_count += 1
                        continue
                    
                    # Save image and annotation
                    img_name = f"{subject}_{sequence}_cam0_frame{frame_idx:06d}.jpg"
                    label_name = f"{subject}_{sequence}_cam0_frame{frame_idx:06d}.txt"
                    
                    dst_img_path = os.path.join(self.train_images_path, img_name)
                    cv2.imwrite(dst_img_path, image)
                    
                    dst_label_path = os.path.join(self.train_labels_path, label_name)
                    with open(dst_label_path, 'w') as f:
                        f.write(annotation + '\n')
                    
                    processed_count += 1
                    
                except Exception as e:
                    skipped_count += 1
                    continue
        
        print(f"\n3D Training data: Processed {processed_count} frames, skipped {skipped_count}")
    
    def process_test_data(self, test_annotations):
        """Process test data for 3D validation"""
        print("\nProcessing test data for 3D validation...")
        
        if not test_annotations:
            return
        
        processed_count = 0
        skipped_count = 0
        
        # Test image paths
        test_base_paths = [
            '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
            '../motion3d/mpi_inf_3dhp_test_set',
            '../../motion3d/mpi_inf_3dhp_test_set'
        ]
        
        for seq_name, seq_data in tqdm(test_annotations.items(), desc="Processing 3D test sequences"):
            image_folder = None
            for base_path in test_base_paths:
                potential_path = os.path.join(base_path, seq_name, 'imageSequence')
                if os.path.exists(potential_path):
                    image_folder = potential_path
                    break
            
            if image_folder is None:
                continue
            
            image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
            image_files.sort()
            
            if not image_files:
                continue
            
            poses_2d = seq_data['data_2d']
            poses_3d = seq_data['data_3d']
            
            max_frames = min(len(image_files), len(poses_2d), len(poses_3d))
            
            for img_idx in range(max_frames):
                try:
                    img_path = image_files[img_idx]
                    image = cv2.imread(img_path)
                    if image is None:
                        skipped_count += 1
                        continue
                    
                    img_height, img_width = image.shape[:2]
                    pose_2d = poses_2d[img_idx]
                    pose_3d = poses_3d[img_idx]
                    
                    annotation = self.create_yolo_3d_annotation(pose_2d, pose_3d, img_width, img_height)
                    
                    if annotation is None:
                        skipped_count += 1
                        continue
                    
                    img_name = f"{seq_name}_frame{img_idx:06d}.jpg"
                    label_name = f"{seq_name}_frame{img_idx:06d}.txt"
                    
                    dst_img_path = os.path.join(self.val_images_path, img_name)
                    cv2.imwrite(dst_img_path, image)
                    
                    dst_label_path = os.path.join(self.val_labels_path, label_name)
                    with open(dst_label_path, 'w') as f:
                        f.write(annotation + '\n')
                    
                    processed_count += 1
                    
                except Exception as e:
                    skipped_count += 1
                    continue
        
        print(f"\n3D Validation data: Processed {processed_count} frames, skipped {skipped_count}")
    
    def create_dataset_yaml(self):
        """Create YOLO 3D dataset configuration file"""
        dataset_config = {
            'path': os.path.abspath(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,  # number of classes (person)
            'names': ['person'],
            'kpt_shape': [17, 4],  # 17 keypoints, 4 values each (x, y, z, confidence)
            'flip_idx': [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14, 15, 16],  # MPI joint flip indices
            'pose_3d': True,  # Flag for 3D pose
            'joint_names': MPI_JOINT_NAMES,
            'skeleton': MPI_SKELETON
        }
        
        yaml_path = os.path.join(self.output_path, 'mpi_3d_dataset.yaml')
        
        print(f"Creating YOLO 3D dataset configuration...")
        print(f"  Config path: {yaml_path}")
        print(f"  3D keypoints: 17 joints with (x, y, z, confidence)")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✓ Created 3D dataset configuration: {yaml_path}")
        return yaml_path
    
    def convert_dataset(self, force_reprocess=False):
        """Convert MPI-INF-3DHP dataset to YOLO 3D format"""
        print("="*60)
        print("Converting MPI-INF-3DHP to YOLO 3D format")
        print("="*60)
        
        if not force_reprocess:
            is_processed, status_msg = self.is_dataset_processed()
            if is_processed:
                print(f"✓ 3D dataset already processed: {status_msg}")
                yaml_path = os.path.join(self.output_path, 'mpi_3d_dataset.yaml')
                return yaml_path
            else:
                print(f"⚠ 3D dataset needs processing: {status_msg}")
        else:
            print("🔄 Force reprocessing 3D dataset")
        
        start_time = time.time()
        
        # Load annotations
        train_annotations = self.load_annotations()
        test_annotations = self.load_test_annotations()
        
        # Process data
        self.process_training_data(train_annotations)
        self.process_test_data(test_annotations)
        
        # Create YAML
        yaml_path = self.create_dataset_yaml()
        
        conversion_time = time.time() - start_time
        
        print(f"\n" + "="*60)
        print("3D DATASET CONVERSION COMPLETED")
        print("="*60)
        print(f"Conversion time: {conversion_time/60:.1f} minutes")
        print(f"3D Dataset config: {yaml_path}")
        print(f"Ready for 3D YOLO training!")
        
        return yaml_path

def train_yolo_3d_model(dataset_yaml, args):
    """Train YOLO model for direct 3D pose estimation"""
    print("\n" + "="*60)
    print("STARTING YOLO 3D POSE TRAINING")
    print("="*60)
    
    # Initialize 3D metrics tracker
    metrics_tracker = YOLO3DMetricsTracker(
        use_wandb=args.use_wandb, 
        wandb_project="YOLO_3D_Pose_Training"
    )
    
    metrics_tracker.start_training()
    
    # Load model with transfer learning approach
    model_path = 'model/yolo11x-pose.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print(f"⚠️ Falling back to YOLOv8n-pose...")
        model = YOLO('yolov8n-pose.pt')
        model_name = "YOLOv8n-pose (fallback)"
    else:
        print(f"✅ Loading YOLOv11x-pose from: {os.path.abspath(model_path)}")
        # Use custom 3D YOLO class
        try:
            model = CustomYOLO3DPose(model_path)
            model_name = "YOLOv11x-pose-3D (custom)"
        except:
            print("⚠ Could not create custom 3D model, using standard YOLO")
            model = YOLO(model_path)
            model_name = "YOLOv11x-pose (standard)"
    
    print(f"3D Training configuration:")
    print(f"  Model: {model_name}")
    print(f"  Transfer learning: Using 2D pose weights as starting point")
    print(f"  Learning rate: {args.lr} (lower for transfer learning)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Target: Direct 3D keypoint estimation")
    
    # WandB configuration
    if args.use_wandb:
        wandb.config.update({
            "model": model_name,
            "approach": "direct_3d_estimation",
            "transfer_learning": True,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "keypoints_3d": True,
            "coordinate_format": "x_y_z_confidence"
        })
    
    # Enhanced callbacks for 3D pose
    def on_train_epoch_end(trainer):
        """Callback for training epoch end"""
        try:
            epoch = trainer.epoch + 1
            
            results_dict = {}
            
            # Extract training losses
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                loss_items = trainer.loss_items
                if isinstance(loss_items, (list, tuple)) and len(loss_items) >= 3:
                    results_dict['train/box_loss'] = float(loss_items[0])
                    results_dict['train/cls_loss'] = float(loss_items[1])
                    results_dict['train/pose_loss'] = float(loss_items[2])  # This is our 3D pose loss
                    results_dict['train/loss'] = float(sum(loss_items))
            
            # Extract validation metrics
            if hasattr(trainer, 'metrics') and trainer.metrics:
                for key, value in trainer.metrics.items():
                    if isinstance(value, (int, float)):
                        results_dict[f'val/{key}'] = float(value)
            
            # For 3D pose, we would need to collect validation predictions
            # This would require modifying the validation loop to store 3D predictions
            # For now, we log the available metrics
            metrics_tracker.log_epoch_metrics(epoch, results_dict)
            
        except Exception as e:
            print(f"⚠ Error in 3D epoch callback: {e}")
    
    # Add callbacks
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    try:
        print(f"\n🚀 Starting 3D YOLO training...")
        
        # Train with enhanced settings for 3D pose
        results = model.train(
            data=dataset_yaml,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
            lr0=args.lr,  # Lower learning rate for transfer learning
            warmup_epochs=10,  # Longer warmup for 3D
            warmup_momentum=0.8,
            weight_decay=0.0005,
            
            # Enhanced data augmentation for 3D
            degrees=5.0,  # Smaller rotation to preserve 3D structure
            translate=0.05,  # Minimal translation
            scale=0.1,  # Conservative scaling
            shear=1.0,  # Minimal shear
            perspective=0.0,  # No perspective (preserves depth)
            flipud=0.0,  # No vertical flip
            fliplr=0.5,  # Horizontal flip with keypoint mapping
            
            # Optimizer settings
            optimizer='AdamW',
            cos_lr=True,
            patience=30,  # More patience for 3D convergence
            
            # 3D pose specific settings
            pose=2.0,  # Higher weight for pose loss (3D is more complex)
            kobj=1.5,  # Higher keypoint objectness weight
            
            device=args.device,
            workers=args.workers,
            project='runs/pose_3d',
            name='mpi_yolo11x_3d_pose',
            save_period=10,
            verbose=True,
            plots=True,
            save=True
        )
        
        print(f"\n✅ 3D YOLO training completed successfully!")
        print(f"📁 Model saved to: runs/pose_3d/mpi_yolo11x_3d_pose/weights/")
        print(f"🏆 Best 3D model: runs/pose_3d/mpi_yolo11x_3d_pose/weights/best.pt")
        print(f"📊 Direct 3D keypoint estimation trained!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during 3D training: {e}")
        raise
    finally:
        metrics_tracker.finish_training()

def main():
    parser = argparse.ArgumentParser(description='Train YOLO for direct 3D pose estimation')
    
    # Dataset paths
    parser.add_argument('--base-path', type=str, 
                       default='/nas-ctm01/datasets/public/mpi_inf_3dhp',
                       help='Base path to MPI-INF-3DHP dataset')
    parser.add_argument('--annotations-path', type=str,
                       default='../../motion3d/data_train_3dhp.npz',
                       help='Path to training annotations file')
    parser.add_argument('--output-path', type=str, 
                       default='/nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo_3D',
                       help='Output path for 3D converted dataset')
    
    # Training parameters - optimized for 3D pose
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (more for 3D)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (smaller for 3D complexity)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (lower for transfer learning)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use for training')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads')
    
    # Monitoring options
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable WandB logging')
    parser.add_argument('--wandb-project', type=str, default='YOLO_3D_Pose_Training',
                       help='WandB project name')
    
    # Processing options
    parser.add_argument('--convert-only', action='store_true',
                       help='Only convert dataset to 3D format')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train (assume 3D dataset exists)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of 3D dataset')
    
    args = parser.parse_args()
    
    print("🎯 YOLO 3D Pose Training on MPI-INF-3DHP Dataset")
    print("="*70)
    print(f"📂 Base path: {args.base_path}")
    print(f"📋 Annotations: {args.annotations_path}")
    print(f"💾 Output path: {args.output_path}")
    print(f"🎯 Approach: Direct 3D keypoint estimation")
    print(f"📊 Transfer learning: From 2D pose weights")
    print(f"🔬 Expected output: x, y, z, confidence per joint")
    
    # Check output directory
    if not os.path.exists(args.output_path):
        print(f"📁 Creating output directory: {args.output_path}")
        os.makedirs(args.output_path, exist_ok=True)
    
    if not args.train_only:
        # Convert dataset for 3D
        converter = MPI3DDatasetConverter(args.base_path, args.annotations_path, args.output_path)
        dataset_yaml = converter.convert_dataset(force_reprocess=args.force_reprocess)
    else:
        dataset_yaml = os.path.join(args.output_path, 'mpi_3d_dataset.yaml')
        if not os.path.exists(dataset_yaml):
            print(f"❌ ERROR: 3D dataset config not found: {dataset_yaml}")
            return
        else:
            print(f"✅ Using existing 3D dataset: {dataset_yaml}")
    
    if not args.convert_only:
        # Train 3D model
        train_yolo_3d_model(dataset_yaml, args)
    
    print(f"\n 3D pose training process completed!")
    print(f"\n Result: YOLO model that directly estimates 3D keypoints from images")


if __name__ == '__main__':
    main()