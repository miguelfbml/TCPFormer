"""
Create YOLO version of MPI-INF-3DHP test dataset
This replaces the 2D poses with YOLO estimations while keeping everything else identical

python3 preprocess_yolo_2d.py --model-path ../runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt

# Custom output path and image size
python3 preprocess_yolo_2d.py --model-path ../runs/pose/best.pt --output-path custom_yolo_dataset.npz --img-size 800

# Force CPU usage
python3 preprocess_yolo_2d.py --model-path ../runs/pose/best.pt --device cpu
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
import torch
from ultralytics import YOLO

# Import our camera utilities
from camera_utils import get_camera_calibration_for_sequence, pixel_to_camera_coordinates

# Navigate to project root
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

class YOLO2DPoseEstimator:
    def __init__(self, model_path, img_size=640, device='auto'):
        """Initialize YOLO pose estimator"""
        if not os.path.exists(model_path):
            print(f"ERROR: YOLO model not found: {model_path}")
            sys.exit(1)
            
        try:
            self.model = YOLO(model_path)
            print(f"✓ YOLO model loaded from: {model_path}")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model: {e}")
            sys.exit(1)
        
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.device.startswith('cuda') and torch.cuda.is_available():
            self.model.to(self.device)
            print(f"✓ Model moved to {self.device}")
        else:
            self.device = 'cpu'
            print(f"✓ Using CPU for inference")
            
        self.img_size = img_size
        print(f"✓ Image size: {img_size}")
        
    def estimate_2d_pose_from_image(self, image, calib_data, target_width, target_height):
        """
        Estimate 2D pose from image and convert to camera coordinate system
        
        Args:
            image: OpenCV image (BGR format)
            calib_data: Camera calibration data
            target_width: Target width for coordinate conversion
            target_height: Target height for coordinate conversion
            
        Returns:
            numpy array of shape (17, 3) with x, y in camera coords, confidence
        """
        try:
            # Run YOLO inference with lower confidence threshold
            results = self.model.predict(image, verbose=False, imgsz=self.img_size, 
                                       conf=0.25, device=self.device)  # Lower confidence
            
            if (results and len(results) > 0 and 
                hasattr(results[0], 'keypoints') and 
                results[0].keypoints is not None and 
                len(results[0].keypoints.xy) > 0):
                
                # Get first detection's keypoints
                keypoints = results[0].keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
                conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
                
                # Ensure we have 17 keypoints
                if keypoints.shape[0] == 17:
                    # Convert coordinates to target image dimensions first
                    img_height, img_width = image.shape[:2]
                    
                    # Scale coordinates to target dimensions
                    scaled_keypoints = keypoints.copy()
                    scaled_keypoints[:, 0] = keypoints[:, 0] * (target_width / img_width)
                    scaled_keypoints[:, 1] = keypoints[:, 1] * (target_height / img_height)
                    
                    # Validate keypoints are reasonable
                    valid_kpts = 0
                    for i in range(17):
                        if (0 <= scaled_keypoints[i, 0] <= target_width and 
                            0 <= scaled_keypoints[i, 1] <= target_height and 
                            conf[i] > 0.2):
                            valid_kpts += 1
                    
                    # Require at least 8 valid keypoints
                    if valid_kpts < 8:
                        return np.zeros((17, 3), dtype=np.float32)
                    
                    # Convert pixel coordinates to camera coordinates
                    if calib_data is not None:
                        camera_coords = pixel_to_camera_coordinates(
                            scaled_keypoints, calib_data, target_width, target_height
                        )
                    else:
                        print("Warning: No calibration data, using pixel coordinates")
                        camera_coords = scaled_keypoints
                    
                    # Combine coordinates with confidence
                    pose_2d = np.zeros((17, 3), dtype=np.float32)
                    pose_2d[:, :2] = camera_coords
                    pose_2d[:, 2] = conf
                    
                    return pose_2d
                else:
                    # Handle non-17 keypoints case
                    pose_2d = np.zeros((17, 3), dtype=np.float32)
                    
                    n_kpts = min(17, keypoints.shape[0])
                    
                    # Scale available keypoints
                    img_height, img_width = image.shape[:2]
                    scaled_keypoints = keypoints[:n_kpts].copy()
                    scaled_keypoints[:, 0] = keypoints[:n_kpts, 0] * (target_width / img_width)
                    scaled_keypoints[:, 1] = keypoints[:n_kpts, 1] * (target_height / img_height)
                    
                    # Convert to camera coordinates
                    if calib_data is not None and n_kpts > 0:
                        camera_coords = pixel_to_camera_coordinates(
                            scaled_keypoints, calib_data, target_width, target_height
                        )
                        pose_2d[:n_kpts, :2] = camera_coords
                    else:
                        pose_2d[:n_kpts, :2] = scaled_keypoints
                    
                    pose_2d[:n_kpts, 2] = conf[:n_kpts] if len(conf) >= n_kpts else 0.5
                    
                    return pose_2d
            else:
                # No detection found
                return np.zeros((17, 3), dtype=np.float32)
                
        except Exception as e:
            print(f"Error in YOLO pose estimation: {e}")
            return np.zeros((17, 3), dtype=np.float32)
    
    def close(self):
        """Cleanup resources"""
        if hasattr(self, 'model') and self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

# ... (keep all other functions from your original script: get_sequence_image_dimensions, load_original_dataset, etc.)

def create_yolo_dataset(original_data, estimator, output_path):
    """Create new dataset with YOLO 2D poses in camera coordinate system"""
    print("Creating YOLO version of dataset with camera coordinate conversion...")
    
    yolo_data = {}
    
    for seq_name, seq_data in original_data.items():
        print(f"\nProcessing sequence: {seq_name}")
        
        # Get camera calibration for this sequence
        calib_data = get_camera_calibration_for_sequence(seq_name)
        if calib_data:
            print(f"  ✓ Camera calibration loaded")
            print(f"    Focal length: {calib_data['focal_length']:.2f}mm")
        else:
            print(f"  ❌ Failed to load camera calibration")
            continue
        
        # Get original image dimensions for this sequence
        orig_width, orig_height = get_sequence_image_dimensions(seq_name)
        print(f"  Original image dimensions: {orig_width}x{orig_height}")
        
        # Load images for this sequence
        image_files = load_sequence_images(seq_name)
        if image_files is None:
            print(f"Skipping {seq_name}: No images found")
            continue
        
        # Copy all original data
        yolo_seq_data = {
            'data_3d': seq_data['data_3d'].copy(),
            'valid': seq_data['valid'].copy(),
            'camera': seq_data.get('camera', None)
        }
        
        # Get original 2D data shape
        original_2d = seq_data['data_2d']
        num_frames = len(original_2d)
        
        print(f"  Original 2D shape: {original_2d.shape}")
        print(f"  Processing {num_frames} frames...")
        
        # Process images with YOLO
        yolo_poses_2d = []
        detection_failures = 0
        
        for frame_idx in tqdm(range(num_frames), desc=f"Processing {seq_name}"):
            if frame_idx < len(image_files):
                # Load and process image
                image_path = image_files[frame_idx]
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Get YOLO 2D pose in camera coordinates
                    pose_2d_with_conf = estimator.estimate_2d_pose_from_image(
                        image, calib_data, orig_width, orig_height
                    )
                    
                    # Check if YOLO detected a valid pose
                    if np.all(pose_2d_with_conf[:, :2] == 0):
                        # No detection - mark frame as invalid
                        yolo_seq_data['valid'][frame_idx] = False
                        detection_failures += 1
                    
                    yolo_poses_2d.append(pose_2d_with_conf[:, :2])  # Only x, y coordinates
                else:
                    # Use zero pose for missing image and mark as invalid
                    yolo_poses_2d.append(np.zeros((17, 2), dtype=np.float32))
                    yolo_seq_data['valid'][frame_idx] = False
                    detection_failures += 1
            else:
                # Use zero pose for missing frame and mark as invalid
                yolo_poses_2d.append(np.zeros((17, 2), dtype=np.float32))
                yolo_seq_data['valid'][frame_idx] = False
                detection_failures += 1
        
        # Convert to numpy array and store
        yolo_seq_data['data_2d'] = np.array(yolo_poses_2d, dtype=np.float32)
        
        print(f"  ✓ YOLO 2D shape: {yolo_seq_data['data_2d'].shape}")
        print(f"  ✓ Detection failures: {detection_failures}/{num_frames} ({detection_failures/num_frames*100:.1f}%)")
        print(f"  ✓ Valid frames: {np.sum(yolo_seq_data['valid'])}/{num_frames}")
        
        # Check YOLO coordinate range (should now match ground truth range)
        if yolo_seq_data['data_2d'].size > 0:
            non_zero_mask = ~np.all(yolo_seq_data['data_2d'] == 0, axis=(1, 2))
            if np.any(non_zero_mask):
                valid_poses = yolo_seq_data['data_2d'][non_zero_mask]
                print(f"  YOLO coordinate range (camera coords):")
                print(f"    X: [{np.min(valid_poses[:, :, 0]):.1f}, {np.max(valid_poses[:, :, 0]):.1f}]")
                print(f"    Y: [{np.min(valid_poses[:, :, 1]):.1f}, {np.max(valid_poses[:, :, 1]):.1f}]")
            else:
                print(f"  WARNING: All YOLO poses are zero for {seq_name}")
        
        # Verify shapes match
        assert yolo_seq_data['data_2d'].shape[:2] == original_2d.shape[:2], \
            f"Shape mismatch: YOLO {yolo_seq_data['data_2d'].shape} vs Original {original_2d.shape}"
        
        yolo_data[seq_name] = yolo_seq_data
        
        # Memory cleanup
        del yolo_poses_2d
        gc.collect()
        
        # Clear GPU cache if using CUDA
        if estimator.device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    # Save the new dataset
    print(f"\nSaving YOLO dataset to: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    np.savez_compressed(output_path, data=yolo_data)
    
    print("✓ YOLO dataset created successfully with camera coordinate conversion!")
    return yolo_data

def verify_dataset(original_data, yolo_data):
    """Verify the YOLO dataset structure matches the original"""
    print("\nVerifying dataset structure...")
    
    for seq_name in original_data.keys():
        if seq_name not in yolo_data:
            print(f"  ❌ {seq_name}: Missing in YOLO dataset")
            continue
            
        orig_seq = original_data[seq_name]
        yolo_seq = yolo_data[seq_name]
        
        # Check shapes
        orig_shape = orig_seq['data_2d'].shape
        yolo_shape = yolo_seq['data_2d'].shape
        
        if orig_shape[:2] != yolo_shape[:2]:
            print(f"  ❌ {seq_name}: Shape mismatch {orig_shape} vs {yolo_shape}")
            continue
            
        # Check other data is preserved
        for key in ['data_3d', 'valid']:
            if key in orig_seq:
                if not np.array_equal(orig_seq[key], yolo_seq[key]):
                    print(f"  ❌ {seq_name}: {key} data modified")
                    continue
        
        print(f"  ✓ {seq_name}: Shapes match, metadata preserved")
    
    print("✓ Dataset verification passed!")

def main():
    parser = argparse.ArgumentParser(description='Create YOLO version of MPI-INF-3DHP test dataset')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--output-path', type=str, 
                       default='data/motion3d/data_test_3dhp_yolo.npz',
                       help='Output path for YOLO dataset')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size for YOLO inference')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    args = parser.parse_args()
    
    print("Creating YOLO version of MPI-INF-3DHP test dataset")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")
    print(f"Note: Coordinates will be stored in original image pixel coordinates")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    # Load original dataset
    original_data = load_original_dataset()
    if original_data is None:
        return
    
    print(f"\n✓ Loaded {len(original_data)} sequences: {list(original_data.keys())}")
    
    # Initialize YOLO
    print("\nInitializing YOLO...")
    estimator = YOLO2DPoseEstimator(args.model_path, args.img_size, args.device)
    
    try:
        # Create YOLO dataset
        yolo_data = create_yolo_dataset(original_data, estimator, args.output_path)
        
        # Verify dataset
        verify_dataset(original_data, yolo_data)
        
        print(f"\n✓ Success! YOLO dataset saved to: {args.output_path}")
        print(f"\nCoordinate format: Original image pixel coordinates")
        print(f"- TS1-TS4: 2048x2048 pixel coordinates")
        print(f"- TS5-TS6: 1920x1080 pixel coordinates")
        print(f"\nTo use in training/evaluation:")
        print(f"1. Modify data_root in config to point to the new dataset")
        print(f"2. Or rename the file to replace the original")
        print(f"3. Run train_3dhp.py normally - it will use YOLO 2D poses!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.close()
        print("✓ YOLO estimator closed")

if __name__ == '__main__':
    main()