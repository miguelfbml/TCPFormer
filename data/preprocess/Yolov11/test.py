"""
Test YOLO pose detection on MPI-INF-3DHP test images
Visualize YOLO keypoints without converting to MPI format

Usage:
python test.py --sequence TS1 --num-images 5
"""

import argparse
import os
import cv2
import numpy as np
import glob
from ultralytics import YOLO
from pathlib import Path

# YOLO pose keypoint names (COCO format)
YOLO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# YOLO skeleton connections (COCO format)
YOLO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Colors for visualization (BGR format)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 128), (255, 165, 0), (255, 192, 203), (173, 216, 230),
    (144, 238, 144), (255, 20, 147), (0, 191, 255), (220, 20, 60), (255, 215, 0),
    (50, 205, 50), (138, 43, 226)
]

def load_mpi_test_images(sequence_name, num_images=10):
    """Load images from MPI-INF-3DHP test sequences"""
    possible_paths = [
        f'/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence',
        f'../motion3d/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence',
        f'../../motion3d/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence',
    ]
    
    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            print(f"Found image directory: {path}")
            break
    
    if image_path is None:
        print(f"Could not find images for sequence {sequence_name}")
        print("Checked paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return []
    
    # Get image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(image_path, ext)))
        image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))
    
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {image_path}")
        return []
    
    print(f"Found {len(image_files)} images, loading first {min(num_images, len(image_files))}")
    
    # Load images
    images = []
    image_names = []
    
    for i, img_path in enumerate(image_files[:num_images]):
        image = cv2.imread(img_path)
        if image is not None:
            images.append(image)
            image_names.append(os.path.basename(img_path))
            print(f"  Loaded: {os.path.basename(img_path)} - {image.shape}")
        else:
            print(f"  Failed to load: {img_path}")
    
    return images, image_names

def draw_yolo_keypoints(image, keypoints, confidence_threshold=0.5):
    """Draw YOLO keypoints and skeleton on image"""
    img_copy = image.copy()
    h, w = image.shape[:2]
    
    # Convert keypoints to pixel coordinates if they're normalized
    if np.max(keypoints[:, :2]) <= 1.0:
        keypoints[:, 0] *= w
        keypoints[:, 1] *= h
    
    # Draw skeleton connections
    for connection in YOLO_SKELETON:
        pt1_idx, pt2_idx = connection
        
        if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and 
            keypoints[pt1_idx, 2] > confidence_threshold and 
            keypoints[pt2_idx, 2] > confidence_threshold):
            
            pt1 = tuple(map(int, keypoints[pt1_idx, :2]))
            pt2 = tuple(map(int, keypoints[pt2_idx, :2]))
            
            cv2.line(img_copy, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            x, y = int(x), int(y)
            color = COLORS[i % len(COLORS)]
            
            # Draw keypoint
            cv2.circle(img_copy, (x, y), 5, color, -1)
            cv2.circle(img_copy, (x, y), 7, (255, 255, 255), 2)
            
            # Draw keypoint name
            cv2.putText(img_copy, f'{YOLO_KEYPOINT_NAMES[i]}', 
                       (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
            
            # Draw confidence score
            cv2.putText(img_copy, f'{conf:.2f}', 
                       (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, color, 1)
    
    return img_copy

def test_yolo_on_mpi_images(model, images, image_names, output_dir, sequence_name):
    """Test YOLO on MPI-INF-3DHP images and save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing {len(images)} images from sequence {sequence_name}...")
    
    for i, (image, img_name) in enumerate(zip(images, image_names)):
        print(f"\nProcessing image {i+1}/{len(images)}: {img_name}")
        
        # Run YOLO inference
        results = model(image, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            # Get keypoints for first detected person
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # Shape: (17, 3)
            
            print(f"  Detected {len(keypoints)} keypoints")
            print(f"  Keypoints shape: {keypoints.shape}")
            
            # Draw keypoints on image
            annotated_image = draw_yolo_keypoints(image, keypoints)
            
            # Add title
            title = f"YOLO Pose Detection - {sequence_name} - {img_name}"
            cv2.putText(annotated_image, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add keypoint info
            valid_keypoints = np.sum(keypoints[:, 2] > 0.5)
            info_text = f"Valid keypoints: {valid_keypoints}/17"
            cv2.putText(annotated_image, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save result
            output_name = f"{sequence_name}_{img_name.split('.')[0]}_yolo_pose.jpg"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, annotated_image)
            
            print(f"  Saved: {output_path}")
            print(f"  Valid keypoints: {valid_keypoints}/17")
            
            # Print keypoint details
            for j, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    print(f"    {YOLO_KEYPOINT_NAMES[j]}: ({x:.1f}, {y:.1f}) conf={conf:.3f}")
        
        else:
            print(f"  No pose detected in {img_name}")
            
            # Save image with "No pose detected" text
            no_pose_image = image.copy()
            cv2.putText(no_pose_image, f"No Pose Detected - {img_name}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            output_name = f"{sequence_name}_{img_name.split('.')[0]}_no_pose.jpg"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, no_pose_image)
            print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Test YOLO pose detection on MPI-INF-3DHP images')
    parser.add_argument('--sequence', type=str, default='TS1',
                       help='MPI-INF-3DHP test sequence (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to process')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for visualization results')
    parser.add_argument('--model-path', type=str, default='model/yolov11x-pose.pt',
                       help='Path to YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for keypoint visualization')
    
    args = parser.parse_args()
    
    print("YOLO Pose Detection Test on MPI-INF-3DHP")
    print("=" * 50)
    print(f"Sequence: {args.sequence}")
    print(f"Number of images: {args.num_images}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Confidence threshold: {args.confidence}")
    
    # Load YOLO model
    if not os.path.exists(args.model_path):
        print(f"\nERROR: Model file not found: {args.model_path}")
        print("Please download the YOLO pose model or check the path")
        return
    
    print(f"\nLoading YOLO model: {args.model_path}")
    try:
        model = YOLO(args.model_path)
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"ERROR loading YOLO model: {e}")
        return
    
    # Load MPI-INF-3DHP test images
    print(f"\nLoading images from sequence {args.sequence}...")
    images, image_names = load_mpi_test_images(args.sequence, args.num_images)
    
    if not images:
        print("No images loaded. Exiting.")
        return
    
    # Test YOLO on images
    test_yolo_on_mpi_images(model, images, image_names, args.output_dir, args.sequence)
    
    print(f"\n✓ Processing complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total images processed: {len(images)}")
    
    # Print YOLO vs MPI-INF-3DHP keypoint comparison
    print(f"\nYOLO Keypoint Format (COCO):")
    print(f"Total keypoints: 17")
    for i, name in enumerate(YOLO_KEYPOINT_NAMES):
        print(f"  {i:2d}: {name}")
    
    print(f"\nNote: YOLO uses COCO format (17 keypoints)")
    print(f"MPI-INF-3DHP uses different joint definitions (17 keypoints)")
    print(f"This is just for visualization - no conversion applied")

if __name__ == '__main__':
    main()