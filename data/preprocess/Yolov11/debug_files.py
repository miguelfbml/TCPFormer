"""
Visualize MPI-INF-3DHP keypoints with labels to understand the order
This script loads annotations and displays keypoints on sample images
"""

import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# MPI-INF-3DHP joint names (17 keypoints) - Current order in .npz files
MPI_JOINT_NAMES = [
'Head',
'SpineShoulder', 
'RShoulder',
'RElbow',
'RHand',
'LShoulder',
'LElbow',
'LHand',
'RHip',
'RKnee',
'RAnkle',
'LHip',
'LKnee',
'LAnkle',
'Sacrum',
'Spine',
'Neck'

]

def print_keypoint_order():
    """Print the current keypoint order"""
    print("="*60)
    print("MPI-INF-3DHP KEYPOINT ORDER (Current .npz files)")
    print("="*60)
    for i, joint_name in enumerate(MPI_JOINT_NAMES):
        print(f"{i:2d}: {joint_name}")
    print("="*60)

def visualize_keypoints_on_image(image, keypoints_2d, joint_names, title, save_path):
    """
    Visualize keypoints on an image with labels
    
    Args:
        image: Input image
        keypoints_2d: Array of shape (17, 2) with keypoint coordinates
        joint_names: List of joint names
        title: Title for the image
        save_path: Path to save the image
    """
    # Create a copy of the image
    vis_image = image.copy()
    
    # Define colors for different body parts
    colors = {
        'head': (0, 255, 255),      # Yellow - Head, Nose
        'torso': (255, 0, 255),     # Magenta - Spine, Thorax, Root
        'left_arm': (0, 255, 0),    # Green - Left arm joints
        'right_arm': (255, 0, 0),   # Blue - Right arm joints  
        'left_leg': (0, 128, 255),  # Orange - Left leg joints
        'right_leg': (255, 128, 0), # Light blue - Right leg joints
    }
    
    def get_joint_color(joint_name):
        if 'Head' in joint_name or 'Nose' in joint_name:
            return colors['head']
        elif 'Spine' in joint_name or 'Thorax' in joint_name or 'Root' in joint_name:
            return colors['torso']
        elif joint_name.startswith('L') and ('Shoulder' in joint_name or 'Elbow' in joint_name or 'Wrist' in joint_name):
            return colors['left_arm']
        elif joint_name.startswith('R') and ('Shoulder' in joint_name or 'Elbow' in joint_name or 'Wrist' in joint_name):
            return colors['right_arm']
        elif joint_name.startswith('L') and ('Hip' in joint_name or 'Knee' in joint_name or 'Ankle' in joint_name):
            return colors['left_leg']
        elif joint_name.startswith('R') and ('Hip' in joint_name or 'Knee' in joint_name or 'Ankle' in joint_name):
            return colors['right_leg']
        else:
            return (128, 128, 128)  # Gray for unknown
    
    print(f"\n{title} - Keypoint coordinates:")
    print("-" * 50)
    
    # Draw keypoints and labels
    for i, (joint_name, (x, y)) in enumerate(zip(joint_names, keypoints_2d)):
        if x > 0 and y > 0:  # Only draw visible keypoints
            color = get_joint_color(joint_name)
            
            # Draw circle for keypoint
            cv2.circle(vis_image, (int(x), int(y)), 8, color, -1)
            cv2.circle(vis_image, (int(x), int(y)), 10, (255, 255, 255), 2)
            
            # Draw text label with index
            label = f"{i}:{joint_name}"
            font_scale = 0.6
            thickness = 2
            
            # Get text size to create background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(vis_image, 
                         (int(x) - 5, int(y) - text_height - 10), 
                         (int(x) + text_width + 5, int(y) + 5), 
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(vis_image, label, (int(x), int(y) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            print(f"{i:2d}: {joint_name:12s} -> ({x:6.1f}, {y:6.1f})")
        else:
            print(f"{i:2d}: {joint_name:12s} -> (not visible)")
    
    # Add title to image
    cv2.putText(vis_image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(vis_image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Save the image
    cv2.imwrite(save_path, vis_image)
    print(f"\n✓ Saved visualization: {save_path}")
    
    return vis_image

def load_and_visualize_training_sample():
    """Load and visualize a sample from training data"""
    print("\n" + "="*60)
    print("LOADING TRAINING SAMPLE")
    print("="*60)
    
    # Paths
    base_path = '/nas-ctm01/datasets/public/mpi_inf_3dhp'
    annotations_path = '../../motion3d/data_train_3dhp.npz'
    
    # Alternative paths if the above don't exist
    alt_annotations_paths = [
        'data_train_3dhp.npz',
        '../motion3d/data_train_3dhp.npz',
        '../../data_train_3dhp.npz'
    ]
    
    # Find annotations file
    if not os.path.exists(annotations_path):
        for alt_path in alt_annotations_paths:
            if os.path.exists(alt_path):
                annotations_path = alt_path
                break
    
    if not os.path.exists(annotations_path):
        print(f"❌ Training annotations not found. Tried:")
        print(f"  - {annotations_path}")
        for alt_path in alt_annotations_paths:
            print(f"  - {alt_path}")
        return None
    
    print(f"📋 Loading training annotations from: {annotations_path}")
    
    # Load annotations
    try:
        data = np.load(annotations_path, allow_pickle=True)['data'].item()
        print(f"✓ Loaded training data for {len(data)} sequences")
    except Exception as e:
        print(f"❌ Error loading training annotations: {e}")
        return None
    
    # Get first available sequence
    seq_name = list(data.keys())[0]
    seq_data = data[seq_name]
    print(f"📂 Using sequence: {seq_name}")
    
    # Get camera data
    camera_dict = seq_data[0]
    camera_data = camera_dict['0']  # Camera 0
    poses_2d = camera_data['data_2d']  # Shape: (frames, 17, 2)
    
    print(f"📸 Found {len(poses_2d)} frames")
    
    # Get corresponding image
    subject, sequence = seq_name.split(' ')
    image_folder = os.path.join(base_path, subject, sequence, 'imageFrames', 'video_0')
    
    if not os.path.exists(image_folder):
        print(f"❌ Image folder not found: {image_folder}")
        return None
    
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.JPG")))
    image_files.sort()
    
    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return None
    
    # Load first frame
    frame_idx = 0
    img_path = image_files[frame_idx]
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"❌ Could not load image: {img_path}")
        return None
    
    print(f"🖼️ Loaded image: {os.path.basename(img_path)}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Get keypoints for this frame
    keypoints_2d = poses_2d[frame_idx]  # Shape: (17, 2)
    
    # Visualize
    save_path = "training_sample_keypoints.jpg"
    title = f"Training Sample - {seq_name} Frame {frame_idx}"
    
    vis_image = visualize_keypoints_on_image(image, keypoints_2d, MPI_JOINT_NAMES, title, save_path)
    
    return vis_image

def load_and_visualize_test_sample():
    """Load and visualize a sample from test data with REAL keypoints"""
    print("\n" + "="*60)
    print("LOADING TEST SAMPLE (TS1)")
    print("="*60)
    
    # Test annotations path
    test_annotations_path = '../../motion3d/data_test_3dhp.npz'
    alt_test_paths = [
        'data_test_3dhp.npz',
        '../motion3d/data_test_3dhp.npz',
        '../../data_test_3dhp.npz'
    ]
    
    # Find test annotations
    if not os.path.exists(test_annotations_path):
        for alt_path in alt_test_paths:
            if os.path.exists(alt_path):
                test_annotations_path = alt_path
                break
    
    if not os.path.exists(test_annotations_path):
        print(f"❌ Test annotations not found. Tried:")
        print(f"  - {test_annotations_path}")
        for alt_path in alt_test_paths:
            print(f"  - {alt_path}")
        return None
    
    print(f"📋 Loading test annotations from: {test_annotations_path}")
    
    # Load test annotations
    try:
        test_data = np.load(test_annotations_path, allow_pickle=True)['data'].item()
        print(f"✓ Loaded test data for {len(test_data)} sequences")
        print(f"📂 Available test sequences: {list(test_data.keys())}")
    except Exception as e:
        print(f"❌ Error loading test annotations: {e}")
        return None
    
    # Look specifically for TS1
    target_seq = 'TS1'
    if target_seq not in test_data:
        print(f"❌ TS1 not found in test data. Available sequences:")
        for seq_name in test_data.keys():
            print(f"   - {seq_name}")
        # Use first available sequence as fallback
        target_seq = list(test_data.keys())[0]
        print(f"📂 Using fallback sequence: {target_seq}")
    else:
        print(f"📂 Using target sequence: {target_seq}")
    
    # Get test sequence data
    seq_data = test_data[target_seq]
    print(f"📊 Test sequence data structure: {type(seq_data)}")
    
    # Debug: Print the structure of test data
    if isinstance(seq_data, dict):
        print(f"📊 Test data keys: {list(seq_data.keys())}")
        # Try to find 2D pose data
        if 'data_2d' in seq_data:
            poses_2d = seq_data['data_2d']
            print(f"📸 Found poses_2d shape: {poses_2d.shape}")
        elif 'annot2' in seq_data:
            poses_2d = seq_data['annot2']
            print(f"📸 Found annot2 shape: {poses_2d.shape}")
        else:
            print(f"📊 Available keys in test sequence: {list(seq_data.keys())}")
            # Try the first available key that looks like pose data
            for key in seq_data.keys():
                if isinstance(seq_data[key], np.ndarray) and len(seq_data[key].shape) >= 2:
                    poses_2d = seq_data[key]
                    print(f"📸 Using key '{key}' with shape: {poses_2d.shape}")
                    break
            else:
                print(f"❌ No suitable pose data found in test sequence")
                return None
    elif isinstance(seq_data, np.ndarray):
        poses_2d = seq_data
        print(f"📸 Test data is array with shape: {poses_2d.shape}")
    else:
        print(f"❌ Unexpected test data format: {type(seq_data)}")
        return None
    
    # Test image paths
    test_base_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
        '../motion3d/mpi_inf_3dhp_test_set',
        '../../motion3d/mpi_inf_3dhp_test_set',
        'mpi_inf_3dhp_test_set'
    ]
    
    # Find test images for TS1
    image_folder = None
    for base_path in test_base_paths:
        potential_path = os.path.join(base_path, target_seq, 'imageSequence')
        if os.path.exists(potential_path):
            image_folder = potential_path
            print(f"📁 Found test images at: {potential_path}")
            break
    
    if image_folder is None:
        print(f"❌ Test images not found for {target_seq}. Tried:")
        for base_path in test_base_paths:
            potential_path = os.path.join(base_path, target_seq, 'imageSequence')
            print(f"  - {potential_path}")
        return None
    
    # Get test images
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
    image_files.sort()
    
    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return None
    
    print(f"📸 Found {len(image_files)} test images")
    
    # Load first image
    img_path = image_files[0]
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"❌ Could not load image: {img_path}")
        return None
    
    print(f"🖼️ Loaded test image: {os.path.basename(img_path)}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Get keypoints for the first frame
    frame_idx = 0
    
    # Handle different pose data shapes
    if len(poses_2d.shape) == 3:  # (frames, joints, coords)
        if frame_idx < poses_2d.shape[0]:
            keypoints_2d = poses_2d[frame_idx]  # Shape: (17, 2) or (17, 3)
        else:
            print(f"❌ Frame {frame_idx} not available. Max frames: {poses_2d.shape[0]}")
            return None
    elif len(poses_2d.shape) == 2:  # (joints, coords) - single frame
        keypoints_2d = poses_2d
    else:
        print(f"❌ Unexpected pose data shape: {poses_2d.shape}")
        return None
    
    # Ensure we have the right shape
    if keypoints_2d.shape[0] != 17:
        print(f"❌ Expected 17 keypoints, got {keypoints_2d.shape[0]}")
        return None
    
    # If keypoints have 3D coordinates, take only x,y
    if keypoints_2d.shape[1] > 2:
        keypoints_2d = keypoints_2d[:, :2]
    
    print(f"📊 Keypoints shape: {keypoints_2d.shape}")
    print(f"📊 Keypoints range - X: [{np.min(keypoints_2d[:, 0]):.1f}, {np.max(keypoints_2d[:, 0]):.1f}], Y: [{np.min(keypoints_2d[:, 1]):.1f}, {np.max(keypoints_2d[:, 1]):.1f}]")
    
    # Visualize with REAL keypoints
    save_path = f"test_sample_keypoints_{target_seq}_real.jpg"
    title = f"Test Sample - {target_seq} Frame {frame_idx} (REAL keypoints)"
    
    vis_image = visualize_keypoints_on_image(image, keypoints_2d, MPI_JOINT_NAMES, title, save_path)
    
    return vis_image

def main():
    """Main function to visualize keypoints from both datasets"""
    print("🎯 MPI-INF-3DHP Keypoint Visualization")
    
    # Print keypoint order
    print_keypoint_order()
    
    # Visualize training sample
    train_vis = load_and_visualize_training_sample()
    
    # Visualize test sample with REAL keypoints
    test_vis = load_and_visualize_test_sample()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED")
    print("="*60)
    print("📸 Check the saved images:")
    print("  - training_sample_keypoints.jpg")
    print("  - test_sample_keypoints_TS1_real.jpg")
    print("\n📋 Keypoint order is printed above in the console output")
    print("🎨 Color coding:")
    print("  - Yellow: Head joints (Head, Nose)")
    print("  - Magenta: Torso joints (Root, Spine, Thorax)")
    print("  - Green: Left arm joints")
    print("  - Blue: Right arm joints")
    print("  - Orange: Left leg joints")
    print("  - Light Blue: Right leg joints")

if __name__ == '__main__':
    main()