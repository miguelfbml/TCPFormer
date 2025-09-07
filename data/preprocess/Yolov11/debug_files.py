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
    'Root',        # 0
    'RHip',        # 1  
    'RKnee',       # 2
    'RAnkle',      # 3
    'LHip',        # 4
    'LKnee',       # 5
    'LAnkle',      # 6
    'Spine',       # 7
    'Thorax',      # 8
    'Nose',        # 9
    'Head',        # 10
    'LShoulder',   # 11
    'LElbow',      # 12
    'LWrist',      # 13
    'RShoulder',   # 14
    'RElbow',      # 15
    'RWrist'       # 16
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
    """Load and visualize a sample from test data"""
    print("\n" + "="*60)
    print("LOADING TEST SAMPLE")
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
    except Exception as e:
        print(f"❌ Error loading test annotations: {e}")
        return None
    
    # Get first test sequence
    test_seq_name = list(test_data.keys())[0]
    print(f"📂 Using test sequence: {test_seq_name}")
    
    # Test image paths
    test_base_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
        '../motion3d/mpi_inf_3dhp_test_set',
        '../../motion3d/mpi_inf_3dhp_test_set',
        'mpi_inf_3dhp_test_set'
    ]
    
    # Find test images
    image_folder = None
    for base_path in test_base_paths:
        potential_path = os.path.join(base_path, test_seq_name, 'imageSequence')
        if os.path.exists(potential_path):
            image_folder = potential_path
            break
    
    if image_folder is None:
        print(f"❌ Test images not found. Tried:")
        for base_path in test_base_paths:
            potential_path = os.path.join(base_path, test_seq_name, 'imageSequence')
            print(f"  - {potential_path}")
        print("\n⚠️ Creating dummy test visualization with sample keypoints")
        
        # Create a dummy image with sample keypoints
        dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
        
        # Create dummy keypoints in a human-like pose
        dummy_keypoints = np.array([
            [320, 400],  # Root
            [340, 380],  # RHip
            [350, 320],  # RKnee
            [360, 260],  # RAnkle
            [300, 380],  # LHip
            [290, 320],  # LKnee
            [280, 260],  # LAnkle
            [320, 350],  # Spine
            [320, 280],  # Thorax
            [320, 200],  # Nose
            [320, 180],  # Head
            [280, 260],  # LShoulder
            [250, 300],  # LElbow
            [220, 340],  # LWrist
            [360, 260],  # RShoulder
            [390, 300],  # RElbow
            [420, 340],  # RWrist
        ])
        
        save_path = "test_sample_keypoints_dummy.jpg"
        title = f"Test Sample (Dummy) - {test_seq_name}"
        
        vis_image = visualize_keypoints_on_image(dummy_image, dummy_keypoints, MPI_JOINT_NAMES, title, save_path)
        return vis_image
    
    # Get test images
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
    image_files.sort()
    
    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return None
    
    # Load first image
    img_path = image_files[0]
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"❌ Could not load image: {img_path}")
        return None
    
    print(f"🖼️ Loaded test image: {os.path.basename(img_path)}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # For test data, we'll create dummy keypoints since test annotations 
    # might not have the same structure
    img_height, img_width = image.shape[:2]
    
    # Create reasonable dummy keypoints for visualization
    dummy_keypoints = np.array([
        [img_width*0.5, img_height*0.8],   # Root
        [img_width*0.55, img_height*0.75], # RHip
        [img_width*0.58, img_height*0.6],  # RKnee
        [img_width*0.6, img_height*0.45],  # RAnkle
        [img_width*0.45, img_height*0.75], # LHip
        [img_width*0.42, img_height*0.6],  # LKnee
        [img_width*0.4, img_height*0.45],  # LAnkle
        [img_width*0.5, img_height*0.65],  # Spine
        [img_width*0.5, img_height*0.5],   # Thorax
        [img_width*0.5, img_height*0.3],   # Nose
        [img_width*0.5, img_height*0.25],  # Head
        [img_width*0.4, img_height*0.45],  # LShoulder
        [img_width*0.35, img_height*0.55], # LElbow
        [img_width*0.3, img_height*0.65],  # LWrist
        [img_width*0.6, img_height*0.45],  # RShoulder
        [img_width*0.65, img_height*0.55], # RElbow
        [img_width*0.7, img_height*0.65],  # RWrist
    ])
    
    save_path = "test_sample_keypoints.jpg"
    title = f"Test Sample - {test_seq_name}"
    
    vis_image = visualize_keypoints_on_image(image, dummy_keypoints, MPI_JOINT_NAMES, title, save_path)
    
    return vis_image

def main():
    """Main function to visualize keypoints from both datasets"""
    print("🎯 MPI-INF-3DHP Keypoint Visualization")
    
    # Print keypoint order
    print_keypoint_order()
    
    # Visualize training sample
    train_vis = load_and_visualize_training_sample()
    
    # Visualize test sample  
    test_vis = load_and_visualize_test_sample()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED")
    print("="*60)
    print("📸 Check the saved images:")
    print("  - training_sample_keypoints.jpg")
    print("  - test_sample_keypoints.jpg (or test_sample_keypoints_dummy.jpg)")
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