import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from data.reader.motion_dataset import MPI3DHP, Fusion
from data.const import H36M_TO_MPI

# MPI-INF-3DHP skeleton connections (same as compare_gt_pred.py)
connections = [
    (10, 9), (9, 8), (8, 11), (8, 14), (14, 15), (15, 16),
    (11, 12), (12, 13), (8, 7), (7, 0), (0, 4), (0, 1),
    (1, 2), (2, 3), (4, 5), (5, 6)
]

def convert_h36m_to_mpi_connection():
    global connections
    new_connections = []
    for connection in connections:
        new_connection = (H36M_TO_MPI[connection[0]], H36M_TO_MPI[connection[1]])
        new_connections.append(new_connection)
    connections = new_connections

def load_ground_truth_annotations(args):
    """Load ground truth annotations exactly like visualize.py"""
    @dataclass
    class DatasetArgs:
        data_root: str
        n_frames: int
        stride: int
        flip: bool
        test_augmentation: bool
        data_augmentation: bool
        reverse_augmentation: bool
        out_all: int
        test_batch_size: int

    # Use same parameters as visualize.py
    dataset_args = DatasetArgs(
        data_root='../motion3d/', 
        n_frames=27,
        stride=9,
        flip=False,
        test_augmentation=False,
        data_augmentation=False,
        reverse_augmentation=False,
        out_all=1,
        test_batch_size=1
    )
    
    if args.dataset == 'mpi':
        dataset = Fusion(dataset_args, train=False)  # Use test set like compare_gt_pred.py
    else:
        dataset = MPI3DHP(dataset_args, train=True)   # Use training set like visualize.py
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Looking for sample index: {args.sample_index}")
    
    if args.sample_index >= len(dataset):
        print(f"ERROR: Sample index {args.sample_index} is out of range! Dataset has {len(dataset)} samples.")
        return None, None, None, None
    
    # Get the specific sample
    try:
        if args.dataset == 'mpi':
            # For test set (Fusion dataset)
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[args.sample_index]
            
            # Extract sequence name
            if isinstance(seq, (list, tuple)):
                seq_name = seq[0]
            else:
                seq_name = str(seq)
            
            print(f"Loaded sample {args.sample_index} from sequence: {seq_name}")
            
            # Process like train_3dhp.py evaluation
            if isinstance(gt_3D, torch.Tensor):
                gt_3D = gt_3D.clone()
            gt_3D = gt_3D.view(1, -1, 17, 3)  # N=1, T, 17, 3
            gt_3D[:, :, 14] = 0  # Set root joint to 0
            
            # Extract center frame
            center_frame = gt_3D.shape[1] // 2
            pose_3d = gt_3D[0, center_frame].cpu().numpy()  # (17, 3)
            
            # Process 2D data
            if isinstance(input_2D, torch.Tensor):
                input_2D = input_2D.clone()
            if input_2D.ndim == 4:  # (1, T, 17, 3)
                pose_2d = input_2D[0, center_frame].cpu().numpy()
            else:
                pose_2d = input_2D[0].cpu().numpy()
            
        else:
            # For training set (MPI3DHP dataset)
            pose_2d_tensor, pose_3d_normalized_tensor = dataset[args.sample_index]
            
            seq_name = f"training_sample_{args.sample_index}"
            print(f"Loaded training sample {args.sample_index}")
            
            # Convert to numpy
            pose_2d = pose_2d_tensor.cpu().numpy()  # (T, 17, 3)
            pose_3d_normalized = pose_3d_normalized_tensor.cpu().numpy()  # (T, 17, 3)
            
            # Take center frame
            center_frame = pose_2d.shape[0] // 2
            pose_2d = pose_2d[center_frame]  # (17, 3)
            pose_3d = pose_3d_normalized[center_frame]  # (17, 3)
        
        # Apply coordinate transformations (same as visualize.py)
        cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
        pose_3d = pose_3d @ cam2real
        
        # Make root-relative
        pose_3d = pose_3d - pose_3d[14:15, :]
        
        convert_h36m_to_mpi_connection()
        
        return pose_2d, pose_3d, seq_name, center_frame
        
    except Exception as e:
        print(f"Error loading sample {args.sample_index}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def load_corresponding_image(seq_name, frame_index, base_image_path):
    """Try to load corresponding image from MPI-INF-3DHP test set"""
    possible_paths = [
        f'{base_image_path}/mpi_inf_3dhp_test_set/{seq_name}/imageSequence',
        f'{base_image_path}/mpi_inf_3dhp_test_set/{seq_name}/images',
        f'{base_image_path}/{seq_name}/imageSequence',
        f'../motion3d/{seq_name}/imageSequence',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found image directory: {path}")
            
            # Look for images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                import glob
                image_files.extend(glob.glob(os.path.join(path, ext)))
                image_files.extend(glob.glob(os.path.join(path, ext.upper())))
            
            if image_files:
                image_files.sort()  # Sort numerically
                print(f"Found {len(image_files)} images")
                
                # Try to get image corresponding to frame_index
                if frame_index < len(image_files):
                    image_path = image_files[frame_index]
                    image = cv2.imread(image_path)
                    if image is not None:
                        print(f"Loaded image: {image_path}")
                        return image, image_path
                
                # If specific frame not found, use first image
                image = cv2.imread(image_files[0])
                if image is not None:
                    print(f"Using first image: {image_files[0]}")
                    return image, image_files[0]
    
    print("No corresponding image found")
    return None, None

def visualize_keypoints(pose_2d, pose_3d, seq_name, frame_index, image=None, image_path=None):
    """Visualize 2D and 3D keypoints"""
    
    if image is not None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    fig.suptitle(f'Ground Truth Keypoints - {seq_name} (Frame {frame_index})', fontsize=16)
    
    # Joint names for labeling
    joint_names = ['Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
                   'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
                   'RShoulder', 'RElbow', 'RWrist']
    
    # Plot 1: 2D keypoints
    ax1.set_title('2D Ground Truth Keypoints', fontsize=14)
    
    # Check if poses are normalized or in pixel coordinates
    if np.max(pose_2d[:, :2]) <= 1.0:
        # Normalized coordinates [0,1]
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.invert_yaxis()
        coord_type = "Normalized"
    else:
        # Pixel coordinates
        ax1.set_xlim(0, np.max(pose_2d[:, 0]) * 1.1)
        ax1.set_ylim(0, np.max(pose_2d[:, 1]) * 1.1)
        ax1.invert_yaxis()
        coord_type = "Pixel"
    
    # Plot 2D keypoints with confidence filtering
    valid_2d = pose_2d[:, 2] > 0.1 if pose_2d.shape[1] > 2 else np.ones(len(pose_2d), dtype=bool)
    
    if np.any(valid_2d):
        ax1.scatter(pose_2d[valid_2d, 0], pose_2d[valid_2d, 1], 
                   c='blue', s=80, alpha=0.8, edgecolors='darkblue', linewidth=1)
        
        # Draw 2D skeleton connections
        for connection in connections:
            joint1, joint2 = connection
            if (joint1 < len(pose_2d) and joint2 < len(pose_2d) and
                valid_2d[joint1] and valid_2d[joint2]):
                ax1.plot([pose_2d[joint1, 0], pose_2d[joint2, 0]], 
                        [pose_2d[joint1, 1], pose_2d[joint2, 1]], 
                        'b-', linewidth=2, alpha=0.7)
        
        # Add joint labels
        for i, (valid, name) in enumerate(zip(valid_2d, joint_names)):
            if valid and i < len(pose_2d):
                ax1.annotate(f'{i}:{name}', (pose_2d[i, 0], pose_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, color='white', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
    
    ax1.set_xlabel(f'X ({coord_type})', fontsize=12)
    ax1.set_ylabel(f'Y ({coord_type})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 3D keypoints
    ax2 = fig.add_subplot(1, 2, 2, projection='3d') if image is None else fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title('3D Ground Truth Keypoints (Root-Relative)', fontsize=14)
    
    # Plot 3D keypoints
    x_3d, y_3d, z_3d = pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2]
    ax2.scatter(x_3d, y_3d, z_3d, c='red', s=80, alpha=0.8, edgecolors='darkred', linewidth=1)
    
    # Draw 3D skeleton connections
    for connection in connections:
        joint1, joint2 = connection
        if joint1 < len(pose_3d) and joint2 < len(pose_3d):
            ax2.plot([pose_3d[joint1, 0], pose_3d[joint2, 0]], 
                    [pose_3d[joint1, 1], pose_3d[joint2, 1]], 
                    [pose_3d[joint1, 2], pose_3d[joint2, 2]], 
                    'r-', linewidth=2, alpha=0.7)
    
    # Highlight root joint (should be at origin after root-relative transformation)
    ax2.scatter(x_3d[14], y_3d[14], z_3d[14], c='green', s=120, marker='*', 
               alpha=1.0, edgecolors='darkgreen', linewidth=2)
    
    # Set equal aspect ratio
    max_range = np.array([x_3d.max()-x_3d.min(), y_3d.max()-y_3d.min(), z_3d.max()-z_3d.min()]).max() / 2.0
    mid_x = (x_3d.max()+x_3d.min()) * 0.5
    mid_y = (y_3d.max()+y_3d.min()) * 0.5
    mid_z = (z_3d.max()+z_3d.min()) * 0.5
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax2.set_xlabel('X (mm)', fontsize=10)
    ax2.set_ylabel('Y (mm)', fontsize=10)
    ax2.set_zlabel('Z (mm)', fontsize=10)
    
    # Plot 3 & 4: Image with overlaid keypoints (if image available)
    if image is not None:
        # Plot 3: Original image
        ax3.set_title(f'Original Image\n{os.path.basename(image_path)}', fontsize=12)
        ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax3.axis('off')
        
        # Plot 4: Image with keypoints overlay
        ax4.set_title('Image with 2D Keypoints Overlay', fontsize=12)
        ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Scale keypoints to image coordinates if needed
        h, w = image.shape[:2]
        if np.max(pose_2d[:, :2]) <= 1.0:
            # Convert normalized coordinates to pixel coordinates
            pose_2d_img = pose_2d.copy()
            pose_2d_img[:, 0] *= w
            pose_2d_img[:, 1] *= h
        else:
            pose_2d_img = pose_2d
        
        # Overlay keypoints
        if np.any(valid_2d):
            ax4.scatter(pose_2d_img[valid_2d, 0], pose_2d_img[valid_2d, 1], 
                       c='yellow', s=100, alpha=0.9, edgecolors='red', linewidth=2)
            
            # Draw skeleton on image
            for connection in connections:
                joint1, joint2 = connection
                if (joint1 < len(pose_2d_img) and joint2 < len(pose_2d_img) and
                    valid_2d[joint1] and valid_2d[joint2]):
                    ax4.plot([pose_2d_img[joint1, 0], pose_2d_img[joint2, 0]], 
                            [pose_2d_img[joint1, 1], pose_2d_img[joint2, 1]], 
                            'yellow', linewidth=3, alpha=0.8)
            
            # Add joint labels on image
            for i, (valid, name) in enumerate(zip(valid_2d, joint_names)):
                if valid and i < len(pose_2d_img):
                    ax4.annotate(f'{i}', (pose_2d_img[i, 0], pose_2d_img[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, color='yellow', weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))
        
        ax4.axis('off')
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n=== Keypoint Statistics ===")
    print(f"Sequence: {seq_name}")
    print(f"Frame index: {frame_index}")
    print(f"2D pose shape: {pose_2d.shape}")
    print(f"3D pose shape: {pose_3d.shape}")
    print(f"Valid 2D joints: {np.sum(valid_2d)}/17")
    
    if pose_2d.shape[1] > 2:
        print(f"2D confidence range: {np.min(pose_2d[:, 2]):.3f} to {np.max(pose_2d[:, 2]):.3f}")
    
    print(f"2D coordinate range: X[{np.min(pose_2d[:, 0]):.3f}, {np.max(pose_2d[:, 0]):.3f}], Y[{np.min(pose_2d[:, 1]):.3f}, {np.max(pose_2d[:, 1]):.3f}]")
    print(f"3D coordinate range: X[{np.min(pose_3d[:, 0]):.1f}, {np.max(pose_3d[:, 0]):.1f}], Y[{np.min(pose_3d[:, 1]):.1f}, {np.max(pose_3d[:, 1]):.1f}], Z[{np.min(pose_3d[:, 2]):.1f}, {np.max(pose_3d[:, 2]):.1f}] mm")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize ground truth keypoints from annotations')
    parser.add_argument('--sample-index', type=int, default=0, 
                       help='Sample index to load from dataset')
    parser.add_argument('--dataset', choices=['mpi', 'h36m'], default='mpi',
                       help='Dataset to use (mpi for test set, h36m for training set)')
    parser.add_argument('--image-path', type=str, 
                       default='/nas-ctm01/datasets/public/mpi_inf_3dhp',
                       help='Base path to MPI-INF-3DHP images')
    parser.add_argument('--save-plot', action='store_true',
                       help='Save the plot as an image')
    parser.add_argument('--sequence-name', type=str, default=None,
                       help='Specific sequence name (TS1, TS2, etc.) - will filter dataset')
    
    args = parser.parse_args()
    
    print(f"Loading ground truth annotations from {args.dataset} dataset...")
    print(f"Sample index: {args.sample_index}")
    
    # Load ground truth data
    pose_2d, pose_3d, seq_name, frame_index = load_ground_truth_annotations(args)
    
    if pose_2d is None:
        print("Failed to load ground truth data. Exiting.")
        return
    
    # Try to load corresponding image
    image, image_path = load_corresponding_image(seq_name, frame_index, args.image_path)
    
    # Create visualization
    fig = visualize_keypoints(pose_2d, pose_3d, seq_name, frame_index, image, image_path)
    
    # Save plot if requested
    if args.save_plot:
        output_name = f'../keypoints_{seq_name}_{args.sample_index}.png'
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_name}")
    
    # Show plot
    plt.show()

if __name__ == '__main__':
    main()