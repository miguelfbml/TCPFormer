import argparse
import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from data.reader.motion_dataset import Fusion
from data.const import H36M_TO_MPI

# MPI-INF-3DHP skeleton connections
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

def load_ground_truth(sequence_idx=0):
    """Load ground truth from test dataset"""
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

    # Configure for T=27 (adjust based on your model)
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
    
    dataset = Fusion(dataset_args, train=False)
    
    if sequence_idx < len(dataset):
        cam, gt_3D, input_2D, seq_name, scale, bb_box = dataset[sequence_idx]
        
        # Convert to numpy - check if it's already numpy or torch tensor
        if hasattr(gt_3D, 'numpy'):
            gt_3D = gt_3D.numpy()  # It's a torch tensor
        # If it's already numpy, gt_3D stays as is
        
        # Get the center frame (shape should be (1, 17, 3) -> (17, 3))
        if gt_3D.ndim == 3 and gt_3D.shape[0] == 1:
            gt_3D = gt_3D[0]  # Remove batch dimension: (17, 3)
        elif gt_3D.ndim == 2:
            pass  # Already (17, 3)
        else:
            print(f"Unexpected GT shape: {gt_3D.shape}")
        
        # Apply camera transformation for visualization
        cam2real = np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, -1, 0]], dtype=np.float32)
        gt_3D = gt_3D @ cam2real
        
        convert_h36m_to_mpi_connection()
        return gt_3D, seq_name
    else:
        raise ValueError(f"Sequence index {sequence_idx} out of range")

def load_predictions(inference_file, seq_name):
    """Load predictions from inference_data.mat"""
    data = scio.loadmat(inference_file)
    
    # Print available sequences for debugging
    available_seqs = [key for key in data.keys() if not key.startswith('__')]
    print(f"Available sequences in inference file: {available_seqs}")
    
    if seq_name in data:
        pred_3d = data[seq_name]  # Shape: (3, 17, 1, T) or similar
        print(f"Original prediction shape: {pred_3d.shape}")
        
        # Handle different possible shapes
        if pred_3d.ndim == 4:  # (3, 17, 1, T)
            pred_3d = pred_3d[:, :, 0, 0]  # Take first frame: (3, 17)
            pred_3d = pred_3d.T  # Convert to (17, 3)
        elif pred_3d.ndim == 3:  # (3, 17, T)
            pred_3d = pred_3d[:, :, 0]  # Take first frame: (3, 17)
            pred_3d = pred_3d.T  # Convert to (17, 3)
        elif pred_3d.ndim == 2:  # (17, 3)
            pass  # Already correct shape
        
        # Apply camera transformation
        cam2real = np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, -1, 0]], dtype=np.float32)
        pred_3d = pred_3d @ cam2real
        
        return pred_3d
    else:
        raise ValueError(f"Sequence {seq_name} not found in inference file. Available: {available_seqs}")

def create_comparison_gif(gt_pose, pred_pose, seq_name, output_path):
    """Create side-by-side comparison visualization"""
    
    # Get global min/max for consistent scaling
    all_poses = np.vstack([gt_pose, pred_pose])
    min_vals = np.min(all_poses, axis=0)
    max_vals = np.max(all_poses, axis=0)
    
    # Add some padding
    padding = (max_vals - min_vals) * 0.1
    min_vals -= padding
    max_vals += padding
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Ground Truth vs Prediction - {seq_name}', fontsize=16)
    
    def update_plot():
        ax1.clear()
        ax2.clear()
        
        # Set titles
        ax1.set_title('Ground Truth', fontsize=14)
        ax2.set_title('Prediction', fontsize=14)
        
        # Set consistent limits
        for ax in [ax1, ax2]:
            ax.set_xlim3d([min_vals[0], max_vals[0]])
            ax.set_ylim3d([min_vals[1], max_vals[1]])
            ax.set_zlim3d([min_vals[2], max_vals[2]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        # Plot ground truth
        x_gt = gt_pose[:, 0]
        y_gt = gt_pose[:, 1]
        z_gt = gt_pose[:, 2]
        
        for connection in connections:
            start = gt_pose[connection[0], :]
            end = gt_pose[connection[1], :]
            ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-', linewidth=2)
        ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=50, alpha=0.8)
        
        # Plot prediction
        x_pred = pred_pose[:, 0]
        y_pred = pred_pose[:, 1]
        z_pred = pred_pose[:, 2]
        
        for connection in connections:
            start = pred_pose[connection[0], :]
            end = pred_pose[connection[1], :]
            ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-', linewidth=2)
        ax2.scatter(x_pred, y_pred, z_pred, c='red', s=50, alpha=0.8)
        
        return ax1, ax2
    
    # Create static plot (since we only have one frame)
    update_plot()
    
    # Save as PNG (static image)
    plt.tight_layout()
    plt.savefig(output_path.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
    print(f"Comparison image saved to: {output_path.replace('.gif', '.png')}")
    
    # If you want to create a simple rotating GIF
    def animate(frame):
        for ax in [ax1, ax2]:
            ax.view_init(elev=20, azim=frame * 2)  # Rotate view
        return ax1, ax2
    
    # Create rotating animation
    ani = FuncAnimation(fig, animate, frames=180, interval=50, blit=False)
    ani.save(output_path, writer='pillow', fps=20)
    print(f"Rotating comparison GIF saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-idx', type=int, default=0, help='Sequence index')
    parser.add_argument('--inference-file', type=str, default='../../checkpoint_mpi/inference_data.mat', 
                       help='Path to inference_data.mat file')
    parser.add_argument('--output-dir', type=str, default='../..', help='Output directory for comparison')
    args = parser.parse_args()
    
    try:
        # Load ground truth
        print(f"Loading ground truth for sequence {args.sequence_idx}...")
        gt_pose, seq_name = load_ground_truth(args.sequence_idx)
        print(f"Ground truth shape: {gt_pose.shape}, Sequence: {seq_name}")
        
        # Load predictions
        print(f"Loading predictions from {args.inference_file}...")
        pred_pose = load_predictions(args.inference_file, seq_name)
        print(f"Prediction shape: {pred_pose.shape}")
        
        # Create comparison
        output_path = os.path.join(args.output_dir, f'comparison_{seq_name}_seq{args.sequence_idx}.gif')
        create_comparison_gif(gt_pose, pred_pose, seq_name, output_path)
        
        # Calculate and print error
        error = np.mean(np.linalg.norm(pred_pose - gt_pose, axis=1))
        print(f"MPJPE for this sequence: {error:.2f} mm")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying to inspect inference file...")
        try:
            data = scio.loadmat(args.inference_file)
            print("Available keys in inference file:")
            for key in data.keys():
                if not key.startswith('__'):
                    print(f"  {key}: {data[key].shape}")
        except Exception as e2:
            print(f"Could not load inference file: {e2}")

if __name__ == '__main__':
    main()