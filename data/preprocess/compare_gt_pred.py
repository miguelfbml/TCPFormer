import argparse
import os
from dataclasses import dataclass
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from utils.data import read_pkl
from data.const import H36M_TO_MPI
from data.reader.motion_dataset import MPI3DHP

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

def read_h36m_gt(args):
    cam2real = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=np.float32)
    scale_factor = 0.298

    sample_joint_seq = read_pkl(f'../motion3d/H36M-243/test/%08d.pkl' % args.sequence_number)['data_label']
    sample_joint_seq = sample_joint_seq.transpose(1, 0, 2)  # (17, T, 3)
    sample_joint_seq = (sample_joint_seq / scale_factor) @ cam2real
    convert_h36m_to_mpi_connection()
    return sample_joint_seq

def read_mpi_gt(args):
    @dataclass
    class DatasetArgs:
        data_root: str
        n_frames: int
        stride: int
        flip: bool

    dataset_args = DatasetArgs('../motion3d/', 243, 81, False)
    dataset = MPI3DHP(dataset_args, train=False)

    print(f"Dataset length: {len(dataset)}")
    print(f"Requesting sequence: {args.sequence_number}")
    
    if args.sequence_number >= len(dataset):
        print(f"ERROR: Sequence {args.sequence_number} is out of range! Dataset has {len(dataset)} sequences.")
        return None

    # Get data from dataset
    data_tuple = dataset[args.sequence_number]
    print(f"Dataset returned {len(data_tuple)} items")
    
    # Handle different tuple lengths
    if len(data_tuple) == 5:
        # Get the raw 3D data (before normalization)
        _, _, raw_gt_3D, seq_name, _ = data_tuple  # Use the raw ground truth
        print(f"=== GROUND TRUTH SEQUENCE NAME: {seq_name} ===")
        sequence_3d = raw_gt_3D
    elif len(data_tuple) == 6:
        _, _, _, raw_gt_3D, seq_name, _ = data_tuple  # Use the raw ground truth
        print(f"=== GROUND TRUTH SEQUENCE NAME: {seq_name} ===")
        sequence_3d = raw_gt_3D
    else:
        print(f"Unexpected data tuple length: {len(data_tuple)}")
        # Try to get raw data - this should be the non-normalized version
        sequence_3d = data_tuple[-3]  # Usually the raw GT is the third from last

    # Convert to numpy if needed
    if hasattr(sequence_3d, 'cpu'):
        sequence_3d = sequence_3d.cpu().numpy()
    elif hasattr(sequence_3d, 'numpy'):
        sequence_3d = sequence_3d.numpy()

    print(f"GT raw shape after conversion: {sequence_3d.shape}")

    # Handle different possible shapes
    if sequence_3d.ndim == 3:
        if sequence_3d.shape[0] == 17:  # (17, T, 3)
            pass  # Already correct
        elif sequence_3d.shape[1] == 17:  # (T, 17, 3)
            sequence_3d = sequence_3d.transpose(1, 0, 2)  # -> (17, T, 3)
        elif sequence_3d.shape[2] == 17:  # (T, 3, 17)
            sequence_3d = sequence_3d.transpose(2, 0, 1)  # -> (17, T, 3)
    elif sequence_3d.ndim == 2:  # (17, 3) - single frame
        sequence_3d = sequence_3d[:, np.newaxis, :]  # -> (17, 1, 3)

    print(f"GT shape after reshaping: {sequence_3d.shape}")

    # Check for problematic values BEFORE transformations
    print(f"GT statistics before transformations:")
    print(f"  Min: {np.min(sequence_3d):.3f}, Max: {np.max(sequence_3d):.3f}")
    print(f"  Root joint 14 (first frame): {sequence_3d[14, 0, :]}")
    print(f"  Hip joint 8 (first frame): {sequence_3d[8, 0, :]}")

    # Apply camera transformation
    cam2real = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real

    # MAKE GROUND TRUTH ROOT-RELATIVE (like predictions)
    # This centers the pose at joint 14 for each frame
    print("Making ground truth root-relative...")
    for t in range(sequence_3d.shape[1]):  # For each frame
        root_position = sequence_3d[14, t, :].copy()  # Get root position
        sequence_3d[:, t, :] = sequence_3d[:, t, :] - root_position  # Subtract from all joints
    
    print(f"GT statistics after making root-relative:")
    print(f"  Min: {np.min(sequence_3d):.3f}, Max: {np.max(sequence_3d):.3f}")
    print(f"  Root joint 14 (first frame): {sequence_3d[14, 0, :]}")  # Should be [0, 0, 0]
    print(f"  Hip joint 8 (first frame): {sequence_3d[8, 0, :]}")

    convert_h36m_to_mpi_connection()
    return sequence_3d

def load_predictions(args):
    data = scio.loadmat(args.inference_file)
    
    # Print available sequences for debugging
    available_seqs = [key for key in data.keys() if not key.startswith('__')]
    print(f"Available sequences in inference file: {available_seqs}")
    
    # Use provided sequence name if available, otherwise try to select by index
    seq_name = args.sequence_name if args.sequence_name else None
    if seq_name is None:
        if args.sequence_number < len(available_seqs):
            seq_name = available_seqs[args.sequence_number]
            print(f"Selected sequence: {seq_name} (index {args.sequence_number})")
        else:
            raise ValueError(f"Sequence index {args.sequence_number} out of range. Available sequences: {available_seqs}")
    
    if seq_name in data:
        pred_3d = data[seq_name]  # Shape: (3, 17, 1, T)
        print(f"Prediction raw shape: {pred_3d.shape}")
        if pred_3d.ndim == 4:  # (3, 17, 1, T)
            pred_3d = pred_3d[:, :, 0, :]  # (3, 17, T)
            pred_3d = pred_3d.transpose(1, 2, 0)  # (17, T, 3)
        elif pred_3d.ndim == 3:  # (3, 17, T)
            pred_3d = pred_3d.transpose(1, 2, 0)  # (17, T, 3)
        elif pred_3d.ndim == 2:  # (17, 3)
            pred_3d = pred_3d[:, np.newaxis, :]  # (17, 1, 3)
        
        print(f"Prediction min/max values before transform: {np.min(pred_3d)}, {np.max(pred_3d)}")
        
        # Apply same camera transformation as GT
        cam2real = np.array([[1, 0, 0],
                             [0, 0, -1],
                             [0, -1, 0]], dtype=np.float32)
        pred_3d = pred_3d @ cam2real
        
        # MAKE PREDICTIONS ROOT-RELATIVE (they should already be, but ensure consistency)
        print("Making predictions root-relative...")
        for t in range(pred_3d.shape[1]):  # For each frame
            root_position = pred_3d[14, t, :].copy()  # Get root position
            pred_3d[:, t, :] = pred_3d[:, t, :] - root_position  # Subtract from all joints
        
        print(f"Prediction keypoint 14 (first frame): {pred_3d[14, 0, :]}")  # Should be [0, 0, 0]
        print(f"Prediction keypoint 8 (first frame): {pred_3d[8, 0, :]}")
        print(f"Prediction min/max values after transform: {np.min(pred_3d)}, {np.max(pred_3d)}")
        
        return pred_3d
    else:
        raise ValueError(f"Sequence {seq_name} not found in {args.inference_file}. Available sequences: {available_seqs}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, help='Specific sequence name in inference file')
    parser.add_argument('--dataset', choices=['h36m', 'mpi'], default='h36m')
    parser.add_argument('--inference-file', type=str, default='../../checkpoint_mpi/inference_data.mat',
                        help='Path to inference_data.mat file')
    args = parser.parse_args()

    print(f"Comparing sequence {args.sequence_number} of {args.dataset} dataset")

    # Load ground truth
    dataset_reader_mapper = {
        'h36m': read_h36m_gt,
        'mpi': read_mpi_gt,
    }
    gt_joint_seq = dataset_reader_mapper[args.dataset](args)
    print(f"Ground truth shape: {gt_joint_seq.shape}")

    available_sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    if args.sequence_name is None:
        # Try to match by trying different sequences
        args.sequence_name = available_sequences[args.sequence_number % len(available_sequences)]
        print(f"Trying sequence: {args.sequence_name}")

    # Load predictions
    pred_joint_seq = load_predictions(args)
    print(f"Prediction shape: {pred_joint_seq.shape}")


    # Ensure same number of frames
    num_frames = min(gt_joint_seq.shape[1], pred_joint_seq.shape[1])
    gt_joint_seq = gt_joint_seq[:, :num_frames, :]
    pred_joint_seq = pred_joint_seq[:, :num_frames, :]
    print(f"Number of frames for visualization: {num_frames}")

    # Compute global min/max for consistent scaling, excluding invalid GT values
    valid_gt = gt_joint_seq[~np.isnan(gt_joint_seq) & ~np.isinf(gt_joint_seq)]
    valid_pred = pred_joint_seq[~np.isnan(pred_joint_seq) & ~np.isinf(pred_joint_seq)]
    if valid_gt.size == 0:
        print("Warning: No valid GT data for scaling; using prediction data only")
        all_poses = valid_pred
    else:
        all_poses = np.vstack([valid_gt.reshape(-1, 3), valid_pred.reshape(-1, 3)])
    min_value = np.min(all_poses, axis=0)
    max_value = np.max(all_poses, axis=0)
    padding = (max_value - min_value) * 0.1
    min_value -= padding
    max_value += padding

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    seq_title = args.sequence_name if args.sequence_name else f"Sequence {args.sequence_number}"
    fig.suptitle(f'Ground Truth vs Prediction - {args.dataset} {seq_title}')

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Set limits
        for ax in [ax1, ax2]:
            ax.set_xlim3d([min_value[0], max_value[0]])
            ax.set_ylim3d([min_value[1], max_value[1]])
            ax.set_zlim3d([min_value[2], max_value[2]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        # Plot ground truth - FORCE PLOTTING
        ax1.set_title('Ground Truth')
        x_gt = gt_joint_seq[:, frame, 0]
        y_gt = gt_joint_seq[:, frame, 1]
        z_gt = gt_joint_seq[:, frame, 2]
        
        print(f"Frame {frame} GT debug:")
        print(f"  X range: [{np.min(x_gt):.3f}, {np.max(x_gt):.3f}]")
        print(f"  Y range: [{np.min(y_gt):.3f}, {np.max(y_gt):.3f}]")
        print(f"  Z range: [{np.min(z_gt):.3f}, {np.max(z_gt):.3f}]")
        
        # ALWAYS plot GT joints regardless of validity
        ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=50, alpha=0.8)
        
        # ALWAYS try to draw GT connections
        connections_drawn = 0
        for i, connection in enumerate(connections):
            start = gt_joint_seq[connection[0], frame, :]
            end = gt_joint_seq[connection[1], frame, :]
            try:
                ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-', linewidth=2)
                connections_drawn += 1
            except Exception as e:
                print(f"Failed to draw GT connection {i} ({connection}): {e}")
        
        print(f"Drew {connections_drawn}/{len(connections)} GT connections")
        
        # Highlight keypoint 14
        ax1.scatter(x_gt[14], y_gt[14], z_gt[14], c='green', s=100, marker='*', label='Root Joint')
        ax1.legend()

        # Plot prediction (keep original code)
        ax2.set_title('Prediction')
        x_pred = pred_joint_seq[:, frame, 0]
        y_pred = pred_joint_seq[:, frame, 1]
        z_pred = pred_joint_seq[:, frame, 2]
        
        for connection in connections:
            start = pred_joint_seq[connection[0], frame, :]
            end = pred_joint_seq[connection[1], frame, :]
            ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-', linewidth=2)
        ax2.scatter(x_pred, y_pred, z_pred, c='red', s=50)
        ax2.scatter(x_pred[14], y_pred[14], z_pred[14], c='green', s=100, marker='*', label='Root Joint')
        ax2.legend()

        return ax1, ax2

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)
    output_path = f'../{args.dataset}_comparison_{seq_title.replace(" ", "_").lower()}.gif'
    ani.save(output_path, writer='pillow', fps=20)
    print(f"Comparison GIF saved to: {output_path}")

    # Save static image of first frame
    update(0)  # Render first frame
    plt.savefig(output_path.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
    print(f"Static comparison image saved to: {output_path.replace('.gif', '.png')}")

if __name__ == '__main__':
    main()