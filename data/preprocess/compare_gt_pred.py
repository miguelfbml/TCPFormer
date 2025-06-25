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
    dataset = MPI3DHP(dataset_args, train=True)

    _, sequence_3d = dataset[args.sequence_number]
    sequence_3d = sequence_3d.cpu().numpy()

    cam2real = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d.transpose(1, 0, 2)  # (17, T, 3)
    # Remove zeroing of keypoint 14 to avoid GT error
    # sequence_3d[14, ...] = 0  # Commented out to preserve keypoint 14
    sequence_3d = sequence_3d @ cam2real
    convert_h36m_to_mpi_connection()
    return sequence_3d

def load_predictions(args):
    data = scio.loadmat(args.inference_file)
    seq_name = f'seq_{args.sequence_number:08d}'  # Adjust based on .mat file naming
    if seq_name in data:
        pred_3d = data[seq_name]  # Shape: (3, 17, 1, T) or similar
        if pred_3d.ndim == 4:  # (3, 17, 1, T)
            pred_3d = pred_3d[:, :, 0, :]  # (3, 17, T)
            pred_3d = pred_3d.transpose(1, 2, 0)  # (17, T, 3)
        elif pred_3d.ndim == 3:  # (3, 17, T)
            pred_3d = pred_3d.transpose(1, 2, 0)  # (17, T, 3)
        elif pred_3d.ndim == 2:  # (17, 3)
            pred_3d = pred_3d[:, np.newaxis, :]  # (17, 1, 3)

        # Apply same camera transformation as GT
        cam2real = np.array([[1, 0, 0],
                             [0, 0, -1],
                             [0, -1, 0]], dtype=np.float32)
        pred_3d = pred_3d @ cam2real
        return pred_3d
    else:
        raise ValueError(f"Sequence {seq_name} not found in {args.inference_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0)
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

    # Load predictions
    pred_joint_seq = load_predictions(args)
    print(f"Prediction shape: {pred_joint_seq.shape}")

    # Ensure same number of frames
    num_frames = min(gt_joint_seq.shape[1], pred_joint_seq.shape[1])
    gt_joint_seq = gt_joint_seq[:, :num_frames, :]
    pred_joint_seq = pred_joint_seq[:, :num_frames, :]

    # Compute global min/max for consistent scaling
    all_poses = np.vstack([gt_joint_seq.reshape(-1, 3), pred_joint_seq.reshape(-1, 3)])
    min_value = np.min(all_poses, axis=0)
    max_value = np.max(all_poses, axis=0)
    padding = (max_value - min_value) * 0.1
    min_value -= padding
    max_value += padding

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Ground Truth vs Prediction - {args.dataset} Sequence {args.sequence_number}')

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

        # Plot ground truth
        ax1.set_title('Ground Truth')
        x_gt = gt_joint_seq[:, frame, 0]
        y_gt = gt_joint_seq[:, frame, 1]
        z_gt = gt_joint_seq[:, frame, 2]
        for connection in connections:
            start = gt_joint_seq[connection[0], frame, :]
            end = gt_joint_seq[connection[1], frame, :]
            ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b')
        ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=50)

        # Plot prediction
        ax2.set_title('Prediction')
        x_pred = pred_joint_seq[:, frame, 0]
        y_pred = pred_joint_seq[:, frame, 1]
        z_pred = pred_joint_seq[:, frame, 2]
        for connection in connections:
            start = pred_joint_seq[connection[0], frame, :]
            end = pred_joint_seq[connection[1], frame, :]
            ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r')
        ax2.scatter(x_pred, y_pred, z_pred, c='red', s=50)

        return ax1, ax2

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)
    output_path = f'../{args.dataset}_comparison_seq{args.sequence_number}.gif'
    ani.save(output_path, writer='pillow', fps=20)
    print(f"Comparison GIF saved to: {output_path}")

    # Save static image of first frame
    update(0)  # Render first frame
    plt.savefig(output_path.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
    print(f"Static comparison image saved to: {output_path.replace('.gif', '.png')}")

if __name__ == '__main__':
    main()