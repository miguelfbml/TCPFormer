
'''
python generate3D.py --sequence TS3 --num-frames 100
'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Joint connections for MPI-INF-3DHP skeleton
connections_3d = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand', 'RShoulder',
    'RElbow', 'RHand', 'LHip', 'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle',
    'Sacrum', 'Spine', 'Neck'
]

def apply_upright_correction(poses_3d):
    """Rotate poses around X-axis by 90 degrees to make figures upright."""
    rotation_x_90 = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float32)
    return poses_3d @ rotation_x_90.T

def make_root_relative_3d(poses_3d, root_joint_idx=14):
    """Subtract root joint position from all joints."""
    root_relative_poses = poses_3d.copy()
    for frame_idx in range(poses_3d.shape[0]):
        frame = poses_3d[frame_idx]
        root_pos = frame[root_joint_idx]
        if not np.all(frame == 0):
            root_relative_poses[frame_idx] = frame - root_pos[np.newaxis, :]
            root_relative_poses[frame_idx, root_joint_idx] = [0.0, 0.0, 0.0]
    return root_relative_poses

def visualize_gt_sequence(gt_poses_3d, seq_name, num_frames):
    min_frames = min(len(gt_poses_3d), num_frames)
    gt_poses = gt_poses_3d[:min_frames]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f'Ground Truth 3D Pose - {seq_name}', fontsize=16)

    # Calculate visualization bounds
    all_points = gt_poses.reshape(-1, 3)
    valid_points = all_points[~np.all(all_points == 0, axis=1)]
    if len(valid_points) > 0:
        max_range = np.max(np.abs(valid_points)) * 1.2
        coord_min, coord_max = -max_range, max_range
    else:
        coord_min, coord_max = -500, 500

    def update(frame_idx):
        ax.clear()
        ax.set_xlim3d([coord_min, coord_max])
        ax.set_ylim3d([coord_min, coord_max])
        ax.set_zlim3d([coord_min, coord_max])
        ax.set_xlabel('X (mm, right)', fontsize=12)
        ax.set_ylabel('Y (mm, forward)', fontsize=12)
        ax.set_zlabel('Z (mm, up)', fontsize=12)
        ax.set_title(f'Frame {frame_idx+1}/{min_frames}', fontsize=14, pad=20)

        # Mark root joint at origin
        ax.scatter(0, 0, 0, c='green', s=200, marker='*', alpha=1.0,
                   edgecolors='darkgreen', linewidth=3, label='Root (Hip Center)')

        # Coordinate axes
        ax.plot([0, 100], [0, 0], [0, 0], 'r-', linewidth=2, alpha=0.7)
        ax.plot([0, 0], [0, 100], [0, 0], 'g-', linewidth=2, alpha=0.7)
        ax.plot([0, 0], [0, 0], [0, 100], 'b-', linewidth=2, alpha=0.7)
        ax.text(100, 0, 0, 'X(R)', fontsize=10, color='red')
        ax.text(0, 100, 0, 'Y(F)', fontsize=10, color='green')
        ax.text(0, 0, 100, 'Z(U)', fontsize=10, color='blue')

        gt_frame = gt_poses[frame_idx]
        gt_valid = not np.all(gt_frame == 0)
        if gt_valid:
            # Draw skeleton connections
            for connection in connections_3d:
                joint1, joint2 = connection
                if joint1 < len(gt_frame) and joint2 < len(gt_frame):
                    x1, y1, z1 = gt_frame[joint1]
                    x2, y2, z2 = gt_frame[joint2]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=3, alpha=0.8)
            # Draw joint points
            for joint_idx, (x, y, z) in enumerate(gt_frame):
                if joint_idx != 14:
                    ax.scatter(x, y, z, c='blue', s=80, alpha=0.9,
                               edgecolors='darkblue', linewidth=2)
                    ax.text(x+30, y+30, z+30, str(joint_idx), fontsize=11,
                            color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.7))
        else:
            ax.text(0, 0, 100, 'No GT Data', ha='center', va='center',
                    fontsize=18, color='red', weight='bold')
        ax.view_init(elev=15, azim=45)
        plt.tight_layout()
        return [ax]

    ani = FuncAnimation(fig, update, frames=min_frames, interval=400, repeat=True, blit=False)
    print("Showing interactive 3D visualization...")
    update(0)
    # Keep a reference to the animation object
    global _ani_ref
    _ani_ref = ani
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize Ground Truth 3D poses from MPI-INF-3DHP .npz file')
    parser.add_argument('--sequence', type=str, default='TS1',
                        help='Sequence to visualize (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--num-frames', type=int, default=50,
                        help='Number of frames to visualize')
    parser.add_argument('--data-path', type=str,
                        default='../data1/motion3d/data_test_3dhp.npz',
                        help='Path to ground truth .npz file')
    args = parser.parse_args()

    # Load ground truth data
    if not os.path.exists(args.data_path):
        print(f"ERROR: Ground truth dataset not found: {args.data_path}")
        return
    gt_data = np.load(args.data_path, allow_pickle=True)['data'].item()
    print(f"✓ Ground truth loaded: {list(gt_data.keys())}")

    if args.sequence not in gt_data:
        print(f"ERROR: Sequence {args.sequence} not found in ground truth data")
        print(f"Available sequences: {list(gt_data.keys())}")
        return

    gt_poses_3d = gt_data[args.sequence]['data_3d']
    print(f"Loaded sequence {args.sequence}: {gt_poses_3d.shape[0]} frames")

    # Apply upright correction and make root-relative
    gt_poses_3d_corrected = apply_upright_correction(gt_poses_3d)
    gt_poses_3d_root_rel = make_root_relative_3d(gt_poses_3d_corrected, root_joint_idx=14)

    visualize_gt_sequence(gt_poses_3d_root_rel, args.sequence, args.num_frames)

if __name__ == '__main__':
    main()