"""
Compare Ground Truth and YOLO 2D poses for selected frames only.

This script is a focused variant of compare_gt_yolo_2d.py. It processes a
single sequence and only the frames selected through the command line, then
stores side-by-side GT vs YOLO comparisons as PNG images in an output folder.

Example:
python compare_gt_yolo_selected_frames.py --sequence TS1 --frames 0 10 20 --model-path runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt
"""

import argparse
import gc
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from compare_gt_yolo_2d import (  # noqa: E402
    CONNECTIONS_2D,
    check_gpu_availability,
    convert_coordinates_to_pixels,
    estimate_yolo_poses,
    load_test_3d_data_from_dataset,
    make_root_relative_2d_pixel,
)


TEST_IMAGE_PATHS = [
    '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
]


def load_sequence_image_paths(sequence_name):
    for base_path in TEST_IMAGE_PATHS:
        image_folder = os.path.join(base_path, sequence_name, 'imageSequence')
        if os.path.exists(image_folder):
            image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
            image_files.extend(glob.glob(os.path.join(image_folder, '*.png')))
            image_files.sort()

            if image_files:
                print(f"✓ Found images at: {os.path.abspath(image_folder)}")
                return image_files

    return None


def load_selected_frames(sequence_name, frame_indices):
    image_files = load_sequence_image_paths(sequence_name)
    if image_files is None:
        print(f"❌ Could not find image folder for sequence {sequence_name}")
        return None, None

    frames = []
    valid_indices = []

    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(image_files):
            print(f"⚠ Skipping frame {frame_idx}: out of range (0-{len(image_files) - 1})")
            continue

        frame = cv2.imread(image_files[frame_idx])
        if frame is None:
            print(f"⚠ Skipping frame {frame_idx}: failed to read image")
            continue

        frames.append(frame)
        valid_indices.append(frame_idx)

    if not frames:
        return None, None

    return frames, valid_indices


def draw_pose_panel(ax, pose, title, color, missing_text, bounds=None, show_root=True):
    ax.set_title(title, fontsize=14)

    if pose is None or np.all(pose == 0):
        ax.text(
            0.5,
            0.5,
            missing_text,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=16,
            color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )
        ax.axis('off')
        return

    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        valid_points = pose[~np.all(pose == 0, axis=1)]
        if len(valid_points) > 0:
            x_min = np.min(valid_points[:, 0])
            x_max = np.max(valid_points[:, 0])
            y_min = np.min(valid_points[:, 1])
            y_max = np.max(valid_points[:, 1])
            x_padding = max((x_max - x_min) * 0.1, 10)
            y_padding = max((y_max - y_min) * 0.1, 10)
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        else:
            ax.set_xlim(-500, 500)
            ax.set_ylim(-500, 500)

    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    for joint1, joint2 in CONNECTIONS_2D:
        if joint1 < len(pose) and joint2 < len(pose):
            x1, y1 = pose[joint1]
            x2, y2 = pose[joint2]
            if not ((x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0)):
                ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=3, alpha=0.85)

    for joint_idx, (x, y) in enumerate(pose):
        if joint_idx == 14:
            continue
        if x == 0 and y == 0:
            continue
        ax.scatter(x, y, c=color, s=70, alpha=0.95, edgecolors='black', linewidth=1)
        ax.text(
            x + 15,
            y + 15,
            str(joint_idx),
            fontsize=9,
            ha='left',
            va='bottom',
            color='black',
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
        )

    if show_root:
        ax.scatter(0, 0, c='green', s=120, marker='*', alpha=1.0, edgecolors='darkgreen', linewidth=2)

    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)


def save_frame_comparison(gt_frame, yolo_frame, sequence_name, frame_idx, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    frame_error = np.mean(np.linalg.norm(gt_frame - yolo_frame, axis=1)) if gt_frame is not None and yolo_frame is not None else 0.0
    fig.suptitle(
        f'2D Pose Comparison: GT vs YOLO - {sequence_name} | Frame {frame_idx}',
        fontsize=14,
    )

    all_points = np.vstack([
        gt_frame.reshape(-1, 2),
        yolo_frame.reshape(-1, 2),
    ])
    valid_points = all_points[~np.all(all_points == 0, axis=1)]
    if len(valid_points) > 0:
        x_range = [np.min(valid_points[:, 0]), np.max(valid_points[:, 0])]
        y_range = [np.min(valid_points[:, 1]), np.max(valid_points[:, 1])]
        x_padding = max((x_range[1] - x_range[0]) * 0.1, 10)
        y_padding = max((y_range[1] - y_range[0]) * 0.1, 10)
        bounds = (
            x_range[0] - x_padding,
            x_range[1] + x_padding,
            y_range[0] - y_padding,
            y_range[1] + y_padding,
        )
    else:
        bounds = (-500, 500, -500, 500)

    draw_pose_panel(
        ax1,
        gt_frame,
        f'Ground Truth\nFrame {frame_idx}',
        'blue',
        'No GT Data',
        bounds=bounds,
    )

    draw_pose_panel(
        ax2,
        yolo_frame,
        f'YOLO Prediction\nFrame {frame_idx} | MPJPE: {frame_error:.1f}px',
        'red',
        'No YOLO Detection',
        bounds=bounds,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}_gt_vs_yolo.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def process_selected_frames(model, sequence_name, frame_indices, args, device='cpu'):
    gt_poses_2d, _, _ = load_test_3d_data_from_dataset(sequence_name)
    if gt_poses_2d is None:
        print(f"❌ Failed to load ground truth data for {sequence_name}")
        return None

    frames, valid_indices = load_selected_frames(sequence_name, frame_indices)
    if frames is None:
        print(f"❌ Failed to load selected frames for {sequence_name}")
        return None

    selected_gt = []
    selected_frames = []
    selected_frame_indices = []

    for frame_idx, frame in zip(valid_indices, frames):
        if frame_idx >= len(gt_poses_2d):
            print(f"⚠ Skipping frame {frame_idx}: GT data not available")
            continue
        selected_gt.append(gt_poses_2d[frame_idx])
        selected_frames.append(frame)
        selected_frame_indices.append(frame_idx)

    if not selected_frames:
        print('❌ No valid frame pairs were found')
        return None

    print(f"✓ Processing {len(selected_frames)} selected frames for sequence {sequence_name}")

    yolo_poses_2d, _, performance_metrics = estimate_yolo_poses(
        model,
        selected_frames,
        args.img_size,
        device,
        batch_size=args.batch_size,
    )

    gt_poses_2d_pixel = convert_coordinates_to_pixels(np.array(selected_gt), selected_frames)
    yolo_poses_2d_pixel = convert_coordinates_to_pixels(yolo_poses_2d, selected_frames)
    gt_poses_2d_root_rel = make_root_relative_2d_pixel(gt_poses_2d_pixel, root_joint_idx=14)
    yolo_poses_2d_root_rel = make_root_relative_2d_pixel(yolo_poses_2d_pixel, root_joint_idx=14)

    sequence_output_dir = os.path.join(args.output_dir, sequence_name)
    os.makedirs(sequence_output_dir, exist_ok=True)

    saved_files = []
    for local_idx, frame_idx in enumerate(selected_frame_indices):
        saved_path = save_frame_comparison(
            gt_poses_2d_root_rel[local_idx],
            yolo_poses_2d_root_rel[local_idx],
            sequence_name,
            frame_idx,
            sequence_output_dir,
        )
        saved_files.append(saved_path)
        print(f"✓ Saved comparison for frame {frame_idx} -> {saved_path}")

    print(f"\n✓ Done. Output folder: {os.path.abspath(sequence_output_dir)}")
    print(f"✓ Processed frames: {selected_frame_indices}")
    print(f"✓ Mean inference time: {performance_metrics['mean_inference_time'] * 1000:.2f} ms")
    print(f"✓ FPS: {performance_metrics['fps']:.2f}")

    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Save GT vs YOLO comparisons for selected frames only')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence to compare (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--frames', type=int, nargs='+', required=True, help='Frame indices to process (0-based, space-separated)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--output-dir', type=str, default='comparison_selected_frames', help='Directory to save output images')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size for YOLO inference')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for YOLO inference (default: 32 on CUDA, 16 on CPU)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error('--batch-size must be a positive integer')

    print('🎯 Selected-frame GT vs YOLO comparison')
    print('=' * 80)
    print(f'Sequence: {args.sequence}')
    print(f'Frames: {args.frames}')
    print(f'Model: {args.model_path}')
    print(f'Output dir: {args.output_dir}')
    print(f'Input size: {args.img_size}')
    if args.batch_size is not None:
        print(f'Inference batch size (override): {args.batch_size}')
    else:
        print('Inference batch size: auto (32 on CUDA, 16 on CPU)')
    print('=' * 80)

    if not os.path.exists(args.model_path):
        print(f'❌ Model not found: {args.model_path}')
        return

    if args.device == 'auto':
        device, gpu_available = check_gpu_availability()
    else:
        device = args.device
        gpu_available = device.startswith('cuda') and torch.cuda.is_available()

    print(f'🤖 Loading YOLO model from {args.model_path}...')
    try:
        model = YOLO(args.model_path)
        if gpu_available:
            print(f'📦 Moving model to {device}...')
            model.to(device)
        print('✓ YOLO model loaded successfully')
    except Exception as e:
        print(f'❌ Error loading YOLO model: {e}')
        return

    try:
        process_selected_frames(model, args.sequence, args.frames, args, device)
    finally:
        if gpu_available:
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()