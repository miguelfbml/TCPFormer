#!/bin/bash
#
#SBATCH --partition=gpu_min11gb     # Reserved partition
#SBATCH --qos=gpu_min11gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=testingTimev2    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Running job in reserved partition"


BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}

# Commands / scripts to run (e.g., python3 train.py)
python3 compare_gt_yolo_2d.py --all --model-path runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt --img-size 640 --batch-size "$BATCH_SIZE" --device "$DEVICE"
# python3 compare_gt_yolo_2d.py --all --model-path runs/pose/mpi_yolo11x_pose_enhanced_keypoints/weights/best.pt --img-size 640 --batch-size "$BATCH_SIZE" --device "$DEVICE"