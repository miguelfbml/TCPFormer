#!/bin/bash
#
#SBATCH --partition=gpu_min12gb     # Reserved partition
#SBATCH --qos=gpu_min12gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=trainPreprocess    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Running job in reserved partition"

export CUDA_LAUNCH_BLOCKING=1
# Commands / scripts to run (e.g., python3 train.py)
python3 preprocess_Yolov11_train.py --model-path ../runs/pose/model3/best.pt --output-path ./data_train_3dhp_yolo_all.npz --img-size 640