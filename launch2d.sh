#!/bin/bash
#
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=testingTCPFormerMediaPipe    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)
python3 data/preprocess/MediaPipeTest/compare2d.py \
    --config configs/mpi/TCPFormer_mpi_27.yaml \
    --sequence-name TS1 \
    --checkpoint checkpoint_mpi \
    --max-samples 500
