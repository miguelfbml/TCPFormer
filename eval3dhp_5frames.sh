#!/bin/bash
#
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=testTCPFormer_5    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.
echo "Running job in reserved partition"

python3 train_3dhp.py --eval-only --checkpoint checkpoint_mpi_5 --checkpoint-file best_epoch.pth.tr --config configs/mpi/testing/TCPFormer_mpi_5.yaml
