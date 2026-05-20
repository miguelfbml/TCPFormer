#!/bin/bash
#
#SBATCH --partition=gpu_min24gb     # Reserved partition
#SBATCH --qos=gpu_min24gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=testTCPFormer_27GT    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.
echo "Running job in reserved partition"
python3 train_3dhp.py --eval-only --checkpoint checkpoint_mpi_27GT --checkpoint-file TCPFormer_mpi_27.pth.tr --config configs/mpi/testing/TCPFormer_mpi_27.yaml
