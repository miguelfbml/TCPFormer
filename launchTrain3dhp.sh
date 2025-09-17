#!/bin/bash
#
#SBATCH --partition=gpu_min12gb     # Reserved partition
#SBATCH --qos=gpu_min12gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=trainTCPFormer3dhp    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Starting training job for TCPFormer on MPI-INF-3DHP"

# Set wandb API key if needed
# export WANDB_API_KEY=your_api_key_here

# Commands / scripts to run (e.g., python3 train.py)
#python3 train_3dhp.py \
#    --config configs/mpi/TCPFormer_mpi_27.yaml \
#    --new-checkpoint checkpoint_mpi \
#    --use-wandb \
#    --wandb-name "TCPFormer_MPI_3DHP_training_YOLOv11"


python3 train_3dhp.py --config configs/mpi/TCPFormer_mpi_27.yaml --checkpoint checkpoint_mpi --checkpoint-file TCPFormer_mpi_27.pth.tr --new-checkpoint checkpoint_mpi --resume --use-wandb --wandb-name "TCPFormer_MPI_3DHP_resume_training_YOLOv11"