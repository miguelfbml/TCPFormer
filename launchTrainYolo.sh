#!/bin/bash
#SBATCH --partition=gpu_min32gb     # Reserved partition
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=YoloTrain
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Starting YOLO training on MPI-INF-3DHP"

cd data/preprocess/Yolov11

python train.py \
    --force-reprocess \
    --base-path /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --annotations-path ../../motion3d/data_train_3dhp.npz \
    --epochs 100 \
    --batch-size 12 \
    --img-size 1280 \
    --lr 0.001 \
    --device 0 \
    --workers 8 \
    --use-wandb