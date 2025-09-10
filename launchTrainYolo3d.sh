#!/bin/bash
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=Yolo3DTrain
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Starting YOLO 3D training on MPI-INF-3DHP"

cd data/preprocess/Yolov11

python train_3d.py \
    --force-reprocess \
    --base-path /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --annotations-path ../../motion3d/data_train_3dhp.npz \
    --output-path /nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo_3D \
    --epochs 200 \
    --batch-size 8 \
    --img-size 640 \
    --lr 0.001 \
    --device 0 \
    --workers 16 \
    --use-wandb

echo "3D YOLO training completed!"