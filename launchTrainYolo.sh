#!/bin/bash
#SBATCH --partition=gpu_min32gb     # Reserved partition
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=YoloTrain_Enhanced
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Starting Enhanced YOLO training on MPI-INF-3DHP for superior keypoint accuracy"

cd data/preprocess/Yolov11

python train.py \
    --base-path /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --annotations-path ../../motion3d/data_train_3dhp.npz \
    --epochs 100 \
    --batch-size 4 \
    --img-size 1280 \
    --lr auto \
    --device 0 \
    --workers 8 \
    --patience 20 \
    --cache disk \
    --pose-loss-weight 17.0 \
    --kobj-loss-weight 2.5 \
    --box-loss-weight 7.5 \
    --cls-loss-weight 0.5 \
    --dfl-loss-weight 1.5 \
    --use-wandb \
    --wandb-project YOLO_MPI_3DHP_Enhanced_Keypoints \
    --train-only