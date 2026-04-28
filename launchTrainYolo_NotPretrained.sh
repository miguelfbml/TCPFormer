#!/bin/bash
#SBATCH --partition=gpu_min80gb     # Reserved partition
#SBATCH --qos=gpu_min80gb
#SBATCH --job-name=YoloTrain_notPretrained
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Starting Not Pretrained YOLO training on MPI-INF-3DHP for superior keypoint accuracy"

cd data/preprocess/Yolov11

python train.py \
    --base-path /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --annotations-path ../../motion3d/data_train_3dhp.npz \
    --epochs 100 \
    --batch-size 8 \
    --img-size 1280 \
    --lr auto \
    --device 0 \
    --workers 8 \
    --patience 20 \
    --cache disk \
    --use-wandb \
    --wandb-project YOLO_MPI_3DHP_NotPretrained_Keypoints \
    --no-pretrained \
    --train-only