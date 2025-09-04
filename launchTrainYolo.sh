# Convert dataset and train with full monitoring
python train.py --epochs 100 --batch-size 16 --use-wandb

# With custom paths
python train.py \
    --base-path /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --annotations-path ../../motion3d/data_train_3dhp.npz \
    --epochs 100 \
    --batch-size 16 \
    --use-wandb \
    --wandb-project "YOLO_MPI_3DHP_Custom"

# Convert only
python train.py --convert-only

# Train only (if dataset exists)
python train.py --train-only --epochs 50 --use-wandb