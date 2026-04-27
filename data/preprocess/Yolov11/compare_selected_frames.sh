#!/bin/bash
#
#SBATCH --partition=gpu_min11gb     # Reserved partition
#SBATCH --qos=gpu_min11gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=compareSelected  # Job name
#SBATCH --output=slurm_%x.%j.out    # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err     # File containing STDERR output. If ommited, use STDOUT.


echo "Running selected-frame GT vs YOLO comparison"

SEQUENCE=${SEQUENCE:-TS1}
FRAMES=${FRAMES:-"600 1200 1800 2400 3000 3600 4200 4800 5400 6000"}
MODEL_PATH=${MODEL_PATH:-runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-comparison_selected_frames}
IMG_SIZE=${IMG_SIZE:-640}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}

python3 compare_gt_yolo_selected_frames.py \
    --sequence "$SEQUENCE" \
    --frames $FRAMES \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"


SEQUENCE=${SEQUENCE:-TS2}
FRAMES=${FRAMES:-"600 1200 1800 2400 3000 3600 4200 4800 5400 6000"}
MODEL_PATH=${MODEL_PATH:-runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-comparison_selected_frames}
IMG_SIZE=${IMG_SIZE:-640}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}

python3 compare_gt_yolo_selected_frames.py \
    --sequence "$SEQUENCE" \
    --frames $FRAMES \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"


SEQUENCE=${SEQUENCE:-TS2}
FRAMES=${FRAMES:-"600 1200 1800 2400 3000 3600 4200 4800 5400 6000"}
MODEL_PATH=${MODEL_PATH:-runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-comparison_selected_frames}
IMG_SIZE=${IMG_SIZE:-640}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}

python3 compare_gt_yolo_selected_frames.py \
    --sequence "$SEQUENCE" \
    --frames $FRAMES \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"


SEQUENCE=${SEQUENCE:-TS3}
FRAMES=${FRAMES:-"600 1200 1800 2400 3000 3600 4200 4800 5400 5800"}
MODEL_PATH=${MODEL_PATH:-runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-comparison_selected_frames}
IMG_SIZE=${IMG_SIZE:-640}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}

python3 compare_gt_yolo_selected_frames.py \
    --sequence "$SEQUENCE" \
    --frames $FRAMES \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"

SEQUENCE=${SEQUENCE:-TS4}
FRAMES=${FRAMES:-"600 1200 1800 2400 3000 3600 4200 4800 5400 6000"}
MODEL_PATH=${MODEL_PATH:-runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-comparison_selected_frames}
IMG_SIZE=${IMG_SIZE:-640}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}

python3 compare_gt_yolo_selected_frames.py \
    --sequence "$SEQUENCE" \
    --frames $FRAMES \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"


SEQUENCE=${SEQUENCE:-TS5}
FRAMES=${FRAMES:-"30 60 90 120 150 180 210 240 270 300"}
MODEL_PATH=${MODEL_PATH:-runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-comparison_selected_frames}
IMG_SIZE=${IMG_SIZE:-640}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}

python3 compare_gt_yolo_selected_frames.py \
    --sequence "$SEQUENCE" \
    --frames $FRAMES \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"



SEQUENCE=${SEQUENCE:-TS6}
FRAMES=${FRAMES:-"50 100 150 200 250 300 350 400 450 490"}
MODEL_PATH=${MODEL_PATH:-runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-comparison_selected_frames}
IMG_SIZE=${IMG_SIZE:-640}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}

python3 compare_gt_yolo_selected_frames.py \
    --sequence "$SEQUENCE" \
    --frames $FRAMES \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"