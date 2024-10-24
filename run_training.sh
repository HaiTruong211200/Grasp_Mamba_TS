#!/bin/bash

# Set default values
NETWORK="graspmamba"
INPUT_SIZE=224
USE_DEPTH=0
USE_RGB=1
USE_DROPOUT=1
DROPOUT_PROB=0.1
CHANNEL_SIZE=32
IOU_THRESHOLD=0.25
DATASET="grasp-anything"
DATASET_PATH="/home/cds-robotlab-10/master_thesis/grasp-anything/"
SPLIT=0.9
DS_SHUFFLE=0
DS_ROTATE=0.0
NUM_WORKERS=0
BATCH_SIZE=8
EPOCHS=50
BATCHES_PER_EPOCH=1000
OPTIM="adam"
DESCRIPTION=training_grasp_mamba
LOGDIR="logs/"
VIS=0
FORCE_CPU=0
RANDOM_SEED=123
SEEN=1
ADD_FILE_PATH="/home/cds-robotlab-10/master_thesis/grasp-anything/" #Specific for Grasp-Anywhere

# Construct the options string
options=" \
        --network ${NETWORK} \
        --input-size ${INPUT_SIZE} \
        --use-depth ${USE_DEPTH} \
        --use-rgb ${USE_RGB} \
        --use-dropout ${USE_DROPOUT} \
        --dropout-prob ${DROPOUT_PROB} \
        --channel-size ${CHANNEL_SIZE} \
        --iou-threshold ${IOU_THRESHOLD} \
        --dataset ${DATASET} \
        --dataset-path ${DATASET_PATH} \
        --split ${SPLIT} \
        --ds-rotate ${DS_ROTATE} \
        --num-workers ${NUM_WORKERS} \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --batches-per-epoch ${BATCHES_PER_EPOCH} \
        --optim ${OPTIM} \
        --description ${DESCRIPTION} \
        --logdir ${LOGDIR} \
        --random-seed ${RANDOM_SEED} \
        --seen ${SEEN} \
        --add-file-path ${ADD_FILE_PATH} \
        "
# Append boolean flags if they are set to True
if [ ${DS_SHUFFLE} -eq 1 ]; then
    options+=" --ds-shuffle"
fi

if [ ${VIS} -eq 1 ]; then
    options+=" --vis"
fi

if [ ${FORCE_CPU} -eq 1 ]; then
    options+=" --cpu"
fi

python3 train_network.py $options