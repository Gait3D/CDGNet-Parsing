#!/bin/bash

# CS_PATH='./dataset/LIP'
INPUT_PATH='/your/path/to/input'
BS=1
GPU_IDS='0'
INPUT_SIZE='256,256'
SNAPSHOT_FROM='/your/path/to/model_best.pth'
DATASET='val'
NUM_CLASSES=12
OUTPUT_PATH='/your/path/to/output'
VIS='yes'  # yes or no

CUDA_VISIBLE_DEVICES=1 python inference.py --data-dir ${INPUT_PATH} \
    --gpu ${GPU_IDS} \
    --batch-size ${BS} \
    --input-size ${INPUT_SIZE} \
    --restore-from ${SNAPSHOT_FROM} \
    --dataset ${DATASET} \
    --num-classes ${NUM_CLASSES} \
    --output-path ${OUTPUT_PATH} \
    --vis ${VIS}






