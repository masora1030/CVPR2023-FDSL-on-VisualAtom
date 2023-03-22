#!/bin/bash
cd /home/acc12015ij/CVPR2023-FDSL-on-VisualAtom

# model size
MODEL=base
# initial learning rate
LR=1.0e-3
# name of dataset
DATA_NAME=VisualAtom
# num of classes
CLASSES=21000
# num of epochs
EPOCHS=90
# path to train dataset
SOURCE_DATASET=/PATH/TO/VisualAtom21000
# output dir path
OUT_DIR=./output/pretrain
# num of GPUs
NGPUS=128
# num of processes per node
NPERNODE=4
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64
# global mini-batch size
BATCH_SIZE=$(($NGPUS*$LOCAL_BS))

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"

PYTHONUNBUFFERED=1
PYTHONWARNINGS="ignore"

mpirun -npernode $NPERNODE -np $NGPUS \
python pretrain.py ${SOURCE_DATASET} \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_${DATA_NAME}${CLASSES}_${LR}_${BATCH_SIZE} \
    --input-size 3 224 224 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --min-lr 1.0e-5 --warmup-lr 1.0e-6 --warmup-iter 5000 --cooldown-epochs 0 \
    --batch-size ${LOCAL_BS} --opt adamw --num-classes ${CLASSES} \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 16 --pin-mem --eval-metric loss \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --amp \
    --log-wandb
