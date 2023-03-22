#!/bin/bash
cd /home/acc12015ij/CVPR2023-FDSL-on-VisualAtom

# name of dataset
DATASET_NAME=CIFAR10
# path to train dataset
DATASET_TRAIN_ROOT=/PATH/TO/CIFAR10/TRAIN
# num of images in train dataset
DATASET_TRAIN_IMGS=50000
# path to valid dataset
DATASET_VAL_ROOT=/PATH/TO/CIFAR10/VAL
# num of images in valid dataset
DATASET_VAL_IMGS=10000
# num of classes in dataset
DATASET_CLASSES=10
# model size
MODEL=base
# num of GPUs
NGPUS=8
# num of processes per node
NPERNODE=4
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=96
# global mini-batch size
BATCH_SIZE=$(($NGPUS*$LOCAL_BS))
# name of target pre-trained model
CHECK_POINT_NAME=YOUR_CHECK_POINT_NAME
# path to checkpoint of target pre-trained model
CHECK_POINT_PATH=/PATH/TO/CHECKPOINT/MODEL
# output dir path
OUT_DIR=./output/finetune

# setting place in wandb
LOGGER_ENTITY=YOUR_WANDB_ENTITY_NAME
LOGGER_PROJECT=YOUR_WANDB_PROJECT_NAME
LOGGER_GROUP=finetune
LOGGER_EXPERIMENT=finetune_deit_${MODEL}_${DATASET_NAME}_batch${BATCH_SIZE}_from_${CHECK_POINT_NAME}

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"

PYTHONUNBUFFERED=1
PYTHONWARNINGS="ignore"

mpiexec -npernode ${NPERNODE} -np ${NGPUS} python -B finetune.py \
    data=colorimagefolder \
    data.baseinfo.name=${DATASET_NAME} \
    data.trainset.root=${DATASET_TRAIN_ROOT} \
    data.baseinfo.train_imgs=${DATASET_TRAIN_IMGS} \
    data.valset.root=${DATASET_VAL_ROOT} \
    data.baseinfo.val_imgs=${DATASET_VAL_IMGS} \
    data.baseinfo.num_classes=${DATASET_CLASSES} \
    data.loader.batch_size=${LOCAL_BS} \
    logger.entity=${LOGGER_ENTITY} \
    logger.project=${LOGGER_PROJECT} \
    logger.group=${LOGGER_GROUP} \
    logger.experiment=${LOGGER_EXPERIMENT} \
    logger.save_epoch_freq=100 \
    ckpt=${CHECK_POINT_PATH} \
    model=vit \
    model.arch.model_name=vit_${MODEL}_patch16_224 \
    model.optim.optimizer_name=sgd \
    model.optim.learning_rate=0.01 \
    model.optim.weight_decay=1.0e-4 \
    model.scheduler.args.warmup_epochs=10 \
    epochs=1000 \
    mode=finetune \
    output_dir=${OUT_DIR}
