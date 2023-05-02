#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )

LOGGING_DIR="$SCRIPT_DIR/output"
CKPT_DIR="$SCRIPT_DIR/ckpts"
DATA_DIR="$SCRIPT_DIR/data"

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:train_unet3d}
DOCKER_IMAGE=${3:-"unet3d:original"}
BATCH_SIZE=${4:-4}
NUM_WORKERS=${5:-1}
NUM_EPOCHS=${6:-50}

docker run --ipc=host --name=$CONTAINER_NAME -it --rm --runtime=nvidia \
	-v ${DATA_DIR}:/data \
	-v ${LOGGING_DIR}:/results \
	-v ${CKPT_DIR}:/ckpts \
	$DOCKER_IMAGE /bin/bash run_and_time.sh 1 $NUM_GPUS $BATCH_SIZE $NUM_WORKERS $NUM_EPOCHS 
