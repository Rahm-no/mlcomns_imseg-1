#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )

# Change these directories to your values!
CKPT_DIR=""
DATA_DIR=""


NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:train_unet3d}
LOGGING_DIR=${3:-"$SCRIPT_DIR/output"}
DOCKER_IMAGE=${4:-"unet3d:instrumented"}
BATCH_SIZE=${5:-4}
NUM_EPOCHS=${6:-30}

docker run --ipc=host --name=$CONTAINER_NAME -it --rm --runtime=nvidia \
	-v ${DATA_DIR}:/data \
	-v ${LOGGING_DIR}:/results \
	-v ${CKPT_DIR}:/ckpts \
	$DOCKER_IMAGE /bin/bash run_and_time.sh 1 $NUM_GPUS $BATCH_SIZE $NUM_EPOCHS 
	