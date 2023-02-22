#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )

mkdir -p /raid/data/imseg/run_output

# mkdir -p ${SCRIPT_DIR}/output
# mkdir -p ${SCRIPT_DIR}/ckpts

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:train_imseg} 
LOGGING_DIR=${3:-"$SCRIPT_DIR/output"}
BATCH_SIZE=${4:-4}
DOCKER_IMAGE=${5:-"unet3d:loic"}
NUM_WORKERS=${6:-8}
NUM_EPOCHS=${7:-50}

DOCKER_MEMORY=
DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]
then
	DOCKER_MEMORY_PARAM="-m ${DOCKER_MEMORY}g"
fi

docker run --ipc=host --name=$CONTAINER_NAME -it --rm --runtime=nvidia $DOCKER_MEMORY_PARAM \
	-v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
	-v /raid/data/imseg/29gb-npy/:/data \
	-v ${LOGGING_DIR}:/results \
	-v /raid/data/imseg/run_output:/ckpts \
	$DOCKER_IMAGE /bin/bash run_and_time.sh 1 $NUM_GPUS $BATCH_SIZE $NUM_WORKERS $NUM_EPOCHS 
