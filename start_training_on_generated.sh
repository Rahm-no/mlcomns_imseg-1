#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )
mkdir -p /raid/data/imseg/run_output

NUM_GPUS=${1:-4}
CONTAINER_NAME=${2:train_imseg}
BATCH_SIZE=${3:-2}
DOCKER_IMAGE=${4:-"unet3d:loic"}
NUM_WORKERS=${5:-8}
SKIP_STEP_7=${6:-""}

DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]
then
	DOCKER_MEMORY_PARAM="-m ${DOCKER_MEMORY}g"
fi

docker run --ipc=host --name=$CONTAINER_NAME -it --rm --runtime=nvidia $DOCKER_MEMORY_PARAM \
	-v /raid/data/imseg/29gb-gen:/data \
	-v /raid/data/imseg/run_output:/results \
	-v /raid/data/imseg/run_output:/ckpts \
	$DOCKER_IMAGE /bin/bash run_and_time.sh 1 $NUM_GPUS $BATCH_SIZE $NUM_WORKERS $SKIP_STEP_7 
	