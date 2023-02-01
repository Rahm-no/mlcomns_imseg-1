#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )
mkdir -p ${SCRIPT_DIR}/output
# mkdir -p /raid/data/imseg/run_output
mkdir -p ${SCRIPT_DIR}/ckpts

NUM_GPUS=${1:-4}
CONTAINER_NAME=${2:train_imseg}
BATCH_SIZE=${3:-1}
DOCKER_IMAGE=${4:-"unet3d:no-step7"}
NUM_WORKERS=${5:-1}
SKIP_STEP_7=${6:-""}

DOCKER_MEMORY=
DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]
then
	DOCKER_MEMORY_PARAM="-m ${DOCKER_MEMORY}g"
fi

docker run --ipc=host --shm-size=1024m --name=$CONTAINER_NAME -it --rm --runtime=nvidia $DOCKER_MEMORY_PARAM \
	-v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
	-v /raid/data/imseg/29gb-npy/:/data \
	-v ${SCRIPT_DIR}/output:/results \
	-v ${SCRIPT_DIR}/ckpts:/ckpts \
	$DOCKER_IMAGE /bin/bash run_and_time.sh 1 $NUM_GPUS $BATCH_SIZE $NUM_WORKERS $SKIP_STEP_7
	# $DOCKER_IMAGE /bin/bash
