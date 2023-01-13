#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )
mkdir -p ${SCRIPT_DIR}/results
mkdir -p ${SCRIPT_DIR}/ckpts

NUM_GPUS=${1:-4}
CONTAINER_NAME=${2:train_imseg}
BATCH_SIZE=${3:-2}


docker run --ipc=host --name=$CONTAINER_NAME -it --rm --runtime=nvidia \
	-v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
	-v /raid/data/imseg/29gb-npy/:/data \
	-v ${SCRIPT_DIR}/output:/results \
	-v ${SCRIPT_DIR}/ckpts:/ckpts \
	unet3d:loic /bin/bash run_and_time.sh 1 $NUM_GPUS $BATCH_SIZE
