#!/bin/bash

mkdir -p /raid/data/imseg/run_output

NUM_GPUS=${1:-4}
CONTAINER_NAME=${2:train_imseg}
BATCH_SIZE=${3:-2}
DOCKER_MEMORY=${4:-""}
NUM_WORKERS=${4:-1}

DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]
then
	DOCKER_MEMORY_PARAM="-m ${DOCKER_MEMORY}g"
fi

docker run --ipc=host --name=$CONTAINER_NAME -it --rm --runtime=nvidia $DOCKER_MEMORY_PARAM \
	-v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
	-v /raid/data/unet/200gb_dataset_copied:/data \
	-v /raid/data/imseg/run_output:/results \
	-v /raid/data/imseg/run_output:/ckpts \
	unet3d:loic /bin/bash run_and_time.sh 1 $NUM_GPUS $BATCH_SIZE $NUM_WORKERS | tee /raid/data/imseg/run_output/train.log
	