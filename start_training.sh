#!/bin/bash

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "$0")")
mkdir -p "${SCRIPT_DIR}/output"
mkdir -p "${SCRIPT_DIR}/ckpts"

NUM_GPUS=${1:-4}
CONTAINER_NAME=${2:-train_imseg}
BATCH_SIZE=${3:-4}
DOCKER_MEMORY=${4:-""}

DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]; then
    DOCKER_MEMORY_PARAM="-m $((DOCKER_MEMORY * 1024 * 1024 * 1024))"
fi

docker run --ipc=host --name="$CONTAINER_NAME" --rm --runtime=nvidia $DOCKER_MEMORY_PARAM \
   -v /raid/data/unet3d/rawdata_npy/:/data \
   -v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
   -v "${SCRIPT_DIR}/output":/results \
   -v "${SCRIPT_DIR}/ckpts":/ckpts \
 unet3d:rahma /bin/bash run_and_time.sh 1 "$NUM_GPUS" "$BATCH_SIZE"
#-v /raid/data/unet3d/29gb-npy-prepp/:/data \