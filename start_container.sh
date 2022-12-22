#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )

num_gpus=${1:-4}

mkdir -p ${SCRIPT_DIR}/results
mkdir -p ${SCRIPT_DIR}/ckpts


docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia -v /raid/data/imseg/raw-data/kits19/data/:/raw_data -v /raid/data/imseg/preproc-data/:/data -v ${SCRIPT_DIR}/results:/results -v ${SCRIPT_DIR}/ckpts:/ckpts unet3d:latest /bin/bash
