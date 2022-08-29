#!/bin/bash

num_gpus=${1:-4}
workload_dir="/mlcommons_training/image_segmentation/pytorch"

if [[ ! -d "${workload_dir}/results" ]]
then
    mkdir "${workload_dir}/results"
fi

if [[ ! -d "${workload_dir}/ckpts" ]]
then
    mkdir "${workload_dir}/ckpts"
fi


docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v /data/kits19/preprocessed_data/:/data \
	-v ${workload_dir}/results:/results \
	-v ${workload_dir}/ckpts:/ckpts \
	unet3d:latest /bin/bash run_and_time.sh 1 $num_gpus
