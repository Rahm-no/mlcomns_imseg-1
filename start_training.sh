#!/bin/bash

num_gpus=${1:-4}
container_name=$2
exp_name=$3

workload_dir=$4
data_path=$5
dataset_size=$6
mem_size=$7

# 8 exp_test /raid/data/unet/original_dataset/Original_dataset_500GB 200 256 /dl-bench/ruoyudeng/mlcomns_imseg
if [ $# -lt 6 ]
	then
		echo "Usage: $0 <num_gpus> <experiment_name> <data_path_500GB> <dataset_size> <memory_size> <workload_dir>"
		exit 1
fi

if [[ ! -d "${workload_dir}/results" ]]
then
    mkdir "${workload_dir}/results"
fi

if [[ ! -d "${workload_dir}/ckpts" ]]
then
    mkdir "${workload_dir}/ckpts"
fi

# create my own image to run parameter tunning
# sudo docker build -t unet3d:tuning .

DOCKER_CMD="docker run --ipc=host --name=${container_name}"
if [[ "${mem_size}" != "-1" ]]
then
	DOCKER_CMD="${DOCKER_CMD} -m ${mem_size}g"
fi

DOCKER_CMD="${DOCKER_CMD} -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v ${data_path}:/source_data \
	-v ${workload_dir}/results/${exp_name}/results:/results \
	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
	unet3d:tuning /bin/bash run_and_time.sh 1 $num_gpus $dataset_size"


# echo $DOCKER_CMD
exec $DOCKER_CMD
