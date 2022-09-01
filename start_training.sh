#!/bin/bash

num_gpus=${1:-4}
exp_name=$2
data_path=$3
dataset_size=$4
workload_dir="/dl-bench/ruoyudeng/mlcomns_imseg"

if [ $# -lt 4 ]
	then
		echo "Usage: $0 <num_gpus> <experiment_name> <data_path_500GB> <dataset_size>"
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

# workload_dir="/dl-bench/ruoyudeng/mlcomns_imseg"
# data_path="/data/kits19/preprocessed_data"
# # data_path="/raid/data/unet/original_dataset/preprocessed_data_500GB"
# num_gpus="8"
# exp_name="8gpu_original_16GB"

# run traces with 16GB dataset
# sudo docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia \
# 	-v /data/kits19/data/:/raw_data \
# 	-v /raid/data/unet/original_dataset/original_dataset_500GB:/source_data \
# 	-v /dl-bench/ruoyudeng/mlcomns_imseg/results:/results \
# 	-v /dl-bench/ruoyudeng/mlcomns_imseg/ckpts:/ckpts \
# 	unet3d:tuning /bin/bash run_and_time.sh 1 8 256

# DOCKER_CMD="docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia \
# 	-v /data/kits19/data/:/raw_data \
# 	-v ${data_path}:/source_data \
# 	-v ${workload_dir}/results/${exp_name}/results:/results \
# 	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
# 	unet3d:tuning /bin/bash run_and_time.sh 1 $num_gpus $dataset_size"

DOCKER_CMD="docker run --ipc=host --name=train_imseg"
if [[ "${dataset_size}" != "16" ]]
then
	DOCKER_CMD="${DOCKER_CMD} -m 256g"
fi

DOCKER_CMD="${DOCKER_CMD} -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v ${data_path}:/source_data \
	-v ${workload_dir}/results/${exp_name}/results:/results \
	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
	unet3d:tuning /bin/bash run_and_time.sh 1 $num_gpus $dataset_size"
exec $DOCKER_CMD


# docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia \
# 	-v /data/kits19/data/:/raw_data \
# 	-v ${data_path}:/source_data \
# 	-v ${workload_dir}/results/${exp_name}/results:/results \
# 	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
# 	unet3d:tuning /bin/bash run_and_time.sh 1 $num_gpus $dataset_size

# ./start_training.sh 8 8gpu_original_16GB_20220830123856 /raid/data/unet/original_dataset/Original_dataset_500GB 16