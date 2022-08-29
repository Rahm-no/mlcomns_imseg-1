#!/bin/bash

num_gpus=${1:-4}
exp_name=$2
data_path=$3
workload_dir="/dl-bench/ruoyudeng/mlcomns_imseg"

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
docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v ${data_path}:/data \
	-v ${workload_dir}/results/${exp_name}/results:/results \
	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
	unet3d:tuning /bin/bash run_and_time.sh 1 $num_gpus

