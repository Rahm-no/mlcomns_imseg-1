#!/bin/bash

num_gpus=${1:-4}
exp_name=$2
data_path=$3
workload_dir="/dl-bench/ruoyudeng/mlcomns_imseg"

# we can change -v /data/kits19/preprocessed_data/
docker run --ipc=host --name=training -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v ${data_path}:/data \
	-v ${workload_dir}/results/${exp_name}/results:/results \
	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
	unet3d:latest /bin/bash run_and_time.sh 1 $num_gpus

