#!/bin/bash

num_gpus=${1:-4}
exp_name=$2
workload_dir="/dl-bench/ruoyudeng/mlcomns_imseg"

docker run --ipc=host --name=training -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v /data/kits19/preprocessed_data/:/data \
	# new store results path 
	-v ${workload_dir}/results/${exp_name}/results:/results \ 
	# -v ${workload_dir}/results:/results \ 
	# new store ckpts path
	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
	# -v ${workload_dir}/ckpts:/ckpts \ 
	unet3d:latest /bin/bash run_and_time.sh 1 $num_gpus
