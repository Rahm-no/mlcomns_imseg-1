#!/bin/bash

num_gpus=${1:-4}
exp_name=$2
data_path=$3
workload_dir="/dl-bench/ruoyudeng/mlcomns_imseg"

# we can change -v /data/kits19/preprocessed_data/

# sudo docker build -t unet3d:trace1 .

# this is just running docker images, but we need to remount everything using `docker build` as an update to old image before we run it again
docker run --ipc=host --name=training -it --rm --runtime=nvidia \
	-v /data/kits19/data/:/raw_data \
	-v ${data_path}:/data \
	-v ${workload_dir}/results/${exp_name}/results:/results \
	-v ${workload_dir}/ckpts/${exp_name}/ckpts:/ckpts \
	unet3d:trace1 /bin/bash run_and_time.sh 1 $num_gpus

