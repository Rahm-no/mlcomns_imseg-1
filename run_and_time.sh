#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5> <num_gpus>

SEED=${1:--1} # $SEED is $1 (the first argument passed) OR a int 1 if there is no first argument given
ddplaunch=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")

NUM_GPUS=${2:-1}
MAX_EPOCHS=5
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=25
EVALUATE_EVERY=2
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=10
DATASET_DIR="/data"
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
SAVE_CKPT_PATH="/ckpts"

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
  python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

  python $ddplaunch --nnode=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} main.py \
    --data_dir ${DATASET_DIR} \
    --epochs ${MAX_EPOCHS} \
    --evaluate_every ${EVALUATE_EVERY} \
    --start_eval_at ${START_EVAL_AT} \
    --quality_threshold ${QUALITY_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    --optimizer sgd \
    --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --seed ${SEED} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
    --save_ckpt_path ${SAVE_CKPT_PATH} \
    --num_workers 0
    # my args (need to change in the container, not in here)
    # --oversampling 0.1

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi